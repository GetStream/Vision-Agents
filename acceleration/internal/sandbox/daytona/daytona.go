// Package daytona runs Python through Daytona's official Go SDK.
package daytona

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"mime"
	"net/http"
	"os"
	"path"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	sdk "github.com/daytona/clients/sdk-go/pkg/daytona"
	sdkerrors "github.com/daytona/clients/sdk-go/pkg/errors"
	"github.com/daytona/clients/sdk-go/pkg/options"
	"github.com/daytona/clients/sdk-go/pkg/types"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Provider is the sandbox provider's name, as a session asks for it and as its usage is billed.
const Provider = "daytona"

const apiKeyEnvVar = "DAYTONA_API_KEY"
const defaultTimeout = 60 * time.Second
const defaultRunTimeout = 30 * time.Second

// defaultIdleStop is Daytona's own: a sandbox nothing ran in for 15 minutes is stopped.
const defaultIdleStop = 15 * time.Minute

// stopGrace is how long after an idle sandbox was taken to have stopped its state is read, to
// see that Daytona did stop it: Daytona looks for idle sandboxes about once a minute, and a
// stop takes a moment more.
const stopGrace = 2 * time.Minute

// stateTimeout bounds a read of a sandbox's state that nobody is waiting on.
const stateTimeout = 30 * time.Second

// buildTimeout is how long a sandbox whose image has to be built may take to start. Daytona
// keeps a built image, so only the first sandbox from a given setup waits this long.
const buildTimeout = 20 * time.Minute

// pythonVersion is the slim Python image a setup without an image of its own starts from.
const pythonVersion = "3.13"

type Options struct {
	APIKey, APIURL string
	// Target is the Daytona region the sandbox is made in, such as eu. Empty is DAYTONA_TARGET,
	// then the organization's default.
	Target string
	// Deprecated: the official SDK obtains the toolbox address from Daytona.
	ProxyURL            string
	Timeout, RunTimeout time.Duration
	// Config is how the sandbox is built and how long code may run in it. Its timeout
	// wins over RunTimeout.
	Config     sandbox.Config
	HTTPClient *http.Client
	Logger     *slog.Logger
	// Grants are the connections code in the sandbox may call with, through Broker, which
	// is nil without grants.
	Grants []sandbox.Grant
	Broker sandbox.CredentialBroker
	// Labels are put on the sandbox, so it can be found again and told apart: whose it is and
	// what made it. Their values are ids, never names.
	Labels map[string]string
	// Lifecycle is when Daytona stops the sandbox once nothing runs in it, and deletes it once
	// stopped. Zero IdleStop is Daytona's own 15 minutes; zero StoppedDelete leaves a stopped
	// sandbox to Daytona, which never deletes one on its own.
	Lifecycle sandbox.Lifecycle
	// Meter is told what the sandbox reserves as that changes, which is what Daytona bills.
	// Nil tells nobody.
	Meter Meter
	// Slots bound how many sandboxes run at once: one is taken before the sandbox is made or
	// started again, and given back once it stops or is gone. Nil bounds nothing.
	Slots sandbox.Slots
}

// Reserved is what a sandbox holds, which Daytona bills by the second: its vCPUs, memory and
// disk while it runs, its disk alone while it is stopped. The sizes are the ones Daytona
// reported when it made the sandbox, so a config that left one at zero reads as Daytona's
// default.
type Reserved struct {
	ID                    string
	CPU, MemoryGB, DiskGB float64
	// Class is the sort of sandbox Daytona made: container or linux-vm.
	Class string
}

// Meter is told what a sandbox reserves, from when Daytona makes it until it is deleted. Its
// calls are made one at a time, in the order they happened.
type Meter interface {
	// Created is a sandbox Daytona made, which bills from now. One made again after Daytona
	// deleted the last is Created again, under its own id.
	Created(Reserved)
	// Stopped is the sandbox stopped for idling: only its disk bills from now.
	Stopped()
	// Started is a stopped sandbox running again.
	Started()
	// Orphaned is a sandbox whose session closed and that could not be deleted. It bills on,
	// to the same owner, until it is gone.
	Orphaned()
	// Gone is the sandbox deleted, by its session or by Daytona's own timer: nothing bills
	// from now.
	Gone()
}

// unmetered is the Meter of a sandbox nobody bills.
type unmetered struct{}

func (unmetered) Created(Reserved) {}
func (unmetered) Stopped()         {}
func (unmetered) Started()         {}
func (unmetered) Orphaned()        {}
func (unmetered) Gone()            {}

// A cancellable gate serializes creation, execution and deletion. Identity survives a
// failed delete so Close can be retried; a closed sandbox can never create another VM.
type Sandbox struct {
	client              *sdk.Client
	box                 *sdk.Sandbox
	gate                chan struct{}
	closed              atomic.Bool
	timeout, runTimeout time.Duration
	// check bounds the checks before each run: the timeout, but at most defaultTimeout, as a
	// built sandbox's timeout is buildTimeout and a hung check holds the gate, so every other
	// run of the session waits. Shorter in tests.
	check  time.Duration
	config sandbox.Config
	grants []sandbox.Grant
	broker sandbox.CredentialBroker
	// mount is what the broker's Prepare gave the sandbox being created, and files and setup
	// the ones of its files and setup commands not yet done in it.
	mount sandbox.Mount
	files map[string][]byte
	setup []string

	labels    map[string]string
	lifecycle sandbox.Lifecycle
	meter     Meter
	slots     sandbox.Slots
	logger    *slog.Logger
	// slot gives back the slot the sandbox holds while it runs, nil while it holds none. It is
	// held under the gate.
	slot func()
	// What follows is held under the gate. generation counts the sandboxes made, and used is
	// when work in this one last finished, from which it idles. stoppedAt is when it was
	// taken to have stopped for idling, zero while it runs; confirmed says its state was read
	// since. gone says Daytona's own timer has deleted it, and orphaned that its session
	// closed and could not delete it.
	generation int
	used       time.Time
	stoppedAt  time.Time
	confirmed  bool
	gone       bool
	orphaned   bool
	// timer fires at the sandbox's next point in its lifecycle, and epoch tells the timer
	// armed last from one that fired late. grace is stopGrace, shorter in tests.
	timer       *time.Timer
	epoch       int
	grace       time.Duration
	closeClient sync.Once
}

func New(o Options) (*Sandbox, error) {
	if o.APIKey == "" {
		o.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if o.APIKey == "" {
		return nil, stack.Wrap(errors.New("daytona: DAYTONA_API_KEY is required"))
	}
	if o.Timeout <= 0 {
		o.Timeout = defaultTimeout
		if o.Config.Built() {
			o.Timeout = buildTimeout
		}
	}
	if timeout := o.Config.Timeout(); timeout > 0 {
		o.RunTimeout = timeout
	}
	if o.RunTimeout <= 0 {
		o.RunTimeout = defaultRunTimeout
	}
	if o.Lifecycle.IdleStop <= 0 {
		o.Lifecycle.IdleStop = defaultIdleStop
	}
	if o.Meter == nil {
		o.Meter = unmetered{}
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	c, err := sdk.NewClientWithConfig(&types.DaytonaConfig{APIKey: o.APIKey, APIUrl: o.APIURL, Target: o.Target, HTTPClient: o.HTTPClient})
	if err != nil {
		return nil, stack.Wrap(err)
	}
	return &Sandbox{client: c, gate: make(chan struct{}, 1), timeout: o.Timeout, runTimeout: o.RunTimeout, config: o.Config,
		check:  min(o.Timeout, defaultTimeout),
		grants: o.Grants, broker: o.Broker, labels: maps.Clone(o.Labels), lifecycle: o.Lifecycle, meter: o.Meter,
		slots: o.Slots, logger: o.Logger, grace: stopGrace}, nil
}
func Configured() bool { return os.Getenv(apiKeyEnvVar) != "" }

// done ends a piece of work in the sandbox, from which it idles, and lets the next one in.
func (s *Sandbox) done() {
	s.used = time.Now()
	s.arm()
	<-s.gate
}

// Ready makes sure there is a sandbox running, and says which one it is.
func (s *Sandbox) Ready(ctx context.Context) (int, error) {
	select {
	case s.gate <- struct{}{}:
		defer s.done()
	case <-ctx.Done():
		return 0, ctx.Err()
	}
	err := s.use(ctx, func() error { return nil })
	return s.generation, err
}

func (s *Sandbox) Run(ctx context.Context, code string, outputs []string) (sandbox.Result, error) {
	if strings.TrimSpace(code) == "" {
		return sandbox.Result{}, errors.New("daytona: no code to run")
	}
	select {
	case s.gate <- struct{}{}:
		defer s.done()
	case <-ctx.Done():
		return sandbox.Result{}, ctx.Err()
	}
	var ran *types.ExecuteResponse
	err := s.use(ctx, func() error {
		if s.broker != nil {
			if err := s.broker.BeforeRun(ctx); err != nil {
				return err
			}
		}
		// The checks are bounded on their own, so they leave the code its whole run timeout.
		check, end := context.WithTimeout(ctx, s.check)
		defer end()
		for _, command := range s.mount.EachRun {
			checked, err := s.box.Process.ExecuteCommand(check, "sh -c "+quoted(command), options.WithCommandEnv(s.mount.RunEnv))
			if err != nil {
				return stack.Wrap(err)
			}
			if checked.ExitCode != 0 {
				return stack.Wrap(fmt.Errorf("daytona: readying the sandbox for a run: exit code %d: %s", checked.ExitCode, checked.Result))
			}
		}
		run, cancel := context.WithTimeout(ctx, s.runTimeout+5*time.Second)
		defer cancel()
		var err error
		ran, err = s.box.Process.CodeRun(run, code, options.WithCodeRunTimeout(s.runTimeout),
			options.WithCodeRunParams(types.CodeRunParams{Env: s.mount.RunEnv}))
		return err
	})
	if err != nil {
		return sandbox.Result{}, err
	}
	fetch, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	result := sandbox.Result{Output: ran.Result, ExitCode: ran.ExitCode}
	for i, wanted := range outputs {
		if i == sandbox.MaxFiles {
			result.Missing = append(result.Missing, outputs[i:]...)
			break
		}
		data, err := s.box.FileSystem.DownloadFile(fetch, wanted, nil)
		if err != nil || len(data) == 0 || len(data) > sandbox.MaxFileBytes {
			result.Missing = append(result.Missing, wanted)
			continue
		}
		result.Files = append(result.Files, sandbox.File{Name: path.Base(wanted), MIME: mediaType(wanted, data), Data: data})
	}
	return result, nil
}

// Write puts a file into the sandbox, making the directories it sits in.
func (s *Sandbox) Write(ctx context.Context, name string, data []byte) error {
	select {
	case s.gate <- struct{}{}:
		defer s.done()
	case <-ctx.Done():
		return ctx.Err()
	}
	return s.use(ctx, func() error {
		if dir := path.Dir(name); dir != "." {
			made, err := s.box.Process.ExecuteCommand(ctx, "mkdir -p "+quoted(dir), options.WithCommandEnv(s.mount.RunEnv))
			if err != nil {
				return stack.Wrap(err)
			}
			if made.ExitCode != 0 {
				return stack.Wrap(fmt.Errorf("daytona: mkdir %s: %s", dir, made.Result))
			}
		}
		return stack.Wrap(s.box.FileSystem.UploadFile(ctx, data, name))
	})
}

// use makes sure there is a sandbox running and calls call in it. A sandbox deleted while in
// use, by anyone but this Sandbox, shows as Daytona not finding it: it is forgotten, its meter
// stopped, another made, and call made once more there. The caller holds the gate.
func (s *Sandbox) use(ctx context.Context, call func() error) error {
	err := s.started(ctx)
	if err == nil {
		err = call()
	}
	if !s.vanished(ctx, err) {
		return err
	}
	if err := s.started(ctx); err != nil {
		return err
	}
	return call()
}

// vanished reports whether err is Daytona not finding the sandbox because it was deleted, as
// a read of its state confirms, and forgets it if so. The caller holds the gate.
func (s *Sandbox) vanished(ctx context.Context, err error) bool {
	if s.box == nil || !errors.Is(err, sdkerrors.ErrNotFound) {
		return false
	}
	id := s.box.ID
	read, end := context.WithTimeout(ctx, stateTimeout)
	defer end()
	readErr := s.box.RefreshData(read)
	deleted := errors.Is(readErr, sdkerrors.ErrNotFound) ||
		readErr == nil && (s.box.State == "destroyed" || s.box.State == "destroying")
	if !deleted {
		return false
	}
	s.logger.Warn("a sandbox in use was deleted by someone else; making another", "sandbox", id, "error", err)
	s.lost()
	return true
}

// started creates the sandbox the first time it is needed, starts it again once Daytona has
// stopped it, makes a new one once Daytona has deleted it, and waits for it to be running.
// The caller holds the gate.
func (s *Sandbox) started(ctx context.Context) error {
	if s.closed.Load() {
		return errors.New("daytona: sandbox is closed")
	}
	if s.box != nil && s.lapsed() {
		if err := s.revive(ctx); err != nil {
			return err
		}
	}
	if s.box == nil {
		if err := s.create(ctx); err != nil {
			return err
		}
	}
	ready, end := context.WithTimeout(ctx, s.timeout)
	defer end()
	if err := s.box.WaitForStart(ready, s.timeout); err != nil {
		return err
	}
	// A file that could not be written is written on the next run, before any code.
	for name, data := range s.files {
		if err := s.box.FileSystem.UploadFile(ready, data, name); err != nil {
			return stack.Wrap(err)
		}
		delete(s.files, name)
	}
	for len(s.setup) > 0 {
		ran, err := s.box.Process.ExecuteCommand(ready, "sh -c "+quoted(s.setup[0]), options.WithCommandEnv(s.mount.RunEnv))
		if err != nil {
			return stack.Wrap(err)
		}
		if ran.ExitCode != 0 {
			return stack.Wrap(fmt.Errorf("daytona: setting the sandbox up: exit code %d: %s", ran.ExitCode, ran.Result))
		}
		s.setup = s.setup[1:]
	}
	return nil
}

// create has Daytona make a sandbox, which bills from now. One made again after Daytona
// deleted the last gets its credentials afresh. The caller holds the gate.
func (s *Sandbox) create(ctx context.Context) error {
	create, end := context.WithTimeout(ctx, s.timeout)
	defer end()
	if err := s.hold(create, false); err != nil {
		return err
	}
	if s.broker != nil {
		if s.generation > 0 {
			if err := s.broker.Close(create); err != nil {
				s.letGo()
				return err
			}
		}
		mount, err := s.broker.Prepare(create, s.grants)
		if err != nil {
			s.letGo()
			return err
		}
		s.mount, s.files, s.setup = mount, mount.Files, mount.Setup
	}
	box, err := s.client.Create(create, s.params(), options.WithWaitForStart(false))
	if err != nil {
		s.letGo()
		if s.broker != nil {
			return errors.Join(err, s.broker.Close(ctx))
		}
		return err
	}
	s.box, s.generation, s.used = box, s.generation+1, time.Now()
	s.stoppedAt, s.confirmed, s.gone = time.Time{}, false, false
	s.meter.Created(reserved(box))
	return nil
}

// reserved is what Daytona says box holds.
func reserved(box *sdk.Sandbox) Reserved {
	class := string(types.SandboxClassContainer)
	if box.SandboxClass != nil && *box.SandboxClass != "" {
		class = string(*box.SandboxClass)
	}
	return Reserved{ID: box.ID, CPU: float64(box.Cpu), MemoryGB: float64(box.Memory), DiskGB: float64(box.Disk), Class: class}
}

// hold takes a slot for the sandbox to run in, unless it holds one or nothing bounds how many
// run at once; running is one Daytona runs already. The caller holds the gate.
func (s *Sandbox) hold(ctx context.Context, running bool) error {
	if s.slots == nil || s.slot != nil {
		return nil
	}
	release, err := s.slots.Take(ctx, running)
	if err != nil {
		return err
	}
	s.slot = release
	return nil
}

// letGo gives back the slot the sandbox held while it ran, if it held one. The caller holds the
// gate.
func (s *Sandbox) letGo() {
	if s.slot != nil {
		s.slot()
		s.slot = nil
	}
}

// lapsed reports whether the sandbox may no longer be as it was left: idle for long enough that
// Daytona may have stopped it, or taken to be stopped or deleted. The caller holds the gate.
func (s *Sandbox) lapsed() bool {
	return !s.stoppedAt.IsZero() || s.gone || time.Since(s.used) >= s.lifecycle.IdleStop
}

// revive reads the state of a sandbox that may have lapsed, starts it again if Daytona stopped
// it, waits for one Daytona is bringing up, and forgets it if Daytona deleted it, so that
// started makes another. Only a started sandbox is running: one in any other state, such as
// error or build_failed, is deleted and forgotten too, its meter stopped rather than billed as
// running. The caller holds the gate.
func (s *Sandbox) revive(ctx context.Context) error {
	read, end := context.WithTimeout(ctx, s.timeout)
	defer end()
	err := s.box.RefreshData(read)
	if errors.Is(err, sdkerrors.ErrNotFound) {
		s.lost()
		return nil
	}
	if err != nil {
		return stack.Wrap(err)
	}
	switch s.box.State {
	case "destroyed", "destroying":
		s.lost()
		return nil
	case "started", "stopping", "stopped", "archiving", "archived", "creating", "starting", "restoring", "resuming",
		"pending_build", "building_snapshot", "pulling_snapshot", "resizing", "snapshotting", "forking":
	default:
		s.discard(read)
		return nil
	}
	// A sandbox taken to have stopped runs again only in a slot of its own; refused one, it
	// stays stopped, and is billed so. One Daytona had not stopped yet runs already, and is
	// given a slot whatever the count.
	waking := s.slot == nil
	if err := s.hold(read, s.box.State == "started"); err != nil {
		return err
	}
	if err := s.wake(read); err != nil {
		if waking {
			s.letGo()
		}
		return err
	}
	switch {
	case s.gone:
		// Daytona had not yet deleted what its timer said it would: it bills again.
		s.meter.Created(reserved(s.box))
	case !s.stoppedAt.IsZero():
		s.meter.Started()
	}
	s.stoppedAt, s.confirmed, s.gone = time.Time{}, false, false
	return nil
}

// wake has a sandbox in a state code can run in started: one stopping waited for and started,
// one stopped or archived started, one Daytona is bringing up waited for. The caller holds
// the gate.
func (s *Sandbox) wake(ctx context.Context) error {
	switch s.box.State {
	case "started":
	case "stopping":
		if err := s.box.WaitForStop(ctx, s.timeout); err != nil {
			return stack.Wrap(err)
		}
		fallthrough
	case "stopped", "archiving", "archived":
		if err := s.box.StartWithTimeout(ctx, s.timeout); err != nil {
			return stack.Wrap(err)
		}
	default:
		if err := s.box.WaitForStart(ctx, s.timeout); err != nil {
			return stack.Wrap(err)
		}
	}
	return nil
}

// discard deletes a sandbox in a state code cannot run in, and forgets it. One Daytona would not
// delete is left to the sweep, which deletes it once its session has ended. The caller holds
// the gate.
func (s *Sandbox) discard(ctx context.Context) {
	id, state := s.box.ID, string(s.box.State)
	err := s.box.Delete(ctx)
	if err != nil && !errors.Is(err, sdkerrors.ErrNotFound) {
		s.logger.Warn("could not delete a sandbox in a state it cannot run code in; making another all the same",
			"sandbox", id, "state", state, "error", err)
	} else {
		s.logger.Warn("deleted a sandbox in a state it cannot run code in; making another", "sandbox", id, "state", state)
	}
	s.lost()
}

// lost forgets a sandbox Daytona has deleted, so the next one is made. The caller holds the
// gate.
func (s *Sandbox) lost() {
	s.letGo()
	if !s.gone {
		s.meter.Gone()
	}
	s.box = nil
	s.stoppedAt, s.confirmed, s.gone = time.Time{}, false, false
}

// arm sets the timer for the sandbox's next point in its lifecycle: where Daytona stops it for
// idling, where its stopped state is read to see that it did, and where Daytona deletes it.
// The caller holds the gate.
func (s *Sandbox) arm() {
	s.epoch++
	if s.timer != nil {
		s.timer.Stop()
	}
	if s.box == nil || s.gone {
		return
	}
	var at time.Time
	deletes := s.lifecycle.StoppedDelete > 0
	switch {
	case s.stoppedAt.IsZero():
		at = s.used.Add(s.lifecycle.IdleStop)
	case !s.confirmed:
		at = s.stoppedAt.Add(s.grace)
		if deletes && s.lifecycle.StoppedDelete < s.grace {
			at = s.stoppedAt.Add(s.lifecycle.StoppedDelete)
		}
	case deletes:
		at = s.stoppedAt.Add(s.lifecycle.StoppedDelete)
	default:
		return
	}
	epoch := s.epoch
	s.timer = time.AfterFunc(time.Until(at), func() { s.lapse(epoch) })
}

// lapse moves the sandbox along its lifecycle when its timer fires. Work that holds the gate
// arms the timer again once it is done, so a timer that finds the gate held does nothing.
func (s *Sandbox) lapse(epoch int) {
	select {
	case s.gate <- struct{}{}:
	default:
		return
	}
	defer func() {
		s.arm()
		<-s.gate
	}()
	if epoch != s.epoch || s.box == nil || s.gone {
		return
	}
	now := time.Now()
	switch {
	case s.stoppedAt.IsZero():
		if now.Sub(s.used) < s.lifecycle.IdleStop {
			return
		}
		if s.closed.Load() && s.retryDelete() {
			return
		}
		// The slot is given back before the stop is reported, so whoever reads the stop
		// finds the slot free rather than racing its release.
		s.stoppedAt = now
		s.letGo()
		s.meter.Stopped()
	case s.lifecycle.StoppedDelete > 0 && now.Sub(s.stoppedAt) >= s.lifecycle.StoppedDelete:
		s.gone = true
		s.letGo()
		s.meter.Gone()
		if s.closed.Load() {
			s.box = nil
			s.released(context.Background())
		}
	case !s.confirmed && now.Sub(s.stoppedAt) >= s.grace:
		s.confirmed = true
		s.confirm()
	}
}

// retryDelete deletes a sandbox its session could not, and reports whether it now has. The
// caller holds the gate.
func (s *Sandbox) retryDelete() bool {
	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()
	if err := s.remove(ctx); err != nil {
		s.logger.Warn("could not delete a sandbox whose session closed; Daytona deletes it once it has stopped for long enough",
			"sandbox", s.box.ID, "error", err)
		return false
	}
	s.released(ctx)
	return true
}

// confirm reads the state of a sandbox taken to have stopped for idling, and says so when
// Daytona has not stopped it: the meter bills it as stopped all the same, and the day's
// reconciliation with Daytona's records finds the difference. The caller holds the gate.
func (s *Sandbox) confirm() {
	read, cancel := context.WithTimeout(context.Background(), stateTimeout)
	defer cancel()
	if err := s.box.RefreshData(read); err != nil {
		if !errors.Is(err, sdkerrors.ErrNotFound) {
			s.logger.Warn("could not read the state of a sandbox that should have stopped", "sandbox", s.box.ID, "error", err)
		}
		return
	}
	switch s.box.State {
	case "stopping", "stopped", "archiving", "archived", "destroying", "destroyed":
		return
	}
	s.logger.Warn("Daytona has not stopped a sandbox that is billed as stopped", "sandbox", s.box.ID,
		"billed_as", "stopped", "state", string(s.box.State), "idle_stop", s.lifecycle.IdleStop)
}

// remove deletes the sandbox, one Daytona no longer has counting as deleted, and tells the
// meter it is gone. The caller holds the gate.
func (s *Sandbox) remove(ctx context.Context) error {
	if err := s.box.Delete(ctx); err != nil && !errors.Is(err, sdkerrors.ErrNotFound) {
		return err
	}
	s.letGo()
	if !s.gone {
		s.meter.Gone()
	}
	s.box = nil
	s.stoppedAt, s.confirmed, s.gone = time.Time{}, false, false
	return nil
}

// release closes the client once nothing more is asked of Daytona. Only the first call closes
// it.
func (s *Sandbox) release(ctx context.Context) (err error) {
	s.closeClient.Do(func() { err = s.client.Close(ctx) })
	return err
}

// released is release for a sandbox nobody is waiting on, which logs what failed.
func (s *Sandbox) released(ctx context.Context) {
	if err := s.release(ctx); err != nil {
		s.logger.Warn("could not close the Daytona client", "error", err)
	}
}

// minutes is a lifecycle interval in Daytona's whole minutes, rounded up.
func minutes(d time.Duration) int {
	return int((d + time.Minute - 1) / time.Minute)
}

// quoted is a path or a script as one shell word.
func quoted(word string) string {
	return "'" + strings.ReplaceAll(word, "'", `'\''`) + "'"
}

// params are what the sandbox is created from: Daytona's own Python sandbox, or an image
// built from the config's base and setup.
func (s *Sandbox) params() any {
	domains := s.domains()
	base := types.SandboxBaseParams{Language: types.CodeLanguagePython,
		NetworkBlockAll: s.config.Isolated() && len(domains) == 0 && s.mount.Proxy == nil, Secrets: s.mount.Secrets, EnvVars: s.mount.Env,
		Labels: maps.Clone(s.labels)}
	// Daytona's timers are the ones the meter keeps. Archiving stays Daytona's: it comes a week
	// after a stop, long after the sandbox is deleted.
	idleStop := minutes(s.lifecycle.IdleStop)
	base.AutoStopInterval = &idleStop
	if s.lifecycle.StoppedDelete > 0 {
		stoppedDelete := minutes(s.lifecycle.StoppedDelete)
		base.AutoDeleteInterval = &stoppedDelete
	}
	if len(domains) > 0 {
		joined := strings.Join(domains, ",")
		base.DomainAllowList = &joined
	}
	cidrs := s.config.NetworkAllowList
	if s.mount.Proxy != nil {
		// Daytona's network allow list alone limits every TCP and UDP connection to its ranges and
		// passes TLS as it is. A domain allow list or an OutboundProxyUrl would bring its netleash
		// in, which under a domain list ends TLS itself and refuses the proxy's certificate for a
		// granted host (live probe and staging run, 2026-10-22).
		cidrs = []string{s.mount.Proxy.Hostname() + "/32"}
	}
	if len(cidrs) > 0 {
		joined := strings.Join(cidrs, ",")
		base.NetworkAllowList = &joined
	}
	if !s.config.Built() {
		return types.SnapshotParams{SandboxBaseParams: base}
	}
	version := pythonVersion
	image := sdk.DebianSlim(&version)
	if s.config.Image != "" {
		image = sdk.Base(s.config.Image)
	}
	for _, command := range s.config.Setup {
		image = image.Run(command)
	}
	params := types.ImageParams{SandboxBaseParams: base, Image: image}
	if s.config.CPU > 0 || s.config.MemoryGB > 0 || s.config.DiskGB > 0 {
		params.Resources = &types.Resources{CPU: s.config.CPU, Memory: s.config.MemoryGB, Disk: s.config.DiskGB}
	}
	return params
}

// mediaType names what a file is from its extension, and from its first bytes when the
// extension says nothing.
func mediaType(name string, data []byte) string {
	if byExtension := mime.TypeByExtension(path.Ext(name)); byExtension != "" {
		return strings.SplitN(byExtension, ";", 2)[0]
	}
	return strings.SplitN(http.DetectContentType(data), ";", 2)[0]
}

// Close deletes the sandbox. One Daytona would not delete is an orphan: it keeps its identity,
// so Close can be tried again, and its meter, since Daytona still bills it. Its timer then
// tries the delete again once it has idled, and failing that it is billed as stopped until
// Daytona's own timer deletes it. A Close that gives up waiting for work still in the sandbox
// leaves the rest of the close to be done once the work is done.
func (s *Sandbox) Close() error {
	s.closed.Store(true)
	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()
	select {
	case s.gate <- struct{}{}:
		defer func() {
			s.arm()
			<-s.gate
		}()
		return s.shut(ctx)
	case <-ctx.Done():
	}
	// A run still holds the sandbox, which stays until the run ends, but its credential goes
	// now: Daytona's secret would wait for the sweep, the proxy's record for its expiry.
	go s.closeLater()
	if s.broker == nil {
		return ctx.Err()
	}
	released, end := context.WithTimeout(context.Background(), s.timeout)
	defer end()
	return errors.Join(ctx.Err(), s.broker.Close(released))
}

// closeLater waits for the work a Close gave up on to end, then closes as that Close would have:
// the sandbox goes as soon as it is free rather than once it idles, and one Daytona would not
// delete is an orphan, its timer's to delete.
func (s *Sandbox) closeLater() {
	s.gate <- struct{}{}
	defer func() {
		s.arm()
		<-s.gate
	}()
	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()
	if err := s.shut(ctx); err != nil {
		s.logger.Warn("could not close a sandbox once the run its Close gave up on ended", "error", err)
	}
}

// shut deletes the sandbox and closes the client, or leaves an orphan. The caller holds the gate.
func (s *Sandbox) shut(ctx context.Context) error {
	// The secrets go first: a sandbox Daytona would not delete keeps no credential.
	var released error
	if s.broker != nil {
		released = s.broker.Close(ctx)
	}
	if s.box == nil {
		return errors.Join(released, s.release(ctx))
	}
	if err := s.remove(ctx); err != nil {
		if !s.orphaned {
			s.orphaned = true
			s.meter.Orphaned()
		}
		return errors.Join(released, err)
	}
	return errors.Join(released, s.release(ctx))
}

// Grants are the connections code in the sandbox may call with.
func (s *Sandbox) Grants() []sandbox.Grant {
	return s.grants
}

// domains are the config's allowed domains and, unless the config opens the whole network or
// lists CIDR ranges, which Daytona takes instead of domains, the hosts of each grant the
// sandbox holds a secret for: a sandbox that may reach nothing else still reaches its grants.
// A sandbox behind a proxy reaches the proxy alone, and the proxy the domains.
func (s *Sandbox) domains() []string {
	if s.mount.Proxy != nil {
		return nil
	}
	domains := slices.Clone(s.config.AllowedDomains)
	if s.config.Network && len(domains) == 0 || len(s.config.NetworkAllowList) > 0 {
		return domains
	}
	for _, grant := range s.grants {
		if _, secret := s.mount.Secrets[grant.Env]; secret {
			for _, host := range grant.Hosts {
				if !slices.Contains(domains, host) {
					domains = append(domains, host)
				}
			}
		}
	}
	return domains
}
