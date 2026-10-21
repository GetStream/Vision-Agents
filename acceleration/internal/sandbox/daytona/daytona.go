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
	"net/netip"
	"os"
	"path"
	"slices"
	"strings"
	"sync/atomic"
	"time"

	sdk "github.com/daytona/clients/sdk-go/pkg/daytona"
	"github.com/daytona/clients/sdk-go/pkg/options"
	"github.com/daytona/clients/sdk-go/pkg/types"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

const apiKeyEnvVar = "DAYTONA_API_KEY"
const defaultTimeout = 60 * time.Second
const defaultRunTimeout = 30 * time.Second

// buildTimeout is how long a sandbox whose image has to be built may take to start. Daytona
// keeps a built image, so only the first sandbox from a given setup waits this long.
const buildTimeout = 20 * time.Minute

// pythonVersion is the slim Python image a setup without an image of its own starts from.
const pythonVersion = "3.13"

type Options struct {
	APIKey, APIURL string
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
}

// A cancellable gate serializes creation, execution and deletion. Identity survives a
// failed delete so Close can be retried; a closed sandbox can never create another VM.
type Sandbox struct {
	client              *sdk.Client
	box                 *sdk.Sandbox
	gate                chan struct{}
	closed              atomic.Bool
	timeout, runTimeout time.Duration
	config              sandbox.Config
	grants              []sandbox.Grant
	broker              sandbox.CredentialBroker
	// mount is what the broker's Prepare gave the sandbox being created, and files the ones
	// of its files not yet written into it.
	mount sandbox.Mount
	files map[string][]byte
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
	c, err := sdk.NewClientWithConfig(&types.DaytonaConfig{APIKey: o.APIKey, APIUrl: o.APIURL, HTTPClient: o.HTTPClient})
	if err != nil {
		return nil, stack.Wrap(err)
	}
	return &Sandbox{client: c, gate: make(chan struct{}, 1), timeout: o.Timeout, runTimeout: o.RunTimeout, config: o.Config,
		grants: o.Grants, broker: o.Broker}, nil
}
func Configured() bool { return os.Getenv(apiKeyEnvVar) != "" }
func (s *Sandbox) Run(ctx context.Context, code string, outputs []string) (sandbox.Result, error) {
	if strings.TrimSpace(code) == "" {
		return sandbox.Result{}, errors.New("daytona: no code to run")
	}
	select {
	case s.gate <- struct{}{}:
		defer func() { <-s.gate }()
	case <-ctx.Done():
		return sandbox.Result{}, ctx.Err()
	}
	if err := s.started(ctx); err != nil {
		return sandbox.Result{}, err
	}
	if s.broker != nil {
		if err := s.broker.BeforeRun(ctx); err != nil {
			return sandbox.Result{}, err
		}
	}
	run, cancel := context.WithTimeout(ctx, s.runTimeout+5*time.Second)
	defer cancel()
	ran, err := s.box.Process.CodeRun(run, code, options.WithCodeRunTimeout(s.runTimeout))
	if err != nil {
		return sandbox.Result{}, err
	}
	result := sandbox.Result{Output: ran.Result, ExitCode: ran.ExitCode}
	for i, wanted := range outputs {
		if i == sandbox.MaxFiles {
			result.Missing = append(result.Missing, outputs[i:]...)
			break
		}
		data, err := s.box.FileSystem.DownloadFile(run, wanted, nil)
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
		defer func() { <-s.gate }()
	case <-ctx.Done():
		return ctx.Err()
	}
	if err := s.started(ctx); err != nil {
		return err
	}
	if dir := path.Dir(name); dir != "." {
		made, err := s.box.Process.ExecuteCommand(ctx, "mkdir -p "+quoted(dir))
		if err != nil {
			return stack.Wrap(err)
		}
		if made.ExitCode != 0 {
			return stack.Wrap(fmt.Errorf("daytona: mkdir %s: %s", dir, made.Result))
		}
	}
	return stack.Wrap(s.box.FileSystem.UploadFile(ctx, data, name))
}

// started creates the sandbox the first time it is needed and waits for it to be running.
// The caller holds the gate.
func (s *Sandbox) started(ctx context.Context) error {
	if s.closed.Load() {
		return errors.New("daytona: sandbox is closed")
	}
	if s.box == nil {
		create, end := context.WithTimeout(ctx, s.timeout)
		defer end()
		if s.broker != nil {
			mount, err := s.broker.Prepare(create, s.grants)
			if err != nil {
				return err
			}
			s.mount, s.files = mount, mount.Files
		}
		box, err := s.client.Create(create, s.params(), options.WithWaitForStart(false))
		if err != nil {
			if s.broker != nil {
				return errors.Join(err, s.broker.Close(ctx))
			}
			return err
		}
		s.box = box
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
	return nil
}

// quoted is a path as one shell word.
func quoted(word string) string {
	return "'" + strings.ReplaceAll(word, "'", `'\''`) + "'"
}

// params are what the sandbox is created from: Daytona's own Python sandbox, or an image
// built from the config's base and setup.
func (s *Sandbox) params() any {
	domains := s.domains()
	base := types.SandboxBaseParams{Language: types.CodeLanguagePython, NetworkBlockAll: s.config.Isolated() && len(domains) == 0,
		Secrets: s.mount.Secrets, EnvVars: s.mount.Env}
	if s.mount.Proxy != nil {
		proxy := s.mount.Proxy.String()
		base.OutboundProxyUrl = &proxy
		if direct := noProxy(s.config, s.grants); direct != "" {
			base.EnvVars = maps.Clone(s.mount.Env)
			base.EnvVars["NO_PROXY"], base.EnvVars["no_proxy"] = direct, direct
		}
	}
	if len(domains) > 0 {
		joined := strings.Join(domains, ",")
		base.DomainAllowList = &joined
	}
	if cidrs := s.ranges(); len(cidrs) > 0 {
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

func (s *Sandbox) Close() error {
	s.closed.Store(true)
	ctx, cancel := context.WithTimeout(context.Background(), s.timeout)
	defer cancel()
	select {
	case s.gate <- struct{}{}:
		defer func() { <-s.gate }()
	case <-ctx.Done():
		return ctx.Err()
	}

	// The secrets go first: a sandbox Daytona would not delete keeps no credential.
	var released error
	if s.broker != nil {
		released = s.broker.Close(ctx)
	}
	if s.box == nil {
		return errors.Join(released, s.client.Close(ctx))
	}
	if err := s.box.Delete(ctx); err != nil {
		return errors.Join(released, err)
	}
	s.box = nil
	return errors.Join(released, s.client.Close(ctx))
}

// Grants are the connections code in the sandbox may call with.
func (s *Sandbox) Grants() []sandbox.Grant {
	return s.grants
}

// domains are the config's allowed domains and, unless the config opens the whole network or
// lists CIDR ranges, which Daytona takes instead of domains, the hosts of each grant the
// sandbox holds a secret for: a sandbox that may reach nothing else still reaches its grants.
// A sandbox behind an egress proxy reaches the proxy and the config's own domains, and its
// grants' hosts only through the proxy.
func (s *Sandbox) domains() []string {
	domains := slices.Clone(s.config.AllowedDomains)
	if s.mount.Proxy != nil && len(s.config.NetworkAllowList) == 0 {
		return slices.Insert(domains, 0, s.mount.Proxy.Hostname())
	}
	if s.config.Network && len(domains) == 0 || len(s.config.NetworkAllowList) > 0 {
		return domains
	}
	for _, grant := range s.grants {
		if _, held := s.mount.Secrets[grant.Env]; held {
			for _, host := range grant.Hosts {
				if !slices.Contains(domains, host) {
					domains = append(domains, host)
				}
			}
		}
	}
	return domains
}

// ranges are the config's CIDR ranges and, behind an egress proxy, the proxy's address: Daytona
// takes one allow list, so the proxy joins the one the config uses. A proxy named by a host
// name cannot, and the session refuses that config.
func (s *Sandbox) ranges() []string {
	ranges := s.config.NetworkAllowList
	if len(ranges) == 0 || s.mount.Proxy == nil {
		return ranges
	}
	address, err := netip.ParseAddr(s.mount.Proxy.Hostname())
	if err != nil {
		return ranges
	}
	return append(slices.Clone(ranges), netip.PrefixFrom(address, address.BitLen()).String())
}

// noProxy is NO_PROXY for a sandbox behind an egress proxy: the config's own domains and
// ranges, which code reaches directly, less each domain whose entry would match a grant's host,
// since a request the proxy never sees keeps its placeholder. curl, Python requests and Go take
// an entry as that name and every name under it, and none takes "*.", so *.pypi.org is
// .pypi.org (https://curl.se/docs/manpage.html#ENVIRONMENT,
// https://pkg.go.dev/golang.org/x/net/http/httpproxy#Config).
func noProxy(config sandbox.Config, grants []sandbox.Grant) string {
	var direct []string
	for _, domain := range config.AllowedDomains {
		domain = strings.TrimPrefix(domain, "*")
		name := strings.ToLower(strings.TrimPrefix(domain, "."))
		if !slices.ContainsFunc(grants, func(grant sandbox.Grant) bool { return matches(name, grant.Hosts) }) {
			direct = append(direct, domain)
		}
	}
	return strings.Join(append(direct, config.NetworkAllowList...), ",")
}

// matches says whether a NO_PROXY entry for name matches a host one of hosts covers: hosts are
// exact names, or *. and a domain, which takes the domain and every name under it.
func matches(name string, hosts []string) bool {
	for _, host := range hosts {
		granted, wildcard := strings.CutPrefix(host, "*.")
		if granted == name || strings.HasSuffix(granted, "."+name) || wildcard && strings.HasSuffix(name, "."+granted) {
			return true
		}
	}
	return false
}
