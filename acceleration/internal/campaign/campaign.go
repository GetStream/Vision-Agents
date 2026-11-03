// Package campaign rings a list of people with one agent.
//
// A campaign is the outbound half of telephony: rather than an agent waiting for a call,
// the router places one, has the conversation, and moves on to the next person. Every
// router runs a Runner, and all of them share one queue in Postgres: a tick claims the
// contacts that are due under a lease, so a campaign's concurrency holds however many
// routers there are, and a router that dies mid-call has its contacts taken over once their
// leases run out. An answered call's outcome is read from its transcript a moment later,
// by whichever router claims it.
package campaign

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// callTimeout bounds one conversation. Nothing else ends a campaign call: the agent is
// not told when the person hangs up, and a slot held forever by a call that is over is a
// campaign that stops halfway through.
const callTimeout = 15 * time.Minute

// lease is how long a claimed contact stays this router's. It outlasts the longest call, so
// only a router that died loses its contacts, and is never renewed.
const lease = callTimeout + 5*time.Minute

// writeTimeout bounds the writes a call makes once it is over, which have to happen on a
// context a shutdown did not cancel.
const writeTimeout = 10 * time.Second

// What a campaign records about a person who asked not to be rung.
const (
	voice          = "voice"
	optOutSource   = "campaign"
	reasonOptedOut = "opted_out"
)

// reasonInvalidPlan fails a contact whose campaign's sequence or hours cannot be read, which
// no retry would change.
const reasonInvalidPlan = "invalid_plan"

// defaultTick is how often a router looks for work when Options leaves it out.
const defaultTick = 2 * time.Second

// maxCalls is how many calls one router holds at once, every one of them a conversation
// this process runs; maxClassifying is how many transcripts one tick reads.
const (
	maxCalls       = 100
	maxClassifying = 20
)

// An answered call's transcript is written a moment after the call ends. It is first read
// classifyDelay after, and an empty one is read again every classifyRetry, maxClassifyTries
// times in all, before the call counts as one nobody spoke on.
const (
	classifyDelay    = 10 * time.Second
	classifyRetry    = 30 * time.Second
	maxClassifyTries = 6
)

// Dial is one call to place: Number is 1, or 2 for the second call of a double dial, and
// SessionID is the session the answered call is held in, chosen before it is placed so a
// router taking the contact over can find the call.
type Dial struct {
	Campaign  store.Campaign
	Contact   store.Contact
	Step      Step
	Number    int
	SessionID string
}

// Dialer places a call and returns once it is over. An error is a dial that could not be
// tried; a refusal is a Dialed.
type Dialer interface {
	Dial(ctx context.Context, dial Dial) (Dialed, error)
}

// Transcripts reads what was said on a call, empty until it is written.
type Transcripts interface {
	CallTranscript(ctx context.Context, customerID, callID string) ([]chatlog.Spoken, error)
}

// Classifier picks the outcome of an answered call: an id of step.Choices(), or "" for none.
type Classifier interface {
	Classify(ctx context.Context, campaign store.Campaign, callID string, step Step, said []chatlog.Spoken) (string, error)
}

// Options configures a Runner. Store, Dialer, Transcripts and Classifier are required.
type Options struct {
	Store       *store.Store
	Dialer      Dialer
	Transcripts Transcripts
	Classifier  Classifier
	// Owner is this router in the leases it takes. Empty is the hostname with a random suffix.
	Owner string
	// Now is the clock. Nil is time.Now.
	Now func() time.Time
	// Tick is how often to look for work. Zero is 2 seconds.
	Tick   time.Duration
	Logger *slog.Logger
}

// Runner places the calls of every running campaign that this router claims.
type Runner struct {
	store       *store.Store
	dialer      Dialer
	transcripts Transcripts
	classifier  Classifier
	owner       string
	now         func() time.Time
	every       time.Duration
	logger      *slog.Logger

	// calls is what calls are placed under. Close cancels it only once the calls in flight
	// outlived callTimeout: a router going down does not hang up on somebody mid-sentence.
	calls       context.Context
	cancelCalls context.CancelFunc

	mu       sync.Mutex
	inFlight int
	stop     context.CancelFunc
	// stopped closes when the ticking ends, at the stop signal or Close. Nil before Start.
	stopped <-chan struct{}
	closed  bool

	ticking sync.WaitGroup
	dialing sync.WaitGroup
}

// New validates the options and returns a Runner. It starts nothing.
func New(options Options) (*Runner, error) {
	switch {
	case options.Store == nil:
		return nil, errors.New("campaign: a database is required")
	case options.Dialer == nil:
		return nil, errors.New("campaign: a dialer is required")
	case options.Transcripts == nil:
		return nil, errors.New("campaign: transcripts are required")
	case options.Classifier == nil:
		return nil, errors.New("campaign: a classifier is required")
	}
	if options.Owner == "" {
		host, err := os.Hostname()
		if err != nil {
			return nil, stack.Wrap(fmt.Errorf("campaign: hostname: %w", err))
		}
		options.Owner = fmt.Sprintf("%s-%08x", host, rand.Uint32())
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	calls, cancelCalls := context.WithCancel(context.Background())
	return &Runner{
		store:       options.Store,
		dialer:      options.Dialer,
		transcripts: options.Transcripts,
		classifier:  options.Classifier,
		owner:       options.Owner,
		now:         options.Now,
		every:       cmp.Or(options.Tick, defaultTick),
		logger:      cmp.Or(options.Logger, slog.Default()),
		calls:       calls,
		cancelCalls: cancelCalls,
	}, nil
}

// Owner is this router in the leases it takes.
func (r *Runner) Owner() string { return r.owner }

// Running is how many contacts this router is ringing or about to ring, a call placed
// before its session exists included, for a drain to wait on.
func (r *Runner) Running() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.inFlight
}

// Start ticks until ctx is cancelled or Close is called.
func (r *Runner) Start(ctx context.Context) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return stack.Wrap(errors.New("campaign: the runner is closed"))
	}
	if r.stop != nil {
		return stack.Wrap(errors.New("campaign: the runner is started already"))
	}
	ticking, stop := context.WithCancel(ctx)
	r.stop = stop
	r.stopped = ticking.Done()
	r.ticking.Add(1)
	go func() {
		defer r.ticking.Done()
		r.loop(ticking)
	}()
	return nil
}

// Close stops claiming work and waits for the calls in flight, cutting them off only once
// callTimeout has passed. What each call ended as is written either way.
func (r *Runner) Close() {
	r.mu.Lock()
	r.closed = true
	stop := r.stop
	r.mu.Unlock()
	if stop != nil {
		stop()
	}
	r.ticking.Wait()

	over := make(chan struct{})
	go func() {
		r.dialing.Wait()
		close(over)
	}()
	deadline := time.NewTimer(callTimeout)
	defer deadline.Stop()
	select {
	case <-over:
	case <-deadline.C:
		r.cancelCalls()
		<-over
	}
	r.cancelCalls()
}

func (r *Runner) loop(ctx context.Context) {
	ticker := time.NewTicker(r.every)
	defer ticker.Stop()
	for {
		r.tick(ctx)
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// tick does one round of the shared queue: starts the campaigns whose time came, settles the
// contacts of routers that died, places the calls that are due, reads the outcomes of answered
// calls, and completes the campaigns with nobody left.
func (r *Runner) tick(ctx context.Context) {
	if err := r.store.StartDueCampaigns(ctx, r.now()); err != nil {
		r.logger.Error("could not start the campaigns that are due", "error", err)
	}
	r.takeOver(ctx)
	r.claim(ctx)
	r.classifyDue(ctx)
	if err := r.store.CompleteCampaigns(ctx, r.now()); err != nil {
		r.logger.Error("could not complete the campaigns with nobody left", "error", err)
	}
}

// claim takes as many due contacts as this router has room for, and rings each.
func (r *Runner) claim(ctx context.Context) {
	r.mu.Lock()
	free := maxCalls - r.inFlight
	if r.closed {
		free = 0
	}
	r.mu.Unlock()
	if free <= 0 {
		return
	}

	claimed, err := r.store.ClaimDueContacts(ctx, r.owner, r.now(), lease, free)
	if err != nil {
		r.logger.Error("could not claim the contacts that are due", "error", err)
		return
	}
	r.mu.Lock()
	r.inFlight += len(claimed)
	r.mu.Unlock()
	for _, one := range claimed {
		r.dialing.Add(1)
		go func() {
			defer r.dialing.Done()
			defer func() {
				r.mu.Lock()
				r.inFlight--
				r.mu.Unlock()
			}()
			r.attempt(r.calls, one)
		}()
	}
}

// attempt rings one claimed contact, if it may still be rung, and settles it.
//
// A write that fails here leaves the contact leased: whoever takes it over once the lease
// runs out settles it from its dials, and never rings it twice.
func (r *Runner) attempt(ctx context.Context, claimed store.Claimed) {
	campaign, contact := claimed.Campaign, claimed.Contact
	sequence, hours, ok := r.plan(ctx, campaign, contact)
	if !ok {
		return
	}
	step := sequence.First()
	zone := hours.Zone(contact.TimeZone, contact.ToNumber)
	unused := store.Settle{StepID: step.ID, Attempt: contact.Attempt}

	// The claim does not check the hours: this is the check that keeps anybody from being
	// rung outside them.
	now := r.now()
	if !hours.Open(now, zone) {
		r.settle(ctx, contact, enrolledAt(hours, zone, now, unused))
		return
	}
	opted, err := r.store.OptedOut(ctx, campaign.CustomerID, contact.ToNumber, voice)
	if err != nil {
		r.logger.Error("could not check a campaign contact's opt-outs", "contact", contact.ID, "error", err)
		return
	}
	if opted {
		unused.Stage, unused.Reason = store.Failed, reasonOptedOut
		r.settle(ctx, contact, unused)
		return
	}
	current, ringing, err := r.mayRing(ctx, campaign, contact)
	if err != nil {
		r.logger.Error("could not read a campaign before ringing", "campaign", campaign.ID, "error", err)
		return
	}
	if !ringing {
		unused.Stage, unused.NextAttemptAt = store.Enrolled, &now
		r.settle(ctx, contact, unused)
		return
	}

	// A router stopping ends its sessions, so a call it placed now would ring with no agent.
	if r.stopping() {
		unused.Stage, unused.NextAttemptAt = store.Enrolled, &now
		r.settle(ctx, contact, unused)
		return
	}

	started := r.now()
	for number := 1; ; number++ {
		dialed, err := r.ring(ctx, current, contact, step, number)
		if err != nil {
			r.logger.Error("could not record a campaign dial", "contact", contact.ID, "error", err)
			return
		}
		if dialed.Result == Paused {
			r.pause(ctx, current, dialed.Reason)
		}
		settle, again := afterDial(sequence, hours, zone, contact, number, dialed, started, r.now())
		if again {
			// The first dial took a while to ring out, and the campaign may have been paused
			// or stopped meanwhile: nobody is rung again then, but the first dial rang them.
			var ringing bool
			current, ringing, err = r.mayRing(ctx, current, contact)
			if err != nil {
				r.logger.Error("could not read a campaign before dialing again", "campaign", campaign.ID, "error", err)
				return
			}
			if !ringing {
				settle, again = afterDial(sequence, hours, zone, contact, 2, Dialed{Result: Paused}, started, r.now())
			} else if r.stopping() {
				// Dial 1 may have come back unanswered only because its session was ended.
				settle, again = afterDial(sequence, hours, zone, contact, 2, Dialed{Result: Canceled}, started, r.now())
			}
		}
		if !again {
			r.settle(ctx, contact, settle)
			return
		}
	}
}

// mayRing reads campaign and contact again, since either may have been paused or stopped
// since the contact was claimed. A stopped contact settled as enrolled ends stopped.
func (r *Runner) mayRing(ctx context.Context, campaign store.Campaign, contact store.Contact) (store.Campaign, bool, error) {
	current, err := r.store.Campaign(ctx, campaign.CustomerID, campaign.ID)
	if err != nil {
		return store.Campaign{}, false, err
	}
	// StopContact marks a contact in a call with the reason it would end with.
	marked, _, err := r.store.CampaignContact(ctx, campaign.ID, contact.ID)
	if err != nil {
		return store.Campaign{}, false, err
	}
	return current, current.State == store.Running && marked.Reason != store.ContactStopped, nil
}

// stopping reports whether this router has stopped claiming work, after which it places no
// new call: during the drain a call could be cut off at the drain's deadline, and after it
// the sessions are being ended, so the person would be rung with no agent.
func (r *Runner) stopping() bool {
	r.mu.Lock()
	stopped := r.stopped
	r.mu.Unlock()
	select {
	case <-stopped:
		return true
	default:
		return false
	}
}

// ring places one dial and records it. Only a failure to record it is an error.
func (r *Runner) ring(ctx context.Context, campaign store.Campaign, contact store.Contact, step Step, number int) (Dialed, error) {
	sessionID := uuid.NewString()
	attempt := store.CampaignAttempt{
		CampaignID: campaign.ID, ContactID: contact.ID, StepID: step.ID, Attempt: max(contact.Attempt, 1),
		Dial: number, CallID: sessionID, StartedAt: r.now(),
	}
	if err := r.store.StartDial(ctx, &attempt); err != nil {
		return Dialed{}, err
	}

	var dialed Dialed
	var err error
	if r.stopping() {
		dialed.Result = Canceled
	} else {
		dialed, err = r.dialer.Dial(ctx, Dial{Campaign: campaign, Contact: contact, Step: step, Number: number, SessionID: sessionID})
	}
	if err != nil {
		r.logger.Warn("a campaign dial failed", "campaign", campaign.ID, "contact", contact.ID, "error", err)
		if dialed.Result == "" {
			dialed = resultOf(err)
		}
	}

	written, cancel := context.WithTimeout(context.WithoutCancel(ctx), writeTimeout)
	defer cancel()
	ended := r.now()
	var classifyAfter *time.Time
	if dialed.Result == Answered {
		at := ended.Add(classifyDelay)
		classifyAfter = &at
	}
	if err := r.store.EndDial(written, attempt.ID, string(dialed.Result), dialed.CallID, dialed.VendorCallID, ended, classifyAfter); err != nil {
		return Dialed{}, err
	}
	return dialed, nil
}

// takeOver settles the contacts whose router died mid-call, from the call it placed. None is
// rung again.
func (r *Runner) takeOver(ctx context.Context) {
	claimed, err := r.store.TakeOverExpired(ctx, r.owner, r.now(), lease, maxCalls)
	if err != nil {
		r.logger.Error("could not take over expired campaign leases", "error", err)
		return
	}
	for _, one := range claimed {
		if err := r.recover(ctx, one); err != nil {
			r.logger.Error("could not settle a campaign contact taken over", "contact", one.Contact.ID, "error", err)
		}
	}
}

func (r *Runner) recover(ctx context.Context, claimed store.Claimed) error {
	campaign, contact := claimed.Campaign, claimed.Contact
	sequence, hours, ok := r.plan(ctx, campaign, contact)
	if !ok {
		return nil
	}
	zone := hours.Zone(contact.TimeZone, contact.ToNumber)
	now := r.now()
	latest, found, err := r.store.LatestDial(ctx, contact.ID)
	if err != nil {
		return err
	}
	// A dial of an earlier attempt is not this claim's: the router died before dialing. The
	// attempt number tells them apart without comparing two routers' clocks; a dial of the
	// same attempt that never rang (paused, deferred) settles unused from its own result.
	if !found || latest.Attempt != max(contact.Attempt, 1) {
		r.settle(ctx, contact, store.Settle{Stage: store.Enrolled, StepID: sequence.First().ID, Attempt: contact.Attempt, NextAttemptAt: &now})
		return nil
	}

	result := Result(latest.Result)
	if result == "" {
		answered, err := answeredCall(ctx, r.store, campaign.CustomerID, latest.CallID)
		if err != nil {
			return err
		}
		result = Unanswered
		var classifyAfter *time.Time
		if answered {
			result, classifyAfter = Answered, &now
		}
		if err := r.store.EndDial(ctx, latest.ID, string(result), "", "", now, classifyAfter); err != nil {
			return err
		}
	}
	dialed := Dialed{Result: result, CallID: latest.CallID, VendorCallID: latest.VendorCallID}
	settle, again := afterDial(sequence, hours, zone, contact, latest.Dial, dialed, latest.StartedAt, now)
	if again {
		settle, _ = afterDial(sequence, hours, zone, contact, 2, dialed, latest.StartedAt, now)
	}
	r.settle(ctx, contact, settle)
	return nil
}

// classifyDue reads the outcomes of the answered calls whose transcripts should be written by
// now, and waits for them.
func (r *Runner) classifyDue(ctx context.Context) {
	claimed, err := r.store.ClaimClassifications(ctx, r.owner, r.now(), lease, maxClassifying)
	if err != nil {
		r.logger.Error("could not claim campaign calls to classify", "error", err)
		return
	}
	var reading sync.WaitGroup
	for _, one := range claimed {
		reading.Go(func() {
			if err := r.classify(ctx, one); err != nil && ctx.Err() == nil {
				r.logger.Error("could not classify a campaign call", "attempt", one.Attempt.ID, "error", err)
			}
		})
	}
	reading.Wait()
}

// classify settles a contact from what was said on its answered call. A write that fails
// leaves the dial to be read again once its lease runs out.
func (r *Runner) classify(ctx context.Context, claimed store.Classifying) error {
	campaign, contact, attempt := claimed.Campaign, claimed.Contact, claimed.Attempt
	sequence, hours, ok := r.plan(ctx, campaign, contact)
	if !ok {
		return nil
	}
	step := sequence.First()
	zone := hours.Zone(contact.TimeZone, contact.ToNumber)

	said, unread := r.transcripts.CallTranscript(ctx, campaign.CustomerID, attempt.CallID)
	if unread != nil && ctx.Err() != nil {
		return unread
	}
	if unread != nil {
		r.logger.Warn("could not read a campaign call's transcript", "attempt", attempt.ID, "error", unread)
	}
	if (unread != nil || len(said) == 0) && attempt.ClassifyTries < maxClassifyTries {
		return r.store.PutOffClassification(ctx, r.owner, attempt.ID, r.now().Add(classifyRetry))
	}

	var settle store.Settle
	switch {
	case unread != nil:
		// Answered, and never readable: what was said is unknown, which is other.
		settle = afterOutcome(step, contact, "")
	case !callerSpoke(said):
		return r.unanswered(ctx, sequence, hours, zone, contact, attempt)
	default:
		chosen, err := r.classifier.Classify(ctx, campaign, attempt.CallID, step, said)
		if err != nil {
			if ctx.Err() != nil {
				return err
			}
			r.logger.Warn("could not classify a campaign call, so it counts as other", "attempt", attempt.ID, "error", err)
		}
		if chosen == noPerson {
			return r.unanswered(ctx, sequence, hours, zone, contact, attempt)
		}
		settle = afterOutcome(step, contact, chosen)
	}

	if settle.Outcome == OutcomeDoNotCall {
		optOut := store.OptOut{CustomerID: campaign.CustomerID, Recipient: contact.ToNumber, Channel: voice, Source: optOutSource}
		if err := r.store.OptOut(ctx, &optOut); err != nil {
			return err
		}
	}
	return r.store.ClassifyAndSettle(ctx, r.owner,
		store.Classification{AttemptID: attempt.ID, Outcome: settle.Outcome, Converted: settle.Converted}, contact.ID, settle)
}

// unanswered settles an answered call nobody spoke on like a dial nobody picked up: the next
// attempt goes on schedule.
func (r *Runner) unanswered(ctx context.Context, sequence Sequence, hours CallingHours, zone *time.Location, contact store.Contact, attempt store.CampaignAttempt) error {
	return r.store.ClassifyAndSettle(ctx, r.owner, store.Classification{AttemptID: attempt.ID, Unanswered: true},
		contact.ID, afterSilence(sequence, hours, zone, contact, attempt.StartedAt, r.now()))
}

// settle writes where a call left a contact, even once the call's context is gone.
func (r *Runner) settle(ctx context.Context, contact store.Contact, settle store.Settle) {
	written, cancel := context.WithTimeout(context.WithoutCancel(ctx), writeTimeout)
	defer cancel()
	err := r.store.SettleContact(written, r.owner, contact.ID, settle)
	switch {
	case err == nil:
	case errors.Is(err, store.ErrNotLeaseHolder):
		r.logger.Warn("another router took a campaign contact over before it settled", "contact", contact.ID)
	default:
		r.logger.Error("could not record what became of a campaign contact", "contact", contact.ID, "error", err)
	}
}

// pause stops a campaign whose call was refused for a reason every call after it would be.
func (r *Runner) pause(ctx context.Context, campaign store.Campaign, reason string) {
	written, cancel := context.WithTimeout(context.WithoutCancel(ctx), writeTimeout)
	defer cancel()
	err := r.store.PauseCampaign(written, campaign.CustomerID, campaign.ID, reason)
	switch {
	case err == nil:
		r.logger.Info("paused a campaign that may not place calls", "campaign", campaign.ID, "reason", reason)
	case errors.Is(err, store.ErrCampaignLocked):
		// Another call of it was refused first, or somebody paused or stopped it.
	default:
		r.logger.Error("could not pause a campaign that may not place calls", "campaign", campaign.ID, "error", err)
	}
}

// plan reads how a campaign rings contact. One that cannot be read fails the contact, since
// every router would read it the same way.
func (r *Runner) plan(ctx context.Context, campaign store.Campaign, contact store.Contact) (Sequence, CallingHours, bool) {
	sequence, hours, err := planOf(campaign)
	if err == nil {
		return sequence, hours, true
	}
	r.logger.Error("could not read how a campaign rings people", "campaign", campaign.ID, "error", err)
	r.settle(ctx, contact, store.Settle{Stage: store.Failed, Attempt: contact.Attempt, Reason: reasonInvalidPlan})
	return Sequence{}, CallingHours{}, false
}

// planOf reads a campaign's sequence and calling hours, which launch validated.
func planOf(campaign store.Campaign) (Sequence, CallingHours, error) {
	sequence, err := ParseSequence(campaign.Sequence)
	if err != nil {
		return Sequence{}, CallingHours{}, err
	}
	hours, err := ParseCallingHours(campaign.CallingHours)
	if err != nil {
		return Sequence{}, CallingHours{}, err
	}
	return sequence, hours, nil
}

// callerSpoke reports whether anybody but the agent said anything.
func callerSpoke(said []chatlog.Spoken) bool {
	for _, line := range said {
		if !line.Agent && strings.TrimSpace(line.Text) != "" {
			return true
		}
	}
	return false
}

// placing is the call one contact is rung on. The answered leg is routed into the call of
// the session the phone service chooses, which is opened with that id.
func placing(campaign store.Campaign, contact store.Contact) phone.CallRequest {
	return phone.CallRequest{
		Owner: routing.Owner{CustomerID: campaign.CustomerID, Attribution: scheduledBy(campaign), Tags: campaign.Tags},
		From:  campaign.FromNumber,
		To:    contact.ToNumber,
		// Only a campaign cares who answered, so no other outbound call asks the vendor.
		DetectMachine: true,
	}
}

// scheduledBy owns a campaign's calls as whoever scheduled it: the key, the user and the usage
// categories of the request that created it. One created before campaigns kept a key reads
// back job:campaign (the migration); one that names none now is a gap the writer reports.
func scheduledBy(campaign store.Campaign) routing.Attribution {
	return routing.Attribution{APIKey: campaign.APIKey, UserID: campaign.UserID, UsageCategories: campaign.UsageCategories}
}

// meet opens the session under the id the call was placed for, so the agent is in the call
// the person was routed into.
func meet(spec *session.Spec, placed phone.Placed) {
	spec.ID = placed.SessionID
	spec.Text = false
}

// pausing is why a refused call pauses the whole campaign rather than fails one contact: a
// suspended app's telephony, or an organization that may not pay, refuses every call after it
// too. It is the code the campaign records, or "" for a refusal of this contact alone.
func pausing(err error) string {
	switch {
	case errors.Is(err, phone.ErrSuspended):
		return "telephony_suspended"
	case errors.Is(err, dlc.ErrUnpaid):
		// The gate's refusal always wraps one of the policy's; a refusal it cannot name is
		// still a refusal of money.
		return cmp.Or(policy.Code(err), "insufficient_credits")
	}
	return ""
}

// say adds what this person was rung about to what the agent is always told. The
// contact's own instructions go last, so they are the most recent thing the model read.
func say(configured, contact string) string {
	switch {
	case contact == "":
		return configured
	case configured == "":
		return contact
	default:
		return configured + "\n\n" + contact
	}
}

// merge labels a campaign's calls with the campaign's own labels as well as the config's,
// so what an outbound push cost can be told apart from what the same agent cost inbound.
func merge(configured, campaign routing.Tags) routing.Tags {
	if len(campaign) == 0 {
		return configured
	}
	merged := routing.Tags{}
	for key, tag := range configured {
		merged[key] = tag
	}
	for key, tag := range campaign {
		merged[key] = tag
	}
	return merged
}
