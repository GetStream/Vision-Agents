// Package campaign rings a list of people with one agent.
//
// A campaign is the outbound half of telephony: rather than an agent waiting for a call,
// the process places one, has the conversation, and moves on to the next person. What
// makes it more than a loop is concurrency: the customer says how many of these may be
// happening at once, and the runner holds that many and no more.
//
// Concurrency is enforced in this process, which is the same place sessions already live.
// A second router would run its own campaigns rather than share these, and a contact is
// claimed in the database so a restart mid-campaign resumes rather than starts again.
package campaign

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// callTimeout bounds one conversation. Nothing else ends a campaign call: the agent is
// not told when the person hangs up, and a slot held forever by a call that is over is a
// campaign that stops halfway through.
const callTimeout = 15 * time.Minute

// writeTimeout bounds the writes a campaign makes on its way out, which have to happen
// on a context the campaign being stopped did not cancel.
const writeTimeout = 10 * time.Second

// Options configures a Runner. All of them are required: a campaign is a phone call, a
// conversation and a row, and it cannot be any of the three on its own.
type Options struct {
	Store    *store.Store
	Phone    *phone.Service
	Sessions *session.Manager
	Logger   *slog.Logger
}

// Runner works through the campaigns that have been started.
type Runner struct {
	store    *store.Store
	phone    *phone.Service
	sessions *session.Manager
	logger   *slog.Logger

	mu sync.Mutex
	// running is how a campaign is stopped: the entry is the way to cancel its loop.
	running map[string]context.CancelFunc
	closed  bool

	working sync.WaitGroup
}

// New validates the options and returns a Runner. It starts nothing.
func New(options Options) (*Runner, error) {
	if options.Store == nil {
		return nil, errors.New("campaign: a database is required")
	}
	if options.Phone == nil {
		return nil, errors.New("campaign: telephony is required")
	}
	if options.Sessions == nil {
		return nil, errors.New("campaign: a session manager is required")
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	return &Runner{
		store:    options.Store,
		phone:    options.Phone,
		sessions: options.Sessions,
		logger:   options.Logger,
		running:  map[string]context.CancelFunc{},
	}, nil
}

// Start begins working through a campaign's contacts. Starting one that is already
// running is not an error: it is what a caller who lost track of it meant.
func (r *Runner) Start(ctx context.Context, customerID, id string) error {
	campaign, err := r.store.Campaign(ctx, customerID, id)
	if err != nil {
		return err
	}
	if campaign.ConfigID == "" {
		return errors.New("campaign: a campaign needs an agent config to make its calls with")
	}
	if campaign.FromNumber == "" {
		return errors.New("campaign: a campaign needs one of your numbers to call from")
	}
	// The config is read once, here, rather than per call: a campaign is one agent
	// ringing many people, and editing it halfway through should not change who they
	// are talking to.
	config, err := r.store.AgentConfig(ctx, customerID, campaign.ConfigID)
	if err != nil {
		return err
	}

	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return errors.New("campaign: the runner is shut down")
	}
	if _, already := r.running[campaign.ID]; already {
		r.mu.Unlock()
		return nil
	}
	// The loop outlives the request that started it, so it takes the background rather
	// than a context that is cancelled the moment the caller is answered.
	loop, cancel := context.WithCancel(context.WithoutCancel(ctx))
	r.running[campaign.ID] = cancel
	r.mu.Unlock()

	// A campaign that stopped mid-call left contacts claimed by nobody.
	if err := r.store.ReleaseContacts(ctx, campaign.ID); err != nil {
		r.forget(campaign.ID)
		cancel()
		return err
	}
	if err := r.store.SetCampaignState(ctx, customerID, campaign.ID, store.Running); err != nil {
		r.forget(campaign.ID)
		cancel()
		return err
	}

	r.working.Add(1)
	go func() {
		defer r.working.Done()
		defer cancel()
		r.run(loop, campaign, config)
	}()
	return nil
}

// Pause stops making new calls. The conversations already happening are left alone: a
// campaign is paused to stop ringing people, not to hang up on the ones who answered.
func (r *Runner) Pause(ctx context.Context, customerID, id string) error {
	campaign, err := r.store.Campaign(ctx, customerID, id)
	if err != nil {
		return err
	}

	r.forget(campaign.ID)
	return r.store.SetCampaignState(ctx, customerID, campaign.ID, store.Paused)
}

// Close stops every campaign and waits for the loops to end.
func (r *Runner) Close() {
	r.mu.Lock()
	r.closed = true
	for id, cancel := range r.running {
		cancel()
		delete(r.running, id)
	}
	r.mu.Unlock()

	r.working.Wait()
}

// run works through a campaign's contacts, holding at most its concurrency at once.
func (r *Runner) run(ctx context.Context, campaign store.Campaign, config store.AgentConfig) {
	slots := make(chan struct{}, campaign.Concurrency)
	var calls sync.WaitGroup

	for {
		select {
		case <-ctx.Done():
			calls.Wait()
			return
		case slots <- struct{}{}:
		}

		contact, found, err := r.store.ClaimContact(ctx, campaign.ID)
		if err != nil {
			<-slots
			r.logger.Error("could not take the next contact",
				"campaign", campaign.ID, "error", err)
			calls.Wait()
			r.stop(campaign, store.Paused)
			return
		}
		if !found {
			<-slots
			calls.Wait()
			// Another call may have been running when the queue emptied, so this is
			// checked after waiting rather than before.
			if remaining, err := r.pending(ctx, campaign.ID); err == nil && remaining {
				continue
			}
			r.stop(campaign, store.Finished)
			return
		}

		calls.Add(1)
		go func() {
			defer calls.Done()
			defer func() { <-slots }()
			r.ring(ctx, campaign, config, contact)
		}()
	}
}

// ring calls one person and stays on the line until the conversation is over.
func (r *Runner) ring(
	ctx context.Context,
	campaign store.Campaign,
	config store.AgentConfig,
	contact store.Contact,
) {
	finished := store.Contact{ID: contact.ID, State: store.Done}

	placed, err := r.phone.Call(ctx, phone.CallRequest{
		Owner: routing.Owner{CustomerID: campaign.CustomerID, Tags: campaign.Tags},
		From:  campaign.FromNumber,
		To:    contact.ToNumber,
	})
	if err != nil {
		finished.State = store.Failed
		finished.Error = err.Error()
		r.finish(ctx, finished)
		return
	}
	finished.VendorCallID = placed.VendorCallID

	spec := session.FromConfig(config)
	spec.CallID = "campaign-" + contact.ID
	spec.CampaignID = campaign.ID
	spec.ContactID = contact.ID
	// The agent placed this call, so it is told how to get past whatever answers before
	// it is told what this particular person was rung about.
	spec.Navigating = true
	spec.Instructions = say(spec.Instructions, contact.Instructions)
	spec.Tags = merge(spec.Tags, campaign.Tags)
	spec.Phone = &session.PhoneSpec{
		Number:       campaign.FromNumber,
		To:           contact.ToNumber,
		VendorCallID: placed.VendorCallID,
	}

	created, err := r.sessions.Create(ctx, spec)
	if err != nil {
		finished.State = store.Failed
		finished.Error = err.Error()
		r.finish(ctx, finished)
		return
	}
	finished.CallID = created.ID()

	r.hold(ctx, created)
	r.finish(ctx, finished)
}

// hold waits for the conversation to end, and ends it if nothing else does.
//
// Nothing tells the agent that the person hung up, so a campaign call that is over would
// otherwise hold its slot until the process stopped. The timeout is what makes the
// concurrency mean calls rather than sessions.
func (r *Runner) hold(ctx context.Context, created *session.Session) {
	events, detach := created.Watch()
	defer detach()

	deadline := time.NewTimer(callTimeout)
	defer deadline.Stop()

	for {
		select {
		case _, open := <-events:
			if !open {
				return
			}
		case <-deadline.C:
			r.close(created)
			return
		case <-ctx.Done():
			r.close(created)
			return
		}
	}
}

func (r *Runner) close(created *session.Session) {
	if err := created.Close(); err != nil {
		r.logger.Error("could not end a campaign call", "session", created.ID(), "error", err)
	}
}

// finish records what became of a contact, off the context the campaign was cancelled
// with: a stopped campaign must still write down who it rang.
func (r *Runner) finish(ctx context.Context, contact store.Contact) {
	written, cancel := context.WithTimeout(context.WithoutCancel(ctx), writeTimeout)
	defer cancel()

	if err := r.store.FinishContact(written, contact); err != nil {
		r.logger.Error("could not record what became of a contact",
			"contact", contact.ID, "error", err)
	}
}

// pending reports whether anybody is still waiting to be rung.
func (r *Runner) pending(ctx context.Context, campaignID string) (bool, error) {
	contacts, err := r.store.CampaignContacts(ctx, campaignID)
	if err != nil {
		return false, err
	}
	for _, contact := range contacts {
		if contact.State == store.Pending {
			return true, nil
		}
	}
	return false, nil
}

// stop takes the campaign off the running list and records where it got to.
func (r *Runner) stop(campaign store.Campaign, state string) {
	r.forget(campaign.ID)

	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()

	if err := r.store.SetCampaignState(ctx, campaign.CustomerID, campaign.ID, state); err != nil {
		r.logger.Error("could not record how a campaign ended",
			"campaign", campaign.ID, "error", err)
	}
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

func (r *Runner) forget(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if cancel, ok := r.running[id]; ok {
		cancel()
		delete(r.running, id)
	}
}
