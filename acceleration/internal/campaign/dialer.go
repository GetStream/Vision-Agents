package campaign

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone/outcome"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// reasonConfigMissing pauses a campaign whose agent config was deleted: no call of it could be
// placed, so its contacts wait unrung for the campaign to be given another.
const reasonConfigMissing = "config_missing"

// reasonVoicemailNeedsExactWords pauses a campaign whose step leaves a voicemail message but
// whose agent config became speech-to-speech after launch, which cannot say fixed words.
const reasonVoicemailNeedsExactWords = "voicemail_needs_exact_words"

// reasonSessionRefused pauses a campaign whose agent config makes a session that cannot open:
// every call of it would ring somebody with no agent to talk to.
const reasonSessionRefused = "session_refused"

// reasonNoSession marks a call that rang but whose session did not open. The person was rung,
// so the attempt counts, but a second dial would ring them into the same failure.
const reasonNoSession = "no_session"

// phoneDialer rings a contact through the phone service and puts the campaign's agent in the
// call it rang.
type phoneDialer struct {
	phone    *phone.Service
	sessions *session.Manager
	store    *store.Store
	logger   *slog.Logger
}

// NewPhoneDialer places campaign calls through phone, with sessions from sessions.
func NewPhoneDialer(phone *phone.Service, sessions *session.Manager, store *store.Store, logger *slog.Logger) Dialer {
	return &phoneDialer{phone: phone, sessions: sessions, store: store, logger: logger}
}

// Dial places one call and stays on the line until the conversation is over. The agent config
// is read at each dial, so a change to it reaches the next person rung.
func (d *phoneDialer) Dial(ctx context.Context, dial Dial) (Dialed, error) {
	config, err := d.store.AgentConfig(ctx, dial.Campaign.CustomerID, cmp.Or(dial.Step.ConfigID, dial.Campaign.ConfigID))
	if errors.Is(err, store.ErrNoAgentConfig) {
		return Dialed{Result: Paused, Reason: reasonConfigMissing}, nil
	}
	if err != nil {
		return Dialed{}, err
	}

	if !dial.Step.Voicemail.hangsUp() && session.FromConfig(config).Native() {
		return Dialed{Result: Paused, Reason: reasonVoicemailNeedsExactWords}, nil
	}

	// Checked before the phone rings: a session refused after the call is placed leaves the
	// person on a line with nobody on it.
	opening := campaignSpec(config, dial.Campaign, dial.Step, dial.Contact, phone.Placed{SessionID: dial.SessionID})
	if err := opening.Normalize(); err != nil {
		d.logger.Warn("a campaign's agent config makes a session that cannot open", "campaign", dial.Campaign.ID, "error", err)
		return Dialed{Result: Paused, Reason: reasonSessionRefused}, nil
	}

	request := placing(dial.Campaign, dial.Contact, dial.Step)
	request.SessionID = dial.SessionID
	placed, err := d.phone.Call(ctx, request)
	if err != nil {
		dialed := resultOf(err)
		if dialed.Result == "" {
			return dialed, err
		}
		return dialed, nil
	}

	dialed := Dialed{CallID: placed.SessionID, VendorCallID: placed.VendorCallID}
	created, err := d.sessions.Create(ctx, campaignSpec(config, dial.Campaign, dial.Step, dial.Contact, placed))
	if unanswered, nobody := errors.AsType[*session.CallUnansweredError](err); nobody {
		dialed.Outcome = unanswered.Outcome
		return dialed, nil
	}
	if err != nil {
		// Ending the Stream call frees the call's own trunk and route once the call exists. It
		// does not hang up a leg still ringing at the vendor: that needs a hang-up by vendor
		// call id, which the phone service does not have yet.
		hangUp, cancel := context.WithTimeout(context.WithoutCancel(ctx), writeTimeout)
		defer cancel()
		if ended := d.phone.EndCall(hangUp, dial.Campaign.CustomerID, placed.StreamApp, placed.CallType, placed.CallID); ended != nil {
			d.logger.Error("could not end the Stream call of a campaign call with no session", "session", placed.SessionID, "error", ended)
		}
		// Hung up at once, before the phone could say how the call went.
		dialed.Outcome, dialed.Reason = outcome.Unknown, reasonNoSession
		return dialed, err
	}

	d.hold(ctx, created)
	if created.EndReason() == session.EndVoicemail {
		dialed.Result = Voicemail
	}
	dialed.Outcome, err = callOutcome(context.WithoutCancel(ctx), d.store, dial.Campaign.CustomerID, placed.SessionID)
	return dialed, err
}

// hold waits for the conversation to end, and ends it if nothing else does.
//
// A phone session ends itself once the person hangs up or nobody answers, which closes the
// watch. The deadline covers a call that never ends that way, which would otherwise hold its
// slot until the process stopped, and is what makes the concurrency mean calls rather than
// sessions.
func (d *phoneDialer) hold(ctx context.Context, created *session.Session) {
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
			d.close(created)
			return
		case <-ctx.Done():
			d.close(created)
			return
		}
	}
}

// close ends a call through the manager rather than the session, which is what takes it off
// the list of sessions as well as off the phone.
func (d *phoneDialer) close(created *session.Session) {
	if _, err := d.sessions.Close(created.ID(), session.OwnerOf(created.Spec())); err != nil {
		d.logger.Error("could not end a campaign call", "session", created.ID(), "error", err)
	}
}

// callOutcome is how the call placed into callID ended, as the phone keeps it. A phone that
// joined is completed, which the keeper decides only once its leg ends. One it has not decided
// yet, from a vendor that never reports or one whose report is late, is unknown.
func callOutcome(ctx context.Context, st *store.Store, customerID, callID string) (outcome.Outcome, error) {
	ctx, cancel := context.WithTimeout(ctx, writeTimeout)
	defer cancel()
	calls, err := st.PhoneCalls(ctx, customerID, []string{callID})
	if err != nil {
		return outcome.Unknown, err
	}
	for _, call := range calls {
		if call.AnsweredAt != nil {
			return outcome.Completed, nil
		}
		if call.Outcome != "" {
			return outcome.Outcome(call.Outcome), nil
		}
	}
	return outcome.Unknown, nil
}

// campaignSpec is the session that has the conversation with contact: the agent config with
// the contact's variables filled in, the variables it never names briefed to it, and the
// contact's own instructions last, so they are the most recent thing the model read.
// The goal of step comes right after the config's instructions and the closing rule after
// everything, so the agent knows why it called and when the call is over.
func campaignSpec(config store.AgentConfig, campaign store.Campaign, step Step, contact store.Contact, placed phone.Placed) session.Spec {
	spec := session.FromConfig(config)
	meet(&spec, placed)
	spec.CampaignID = campaign.ID
	spec.ContactID = contact.ID
	// The agent placed this call, so it is told how to get past whatever answers before
	// it is told what this particular person was rung about.
	spec.Navigating = true
	brief := Brief([]string{spec.Instructions, spec.Greeting, step.Goal}, contact.Variables)
	var goal string
	if step.Goal != "" {
		goal = "This call: " + Render(step.Goal, contact.Variables) + "\nWhen a person answers, tell them why you are calling in your first sentence."
	}
	spec.Instructions = say(say(say(say(Render(spec.Instructions, contact.Variables), goal), brief), contact.Instructions), closing(step))
	spec.Greeting = Render(spec.Greeting, contact.Variables)
	spec.Tags = merge(spec.Tags, campaign.Tags)
	spec.Attribution = scheduledBy(campaign)
	if !step.Voicemail.off() {
		spec.Voicemail = &session.VoicemailPlan{}
		if !step.Voicemail.hangsUp() {
			spec.Voicemail.Message = Render(step.Voicemail.Message, contact.Variables)
		}
	}
	spec.Phone = &session.PhoneSpec{
		Number:       campaign.FromNumber,
		To:           contact.ToNumber,
		VendorCallID: placed.VendorCallID,
	}
	return spec
}

// closing tells the agent which outcomes end the call. Without it the agent keeps asking
// after the person has already answered.
func closing(step Step) string {
	var rule strings.Builder
	rule.WriteString("How the call ends: once the call has reached one of these, say a short goodbye and end the call.")
	for _, choice := range step.Choices() {
		fmt.Fprintf(&rule, "\n- %s: %s", strings.TrimSuffix(choice.Name, "."), choice.Description)
	}
	rule.WriteString("\nSay only the words you would speak aloud: never put a name or a label in front of them.")
	return rule.String()
}
