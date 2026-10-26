//go:build integration

package session

import (
	"bytes"
	"context"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"time"

	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

// shown is what a chat's conversation was asked to show: the consents begun, each card on the
// reply with whom it was for, and how many steps were asked to wait for a login.
type shown struct {
	mu     sync.Mutex
	begun  []ConsentRequest
	cards  []persistent.ConnectorAuthorization
	owners []string
	steps  int
}

func (s *shown) counts() (begun, cards, steps int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.begun), len(s.cards), s.steps
}

// askingSandbox is the sandbox of spec's session, on a router whose consents begin begun,
// with its connectors attached, in a conversation of owner's that records in shown what it is
// asked to show. A step asked to wait is refused, so a list_tools call answers at once.
func (s *SandboxGrantsSuite) askingSandbox(spec Spec, owner string, begun Consent) (sandbox.Sandbox, *dispatcher, *shown) {
	seen := &shown{}
	s.manager.options.Connectors.Consents = func(_ context.Context, request ConsentRequest) (Consent, error) {
		seen.mu.Lock()
		defer seen.mu.Unlock()
		seen.begun = append(seen.begun, request)
		return begun, nil
	}
	d, _, _, err := s.attach(spec)
	s.Require().NoError(err)
	s.T().Cleanup(d.closeLogins)
	if d.logins != nil {
		d.logins.canAsk = func(user string) bool { return user == owner }
		d.logins.ask = func(user string, card persistent.ConnectorAuthorization) bool {
			seen.mu.Lock()
			defer seen.mu.Unlock()
			seen.cards, seen.owners = append(seen.cards, card), append(seen.owners, user)
			return user == owner
		}
		d.logins.shows = func(id string) bool {
			seen.mu.Lock()
			defer seen.mu.Unlock()
			return slices.ContainsFunc(seen.cards, func(card persistent.ConnectorAuthorization) bool { return card.AuthorizationID == id })
		}
		d.logins.await = func(string, string, persistent.Authorization) bool {
			seen.mu.Lock()
			defer seen.mu.Unlock()
			seen.steps++
			return false
		}
	}
	box, err := s.manager.box(s.ctx, spec, d)
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = box.Close() })
	return box, d, seen
}

// TestASandboxWithNoLateGrantShowsNothing, the control: a chat whose sandbox binding is the
// app's own connection runs code as on base and shows nobody a consent.
func (s *SandboxGrantsSuite) TestASandboxWithNoLateGrantShowsNothing() {
	s.daytonaAPI()
	s.sandboxRevision()
	app := s.connection("", "secondary")
	config := s.config(sandboxed(s.fixed("app", app, "whoami")))
	box, _, seen := s.askingSandbox(s.chat(config, consent(app)), "alice", consent(app))

	result, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	said, _ := harness.HandBack(s.ctx, nil, result, s.manager.logger)
	begun, cards, steps := seen.counts()
	// All as on base 58f2ce78d (probe, 2026-10-26).
	s.Equal(sandbox.Result{}, result)
	s.Equal("The code ran and printed nothing.", said)
	s.Equal([]int{0, 0, 0}, []int{begun, cards, steps})
}

// unconnectedSaid is what a run of alice's chat with crm unconnected hands the model, as on
// base 58f2ce78d (probe, 2026-10-26): showing her the consent changes none of it.
const unconnectedSaid = "The code ran and printed nothing.\n\nIf this code needed crm (a 401 from api.crm.example, " +
	"*.crm.example), the user has not connected it: call crm__list_tools so they can, then run the code again; if you " +
	"cannot call it, tell the user they must connect crm."

// card is alice's consent on connection, expiring at expires, as T17 begins one.
func card(connection string, expires time.Time) Consent {
	return Consent{ConnectionID: connection, AuthorizationID: "attempt-alice", LaunchURL: "https://router.example/launch/attempt-alice",
		HandoffToken: "handoff-alice", Name: "CRM", ExpiresAt: expires}
}

// openFor is a consent on mine that stays open past the test.
func openFor(mine string) Consent { return card(mine, time.Now().Add(10*time.Minute)) }

// lateChat is alice's chat binding crm as her own, with sandbox, which she has a pending
// connection for and has not connected, in owner's conversation, in writing or on a call.
func (s *SandboxGrantsSuite) lateChat(text bool, owner string, begun func(mine string) Consent) (sandbox.Sandbox, *dispatcher, *shown, string) {
	s.sandboxRevision()
	mine := s.pending("alice", "primary")
	spec := s.chat(s.config(sandboxed(s.chosen("crm", "whoami"))), Consent{})
	spec.Text = text
	box, d, seen := s.askingSandbox(spec, owner, begun(mine))
	return box, d, seen, mine
}

// TestARunWithNoConnectionShowsTheUserTheConsentAndHoldsNothing: the model need not call
// list_tools: the run itself shows alice the consent for her own connection on the reply, and
// answers at once, with no step waiting and what the model reads as on base.
func (s *SandboxGrantsSuite) TestARunWithNoConnectionShowsTheUserTheConsentAndHoldsNothing() {
	s.daytonaAPI()
	box, _, seen, mine := s.lateChat(true, "alice", openFor)

	result, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	said, _ := harness.HandBack(s.ctx, nil, result, s.manager.logger)
	s.Equal(unconnectedSaid, said)
	s.Equal([]ConsentRequest{{CustomerID: s.customerID, ConnectorID: s.connectorID, UserID: "alice"}}, seen.begun,
		"for alice's own connection, the one the consent makes or her newest")
	s.Require().Len(seen.cards, 1)
	begun := openFor(mine)
	s.WithinDuration(begun.ExpiresAt, seen.cards[0].ExpiresAt, time.Minute)
	s.Equal(persistent.ConnectorAuthorization{Name: "crm", ConnectorID: s.connectorID, ConnectionID: mine,
		AuthorizationID: "attempt-alice", Title: "Connect CRM", LaunchURL: begun.LaunchURL, HandoffToken: begun.HandoffToken,
		ExpiresAt: seen.cards[0].ExpiresAt}, seen.cards[0])
	s.Equal([]string{"alice"}, seen.owners)
	s.Zero(seen.steps, "no step waits for the login")
}

// TestARunShowsANewConsentOnlyOnceTheOpenOneEnded: a run in the same reply, and one in a later
// reply, show nothing new while alice's consent is open, and a list_tools call in the reply
// that shows it reuses it. Once the provider declined it, or it expired, it can no longer be
// finished, so the next run shows a new one.
func (s *SandboxGrantsSuite) TestARunShowsANewConsentOnlyOnceTheOpenOneEnded() {
	s.daytonaAPI()
	box, d, seen, mine := s.lateChat(true, "alice", openFor)
	lapsed, _, afterExpiry, _ := s.lateChat(true, "alice", func(mine string) Consent { return card(mine, time.Now().Add(-time.Second)) })

	for _, run := range []sandbox.Sandbox{box, box, lapsed, lapsed} {
		_, err := run.Run(s.ctx, "print(1)", nil)
		s.Require().NoError(err)
	}
	listed, err := s.call(d, "crm__list_tools", `{}`)
	s.Require().NoError(err)
	d.logins.shows = func(string) bool { return false }
	later, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	begun, cards, steps := seen.counts()
	s.Require().True(d.loginEndedWithout("attempt-alice", mine, persistent.AuthorizationDeclined))
	_, err = box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.Equal([]int{1, 1, 1}, []int{begun, cards, steps}, "list_tools asked its step to wait on the consent shown")
	s.Contains(listed, `"status":"authorization_required"`)
	s.Equal([]string{"crm"}, bindings(later.Unconnected), "the model is still told")
	declined, _, _ := seen.counts()
	expired, _, _ := afterExpiry.counts()
	s.Equal([]int{2, 2}, []int{declined, expired}, "a declined, and an expired, consent is followed by a new one")
}

// TestARunShowsNoConsentWhereNobodyCanAnswerIt: on a call nobody presses a button while they
// talk; nobody is asked in a conversation that is not alice's own (CanAskToConnect); no
// consent fixes a token the provider rejected (credentialRejected); and none is begun for a
// binding the config no longer holds as the caller's own (stillWaits).
func (s *SandboxGrantsSuite) TestARunShowsNoConsentWhereNobodyCanAnswerIt() {
	s.daytonaAPI()
	cases := map[string]struct {
		text  bool
		owner string
		spoil func(d *dispatcher)
	}{
		"on a call":             {owner: "alice"},
		"in bob's conversation": {text: true, owner: "bob"},
		"with a rejected token": {text: true, owner: "alice", spoil: func(d *dispatcher) { d.logins.byAlias["crm"].rejected = true }},
		"after the config dropped it": {text: true, owner: "alice", spoil: func(d *dispatcher) {
			config, err := s.store.AgentConfig(s.ctx, s.customerID, d.spec.ConfigID)
			s.Require().NoError(err)
			s.rebind(config, sandboxed(s.chosen("other", "whoami")))
		}},
	}
	for name, run := range cases {
		box, d, seen, _ := s.lateChat(run.text, run.owner, openFor)
		if run.spoil != nil {
			run.spoil(d)
		}

		result, err := box.Run(s.ctx, "print(1)", nil)

		s.Require().NoError(err, name)
		s.Equal([]string{"crm"}, bindings(result.Unconnected), name)
		begun, cards, steps := seen.counts()
		s.Equal([]int{0, 0, 0}, []int{begun, cards, steps}, name)
	}
}

// lockedBuffer is a log a watch writes while the test reads it.
type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

// held puts d in the manager as the dispatcher of a session with no conversation, so a consent
// handed back to it opens d's login, and its follow-up fails and is logged in s.handedBack.
func (s *SandboxGrantsSuite) held(d *dispatcher) {
	s.manager.mu.Lock()
	defer s.manager.mu.Unlock()
	if s.handedBack == nil {
		s.handedBack, s.manager.sessions = &lockedBuffer{}, map[string]*Session{}
		s.manager.logger = slog.New(slog.NewTextHandler(s.handedBack, nil))
	}
	s.manager.sessions[d.spec.ID] = &Session{id: d.spec.ID, spec: d.spec, connectors: d}
}

// carriedOn is how many times the session was told to carry on after a login.
func (s *SandboxGrantsSuite) carriedOn() int {
	s.handedBack.mu.Lock()
	defer s.handedBack.mu.Unlock()
	return strings.Count(s.handedBack.buf.String(), "could not carry on after a login in the conversation")
}

// TestAConsentARunShowedCarriesOnOnceAndTheNextRunBinds: the callback hands alice's consent
// back here. No call waits for it, so the session is told to carry on, once: its own read of
// the connection then finds the login open. The next run puts her token in the sandbox.
func (s *SandboxGrantsSuite) TestAConsentARunShowedCarriesOnOnceAndTheNextRunBinds() {
	api := s.daytonaAPI()
	box, d, seen, mine := s.lateChat(true, "alice", openFor)
	s.held(d)
	_, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)

	s.connect(mine, "primary")
	s.manager.ConnectorConsentFinished(s.ctx, s.customerID, mine, "attempt-alice")
	bound, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	later, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.Equal(1, s.carriedOn())
	s.Equal(sandbox.Result{Connected: []string{"crm"}}, bound)
	s.Equal(sandbox.Result{}, later)
	s.Equal(tokenOf("primary"), api.value("CRM_TOKEN"))
	begun, cards, _ := seen.counts()
	s.Equal([]int{1, 1}, []int{begun, cards})
	s.Never(func() bool { return s.carriedOn() > 1 }, loginPoll+time.Second, 100*time.Millisecond)
}

// TestAConsentFinishedOnAnotherRouterCarriesOnHere: the callback that finishes alice's consent
// lands on another router, which hands it to no session here. Her session reads the
// connection, opens the binding, tells the model to carry on, and the next run binds.
func (s *SandboxGrantsSuite) TestAConsentFinishedOnAnotherRouterCarriesOnHere() {
	api := s.daytonaAPI()
	box, d, _, mine := s.lateChat(true, "alice", openFor)
	s.held(d)
	_, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)

	time.Sleep(loginPoll + loginPoll/2) // a read before she connects finds nothing to hand back
	s.connect(mine, "primary")
	s.Require().Eventually(func() bool { return s.carriedOn() > 0 }, 2*loginPoll+5*time.Second, 50*time.Millisecond)
	bound, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.Equal(1, s.carriedOn())
	s.True(d.logins.byAlias["crm"].loggedIn())
	s.Equal(sandbox.Result{Connected: []string{"crm"}}, bound)
	s.Equal(tokenOf("primary"), api.value("CRM_TOKEN"))
}

// TestAConsentNoLongerWatchedOpensNothing: the session closed, or the consent expired, before
// alice connected, so nothing reads the connection for it any more.
func (s *SandboxGrantsSuite) TestAConsentNoLongerWatchedOpensNothing() {
	s.daytonaAPI()
	closed, closing, _, closedMine := s.lateChat(true, "alice", openFor)
	expiring := func(mine string) Consent { return card(mine, time.Now().Add(loginPoll/2)) }
	lapsed, lapsing, _, lapsedMine := s.lateChat(true, "alice", expiring)
	for _, d := range []*dispatcher{closing, lapsing} {
		s.held(d)
	}
	for _, box := range []sandbox.Sandbox{closed, lapsed} {
		_, err := box.Run(s.ctx, "print(1)", nil)
		s.Require().NoError(err)
	}
	closing.closeLogins()

	time.Sleep(loginPoll / 2)
	s.connect(closedMine, "primary")
	s.connect(lapsedMine, "primary")

	s.Never(func() bool {
		return closing.logins.byAlias["crm"].loggedIn() || lapsing.logins.byAlias["crm"].loggedIn()
	}, loginPoll+time.Second, 100*time.Millisecond)
	s.Zero(s.carriedOn())
}

// TestRunsWhileTheConsentFinishesBeginOneConsent, under -race: runs show the consent while
// alice's consent finishes and opens the binding. One consent is begun, and once a run binds
// her token no run names crm again.
func (s *SandboxGrantsSuite) TestRunsWhileTheConsentFinishesBeginOneConsent() {
	api := s.daytonaAPI()
	box, d, seen, mine := s.lateChat(true, "alice", openFor)

	var wg sync.WaitGroup
	for range 2 {
		wg.Go(func() {
			for range 5 {
				_, err := box.Run(s.ctx, "print(1)", nil)
				s.NoError(err)
			}
		})
	}
	s.Require().Eventually(func() bool { begun, _, _ := seen.counts(); return begun > 0 }, 5*time.Second, time.Millisecond)
	s.connect(mine, "primary")
	_, finished, _ := d.loginFinished(s.ctx, "attempt-alice", mine)
	wg.Wait()
	_, err := box.Run(s.ctx, "print(1)", nil)
	s.Require().NoError(err)
	after, err := box.Run(s.ctx, "print(1)", nil)

	s.Require().NoError(err)
	s.True(finished)
	begun, cards, _ := seen.counts()
	s.Equal([]int{1, 1}, []int{begun, cards})
	s.Equal(sandbox.Result{}, after)
	s.Equal(tokenOf("primary"), api.value("CRM_TOKEN"))
}
