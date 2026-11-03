//go:build integration

package campaign

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type RunnerSuite struct {
	suite.Suite
	ctx   context.Context
	store *store.Store

	clock       *clock
	dialer      *stubDialer
	transcripts *stubTranscripts
	classifier  *stubClassifier
	runner      *Runner
	customerID  string
}

func TestRunnerSuite(t *testing.T) {
	suite.Run(t, new(RunnerSuite))
}

func (s *RunnerSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "campaign_runner"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
}

func (s *RunnerSuite) SetupTest() {
	// A claim reaches every running campaign in the database, so each test starts with none.
	_, err := s.store.DB().ExecContext(s.ctx, "TRUNCATE campaigns, campaign_contacts, campaign_attempts, opt_outs, phone_calls CASCADE")
	s.Require().NoError(err)

	// Tuesday 10:00 UTC, inside the default weekday hours.
	s.clock = &clock{at: time.Date(2026, 3, 3, 10, 0, 0, 0, time.UTC)}
	s.dialer = &stubDialer{results: map[string][]Dialed{}}
	s.transcripts = &stubTranscripts{said: map[string][][]chatlog.Spoken{}}
	s.classifier = &stubClassifier{outcome: OutcomeSuccess}
	s.customerID = uuid.NewString()
	s.runner = s.newRunner("router-a")
}

func (s *RunnerSuite) newRunner(owner string) *Runner {
	runner, err := New(Options{
		Store: s.store, Dialer: s.dialer, Transcripts: s.transcripts, Classifier: s.classifier,
		Owner: owner, Now: s.clock.now, Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(runner.Close)
	return runner
}

// launched is a running campaign of sequence, ringing each number in the zone given.
func (s *RunnerSuite) launched(sequence Sequence, concurrency int, zone string, numbers ...string) store.Campaign {
	rawSequence, err := json.Marshal(sequence)
	s.Require().NoError(err)
	rawHours, err := json.Marshal(DefaultCallingHours())
	s.Require().NoError(err)
	campaign := store.Campaign{CustomerID: s.customerID, Name: "renewals", ConfigID: "config-1",
		FromNumber: "+15550100000", Concurrency: concurrency, Sequence: rawSequence, CallingHours: rawHours}
	s.Require().NoError(s.store.CreateCampaign(s.ctx, &campaign))
	contacts := make([]store.Contact, len(numbers))
	for i, number := range numbers {
		contacts[i] = store.Contact{ToNumber: number, TimeZone: zone}
	}
	_, _, err = s.store.AddContacts(s.ctx, campaign.ID, contacts)
	s.Require().NoError(err)
	launched, err := s.store.LaunchCampaign(s.ctx, s.customerID, campaign.ID, store.Launch{
		StartAt: s.clock.now(), Now: s.clock.now(), APIKey: "key-a", UserID: "user-1", UsageCategories: []string{"sales"},
	})
	s.Require().NoError(err)
	return launched
}

func (s *RunnerSuite) single() Sequence {
	sequence := DefaultSequence("config-1")
	sequence.DoubleDial = false
	return sequence
}

// tick runs one tick of runner and waits for the dials it started.
func (s *RunnerSuite) tick(runner *Runner) {
	runner.tick(s.ctx)
	runner.dialing.Wait()
}

func (s *RunnerSuite) contact(campaign store.Campaign, number string) (store.Contact, []store.CampaignAttempt) {
	contacts, err := s.store.QueryContacts(s.ctx, campaign.ID, store.ContactFilter{Number: number, Limit: 10})
	s.Require().NoError(err)
	s.Require().Len(contacts, 1)
	contact, attempts, err := s.store.CampaignContact(s.ctx, campaign.ID, contacts[0].ID)
	s.Require().NoError(err)
	return contact, attempts
}

func (s *RunnerSuite) campaignNow(campaign store.Campaign) store.Campaign {
	found, err := s.store.Campaign(s.ctx, campaign.CustomerID, campaign.ID)
	s.Require().NoError(err)
	return found
}

// talked is a transcript where the person answered.
func talked() []chatlog.Spoken {
	return []chatlog.Spoken{{Agent: true, Text: "Hi, this is Acme about your renewal."}, {Text: "Yes, renew it please."}}
}

func (s *RunnerSuite) TestNobodyIsCalledOutsideTheirHours() {
	// 18:40 in New York on a Tuesday, which the default hours close at 18:00.
	s.clock.set(time.Date(2026, 3, 3, 23, 40, 0, 0, time.UTC))
	campaign := s.launched(s.single(), 1, "America/New_York", "+15550000001")

	s.tick(s.runner)

	s.Empty(s.dialer.dialed())
	contact, attempts := s.contact(campaign, "+15550000001")
	s.Equal(store.Enrolled, contact.Stage)
	s.Equal(0, contact.Attempt, "waiting for hours uses no attempt")
	s.Empty(attempts)
	s.Require().NotNil(contact.NextAttemptAt)
	newYork, err := time.LoadLocation("America/New_York")
	s.Require().NoError(err)
	s.Equal(time.Date(2026, 3, 4, 9, 0, 0, 0, newYork).UTC(), contact.NextAttemptAt.UTC())
}

func (s *RunnerSuite) TestTwoRunnersTogetherKeepToTheCampaignsConcurrency() {
	s.dialer.hold = make(chan struct{})
	other := s.newRunner("router-b")
	campaign := s.launched(s.single(), 1, "UTC", "+15550000001", "+15550000002", "+15550000003")

	s.runner.tick(s.ctx)
	other.tick(s.ctx)
	s.Require().Eventually(func() bool { return s.dialer.inFlight() == 1 }, 5*time.Second, 10*time.Millisecond)
	s.runner.tick(s.ctx)
	other.tick(s.ctx)
	inProgress, err := s.store.QueryContacts(s.ctx, campaign.ID, store.ContactFilter{Stage: store.InProgress, Limit: 10})
	s.Require().NoError(err)

	s.Len(inProgress, 1)
	s.Equal(1, s.dialer.inFlight())
	close(s.dialer.hold)
	s.runner.dialing.Wait()
	other.dialing.Wait()
	s.Equal(1, s.dialer.most(), "never more than the campaign's one call at once")
}

func (s *RunnerSuite) TestAnUnansweredContactIsDialedTwiceThenRetriedLater() {
	// Wednesday, so three days on is a Saturday, outside the default hours.
	s.clock.set(time.Date(2026, 3, 4, 10, 0, 0, 0, time.UTC))
	campaign := s.launched(DefaultSequence("config-1"), 1, "UTC", "+15550000001")

	s.tick(s.runner)

	s.Len(s.dialer.dialed(), 2, "a double dial rings again at once")
	contact, attempts := s.contact(campaign, "+15550000001")
	s.Require().Len(attempts, 2)
	// Both dials started at the test's one instant, so their order is not theirs to keep.
	s.ElementsMatch([]int{1, 2}, []int{attempts[0].Dial, attempts[1].Dial})
	s.Equal([]string{string(Unanswered), string(Unanswered)}, []string{attempts[0].Result, attempts[1].Result})
	s.Equal(store.Enrolled, contact.Stage)
	s.Equal(2, contact.Attempt)
	s.Require().NotNil(contact.NextAttemptAt)
	s.Equal(time.Date(2026, 3, 9, 9, 0, 0, 0, time.UTC), contact.NextAttemptAt.UTC(), "start + 3 days, moved to Monday's opening")
}

func (s *RunnerSuite) TestAnAnsweredCallGetsItsOutcome() {
	s.dialer.answer("+15550000001")
	campaign := s.launched(s.single(), 1, "UTC", "+15550000001")

	s.tick(s.runner)
	contact, attempts := s.contact(campaign, "+15550000001")
	s.Equal(store.AwaitingOutcome, contact.Stage)
	s.Require().Len(attempts, 1)
	s.transcripts.write(attempts[0].CallID, talked())
	s.clock.advance(11 * time.Second)
	s.tick(s.runner)

	contact, attempts = s.contact(campaign, "+15550000001")
	s.Equal(store.ContactCompleted, contact.Stage)
	s.Equal(OutcomeSuccess, contact.Outcome)
	s.True(contact.Converted)
	s.Equal(OutcomeSuccess, attempts[0].Outcome)
	s.Equal("done", attempts[0].ClassifyState)
}

func (s *RunnerSuite) TestAnAnsweredCallIsNotClassifiedBeforeItsTranscriptIsWritten() {
	s.dialer.answer("+15550000001")
	campaign := s.launched(s.single(), 1, "UTC", "+15550000001")

	s.tick(s.runner)
	_, attempts := s.contact(campaign, "+15550000001")
	s.transcripts.write(attempts[0].CallID, talked())
	s.tick(s.runner)

	contact, _ := s.contact(campaign, "+15550000001")
	s.Equal(store.AwaitingOutcome, contact.Stage, "read a few seconds after the call, not at once")
}

func (s *RunnerSuite) TestDoNotCallOptsTheNumberOut() {
	s.dialer.answer("+15550000001")
	s.classifier.outcome = OutcomeDoNotCall
	campaign := s.launched(s.single(), 1, "UTC", "+15550000001")

	s.tick(s.runner)
	_, attempts := s.contact(campaign, "+15550000001")
	s.transcripts.write(attempts[0].CallID, talked())
	s.clock.advance(11 * time.Second)
	s.tick(s.runner)

	contact, _ := s.contact(campaign, "+15550000001")
	s.Equal(store.ContactCompleted, contact.Stage)
	s.Equal(OutcomeDoNotCall, contact.Outcome)
	s.False(contact.Converted)
	optOuts, err := s.store.OptOuts(s.ctx, s.customerID, 10, nil)
	s.Require().NoError(err)
	s.Require().Len(optOuts, 1)
	s.Equal("+15550000001", optOuts[0].Recipient)
	s.Equal("voice", optOuts[0].Channel)
	s.Equal("campaign", optOuts[0].Source)
	next := s.launched(s.single(), 1, "UTC", "+15550000001")
	rejected, _ := s.contact(next, "+15550000001")
	s.Equal(store.Rejected, rejected.Stage, "another campaign never rings them")
}

func (s *RunnerSuite) TestAnOptOutAfterLaunchFailsTheContactBeforeItIsDialed() {
	campaign := s.launched(s.single(), 1, "UTC", "+15550000001")
	s.Require().NoError(s.store.OptOut(s.ctx, &store.OptOut{CustomerID: s.customerID, Recipient: "+15550000001", Channel: "voice", Source: "api"}))

	s.tick(s.runner)

	s.Empty(s.dialer.dialed())
	contact, attempts := s.contact(campaign, "+15550000001")
	s.Equal(store.Failed, contact.Stage)
	s.Equal("opted_out", contact.Reason)
	s.Empty(attempts)
}

func (s *RunnerSuite) TestARouterThatDiedMidCallIsTakenOver() {
	campaign := s.launched(s.single(), 2, "UTC", "+15550000001", "+15550000002")
	// A router claimed both and died with the calls in flight; its lease has run out since.
	died := s.clock.now()
	claimed, err := s.store.ClaimDueContacts(s.ctx, "router-dead", died, lease, 10)
	s.Require().NoError(err)
	s.Require().Len(claimed, 2)
	outcomes := map[string]string{"+15550000001": "completed", "+15550000002": "no_answer"}
	callIDs := map[string]string{}
	for _, one := range claimed {
		callID := uuid.NewString()
		callIDs[one.Contact.ToNumber] = callID
		s.Require().NoError(s.store.StartDial(s.ctx, &store.CampaignAttempt{CampaignID: campaign.ID, ContactID: one.Contact.ID,
			StepID: "step_1", Attempt: 1, Dial: 1, CallID: callID, StartedAt: died}))
		s.Require().NoError(s.store.InsertPhoneCall(s.ctx, &store.PhoneCall{Token: uuid.NewString(), CustomerID: s.customerID,
			CallType: "default", CallID: callID, Vendor: "twilio", LocalNumber: campaign.FromNumber, RemoteNumber: one.Contact.ToNumber,
			PlacedAt: died, Outcome: outcomes[one.Contact.ToNumber]}))
	}
	s.transcripts.write(callIDs["+15550000001"], talked())
	s.clock.advance(lease + time.Minute)

	s.tick(s.runner)

	s.Empty(s.dialer.dialed(), "a call another router placed is never placed again")
	answered, attempts := s.contact(campaign, "+15550000001")
	s.Equal(store.ContactCompleted, answered.Stage)
	s.Equal(OutcomeSuccess, answered.Outcome)
	s.Equal(string(Answered), attempts[0].Result)
	unanswered, attempts := s.contact(campaign, "+15550000002")
	s.Equal(store.Enrolled, unanswered.Stage)
	s.Equal(2, unanswered.Attempt)
	s.Require().NotNil(unanswered.NextAttemptAt)
	s.Equal(died.Add(3*24*time.Hour), unanswered.NextAttemptAt.UTC())
	s.Equal(string(Unanswered), attempts[0].Result)
}

func (s *RunnerSuite) TestAContactClaimedBeforeItWasEverDialedGoesBackUnused() {
	campaign := s.launched(s.single(), 1, "UTC", "+15550000001")
	_, err := s.store.ClaimDueContacts(s.ctx, "router-dead", s.clock.now(), lease, 10)
	s.Require().NoError(err)
	s.clock.advance(lease + time.Minute)

	s.runner.takeOver(s.ctx)

	contact, attempts := s.contact(campaign, "+15550000001")
	s.Equal(store.Enrolled, contact.Stage)
	s.Equal(0, contact.Attempt)
	s.Empty(attempts)
}

func (s *RunnerSuite) TestATakeOverOfAnAttemptNeverDialedLeavesItUnused() {
	campaign := s.launched(s.single(), 1, "UTC", "+15550000001")
	s.tick(s.runner)
	// Attempt 2 comes due, and a router claims it and dies before dialing it.
	s.clock.advance(3 * 24 * time.Hour)
	_, err := s.store.ClaimDueContacts(s.ctx, "router-dead", s.clock.now(), lease, 10)
	s.Require().NoError(err)
	s.clock.advance(lease + time.Minute)

	s.runner.takeOver(s.ctx)

	contact, attempts := s.contact(campaign, "+15550000001")
	s.Len(attempts, 1, "only attempt 1 was ever dialed")
	s.Equal(store.Enrolled, contact.Stage)
	s.Equal(2, contact.Attempt, "attempt 2 was never dialed, so it is not used up")
}

func (s *RunnerSuite) TestAPauseDuringTheFirstOfADoubleDialStopsTheSecond() {
	s.dialer.hold = make(chan struct{})
	campaign := s.launched(DefaultSequence("config-1"), 1, "UTC", "+15550000001")

	s.runner.tick(s.ctx)
	s.Require().Eventually(func() bool { return s.dialer.inFlight() == 1 }, 5*time.Second, 10*time.Millisecond)
	s.Require().NoError(s.store.PauseCampaign(s.ctx, s.customerID, campaign.ID, ""))
	close(s.dialer.hold)
	s.runner.dialing.Wait()

	s.Len(s.dialer.dialed(), 1, "nobody is rung once the campaign is paused")
	contact, _ := s.contact(campaign, "+15550000001")
	s.Equal(store.Enrolled, contact.Stage)
	s.Equal(2, contact.Attempt, "the first dial rang them, so the attempt is used")
	paused := s.campaignNow(campaign)
	s.Equal(store.Paused, paused.State)
	s.Empty(paused.PausedReason, "somebody paused it; the runner adds no reason of its own")
}

func (s *RunnerSuite) TestAContactOfACampaignThatCannotBeReadFails() {
	campaign := store.Campaign{CustomerID: s.customerID, Name: "broken", ConfigID: "config-1", FromNumber: "+15550100000",
		Sequence: json.RawMessage(`{"bogus": true}`), CallingHours: json.RawMessage(`{"windows": {}}`)}
	s.Require().NoError(s.store.CreateCampaign(s.ctx, &campaign))
	_, _, err := s.store.AddContacts(s.ctx, campaign.ID, []store.Contact{{ToNumber: "+15550000001"}})
	s.Require().NoError(err)
	_, err = s.store.LaunchCampaign(s.ctx, s.customerID, campaign.ID, store.Launch{StartAt: s.clock.now(), Now: s.clock.now()})
	s.Require().NoError(err)

	s.tick(s.runner)

	s.Empty(s.dialer.dialed())
	contact, _ := s.contact(campaign, "+15550000001")
	s.Equal(store.Failed, contact.Stage, "not held until its lease runs out, only to fail the same way")
	s.Equal("invalid_plan", contact.Reason)
}

func (s *RunnerSuite) TestAContactClaimedJustBeforeAPauseIsNotDialed() {
	campaign := s.launched(s.single(), 1, "UTC", "+15550000001")
	claimed, err := s.store.ClaimDueContacts(s.ctx, "router-a", s.clock.now(), lease, 10)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)
	s.Require().NoError(s.store.PauseCampaign(s.ctx, s.customerID, campaign.ID, ""))

	s.runner.attempt(s.ctx, claimed[0])

	s.Empty(s.dialer.dialed())
	contact, attempts := s.contact(campaign, "+15550000001")
	s.Equal(store.Enrolled, contact.Stage)
	s.Equal(0, contact.Attempt)
	s.Empty(attempts)
	s.Require().NotNil(contact.NextAttemptAt)
	s.Equal(s.clock.now(), contact.NextAttemptAt.UTC(), "rung as soon as the campaign resumes")
}

func (s *RunnerSuite) TestAPausedCampaignFinishesItsCallAndSchedulesTheNext() {
	s.dialer.hold = make(chan struct{})
	campaign := s.launched(s.single(), 1, "UTC", "+15550000001")

	s.runner.tick(s.ctx)
	s.Require().Eventually(func() bool { return s.dialer.inFlight() == 1 }, 5*time.Second, 10*time.Millisecond)
	s.Require().NoError(s.store.PauseCampaign(s.ctx, s.customerID, campaign.ID, ""))
	close(s.dialer.hold)
	s.runner.dialing.Wait()

	contact, attempts := s.contact(campaign, "+15550000001")
	s.Len(attempts, 1)
	s.Equal(store.Enrolled, contact.Stage)
	s.Equal(2, contact.Attempt)
	s.Require().NotNil(contact.NextAttemptAt)
	s.Equal(s.clock.now().Add(3*24*time.Hour), contact.NextAttemptAt.UTC())
	s.Equal(store.Paused, s.campaignNow(campaign).State)
}

func (s *RunnerSuite) TestAStoppedCampaignsContactInACallEndsStopped() {
	s.dialer.hold = make(chan struct{})
	campaign := s.launched(s.single(), 1, "UTC", "+15550000001")

	s.runner.tick(s.ctx)
	s.Require().Eventually(func() bool { return s.dialer.inFlight() == 1 }, 5*time.Second, 10*time.Millisecond)
	s.Require().NoError(s.store.StopCampaign(s.ctx, s.customerID, campaign.ID, s.clock.now()))
	close(s.dialer.hold)
	s.runner.dialing.Wait()

	contact, attempts := s.contact(campaign, "+15550000001")
	s.Len(attempts, 1)
	s.Equal(store.ContactStopped, contact.Stage)
	s.Nil(contact.NextAttemptAt)
}

func (s *RunnerSuite) TestAnEmptyTranscriptIsReadAgainBeforeFallingBack() {
	s.dialer.answer("+15550000001")
	campaign := s.launched(s.single(), 1, "UTC", "+15550000001")
	s.tick(s.runner)
	s.clock.advance(11 * time.Second)

	for read := 1; read < maxClassifyTries; read++ {
		s.tick(s.runner)
		contact, _ := s.contact(campaign, "+15550000001")
		s.Equal(store.AwaitingOutcome, contact.Stage, "read %d", read)
		s.Empty(contact.LeasedBy, "read %d leaves the contact free to be read again", read)
		s.clock.advance(classifyRetry)
	}
	s.tick(s.runner)

	contact, attempts := s.contact(campaign, "+15550000001")
	s.Equal(store.Enrolled, contact.Stage, "nobody ever spoke, which is an unanswered call")
	s.Equal(2, contact.Attempt)
	s.Equal(string(Unanswered), attempts[0].Result)
	s.Equal(maxClassifyTries, attempts[0].ClassifyTries)
}

func (s *RunnerSuite) TestATranscriptWrittenLateIsReadSoonAfter() {
	s.dialer.answer("+15550000001")
	campaign := s.launched(s.single(), 1, "UTC", "+15550000001")
	s.tick(s.runner)
	s.clock.advance(11 * time.Second)
	s.tick(s.runner)
	_, attempts := s.contact(campaign, "+15550000001")
	s.transcripts.write(attempts[0].CallID, talked())

	s.clock.advance(classifyRetry)
	s.tick(s.runner)

	contact, _ := s.contact(campaign, "+15550000001")
	s.Equal(store.ContactCompleted, contact.Stage)
	s.Equal(OutcomeSuccess, contact.Outcome)
}

func (s *RunnerSuite) TestACallWhereOnlyTheAgentSpokeIsUnanswered() {
	s.dialer.answer("+15550000001")
	campaign := s.launched(s.single(), 1, "UTC", "+15550000001")
	s.tick(s.runner)
	_, attempts := s.contact(campaign, "+15550000001")
	s.transcripts.write(attempts[0].CallID, []chatlog.Spoken{{Agent: true, Text: "Hello? Is anyone there?"}})

	s.clock.advance(11 * time.Second)
	s.tick(s.runner)

	contact, attempts := s.contact(campaign, "+15550000001")
	s.Equal(store.Enrolled, contact.Stage)
	s.Equal(2, contact.Attempt)
	s.Empty(contact.Outcome)
	s.Equal(string(Unanswered), attempts[0].Result)
}

func (s *RunnerSuite) TestARefusalOfMoneyPausesTheCampaign() {
	s.dialer.results["+15550000001"] = []Dialed{{Result: Paused, Reason: "insufficient_credits"}}
	campaign := s.launched(s.single(), 1, "UTC", "+15550000001")

	s.tick(s.runner)

	paused := s.campaignNow(campaign)
	s.Equal(store.Paused, paused.State)
	s.Equal("insufficient_credits", paused.PausedReason)
	contact, _ := s.contact(campaign, "+15550000001")
	s.Equal(store.Enrolled, contact.Stage)
	s.Equal(0, contact.Attempt, "nobody was rung")
}

func (s *RunnerSuite) TestARefusalOfMoneyOnTheSecondDialStillPausesTheCampaign() {
	s.dialer.results["+15550000001"] = []Dialed{{Result: Unanswered}, {Result: Paused, Reason: "insufficient_credits"}}
	campaign := s.launched(DefaultSequence("config-1"), 1, "UTC", "+15550000001")

	s.tick(s.runner)

	paused := s.campaignNow(campaign)
	s.Equal(store.Paused, paused.State)
	s.Equal("insufficient_credits", paused.PausedReason)
	contact, _ := s.contact(campaign, "+15550000001")
	s.Equal(2, contact.Attempt, "the first dial rang them, so the attempt is used")
}

func (s *RunnerSuite) TestACampaignWhoseConfigIsGonePausesWithoutUsingAnAttempt() {
	// The phone dialer reads the config before it touches the phone, so it needs none here.
	runner, err := New(Options{
		Store: s.store, Dialer: NewPhoneDialer(nil, nil, s.store, slog.New(slog.DiscardHandler)),
		Transcripts: s.transcripts, Classifier: s.classifier, Owner: "router-p", Now: s.clock.now, Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(runner.Close)
	campaign := s.launched(s.single(), 2, "UTC", "+15550000001", "+15550000002")

	s.tick(runner)

	paused := s.campaignNow(campaign)
	s.Equal(store.Paused, paused.State, "every other contact would fail the same way")
	s.Equal("config_missing", paused.PausedReason)
	for _, number := range []string{"+15550000001", "+15550000002"} {
		contact, _ := s.contact(campaign, number)
		s.Equal(store.Enrolled, contact.Stage, number)
		s.Equal(0, contact.Attempt, number)
	}
}

func (s *RunnerSuite) TestTheLastContactCompletesTheCampaign() {
	s.dialer.answer("+15550000001")
	campaign := s.launched(s.single(), 1, "UTC", "+15550000001")

	s.tick(s.runner)
	s.Equal(store.Running, s.campaignNow(campaign).State, "an outcome is still to be read")
	_, attempts := s.contact(campaign, "+15550000001")
	s.transcripts.write(attempts[0].CallID, talked())
	s.clock.advance(11 * time.Second)
	s.tick(s.runner)

	completed := s.campaignNow(campaign)
	s.Equal(store.Completed, completed.State)
	s.NotNil(completed.FinishedAt)
}

func (s *RunnerSuite) TestCloseWaitsForACallInFlight() {
	s.dialer.hold = make(chan struct{})
	campaign := s.launched(s.single(), 1, "UTC", "+15550000001")
	runner, err := New(Options{
		Store: s.store, Dialer: s.dialer, Transcripts: s.transcripts, Classifier: s.classifier,
		Owner: "router-c", Now: s.clock.now, Tick: 10 * time.Millisecond, Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.Require().NoError(runner.Start(s.ctx))
	s.Require().Eventually(func() bool { return s.dialer.inFlight() == 1 }, 5*time.Second, 10*time.Millisecond)

	closed := make(chan struct{})
	go func() {
		runner.Close()
		close(closed)
	}()
	s.Never(func() bool {
		select {
		case <-closed:
			return true
		default:
			return false
		}
	}, 200*time.Millisecond, 10*time.Millisecond, "a person on the phone is not hung up on")
	close(s.dialer.hold)
	s.Eventually(func() bool {
		select {
		case <-closed:
			return true
		default:
			return false
		}
	}, 5*time.Second, 10*time.Millisecond)

	contact, _ := s.contact(campaign, "+15550000001")
	s.Equal(store.Enrolled, contact.Stage, "the call that finished during Close is still written down")
	s.Equal(2, contact.Attempt)
}

func (s *RunnerSuite) TestAStoppingRouterDoesNotDialAgainAfterAnUnansweredFirstDial() {
	// Its session is ended at the stop, which is why dial 1 comes back unanswered.
	s.dialer.hold = make(chan struct{})
	campaign := s.launched(DefaultSequence("config-1"), 1, "UTC", "+15550000001")
	ctx, stop := context.WithCancel(s.ctx)
	defer stop()
	runner, err := New(Options{
		Store: s.store, Dialer: s.dialer, Transcripts: s.transcripts, Classifier: s.classifier,
		Owner: "router-d", Now: s.clock.now, Tick: 10 * time.Millisecond, Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(runner.Close)
	s.Require().NoError(runner.Start(ctx))
	s.Require().Eventually(func() bool { return s.dialer.inFlight() == 1 }, 5*time.Second, 10*time.Millisecond)

	stop()
	close(s.dialer.hold)
	s.Require().Eventually(func() bool { return runner.Running() == 0 }, 5*time.Second, 10*time.Millisecond)

	s.Len(s.dialer.dialed(), 1, "nobody is rung again by a router whose sessions are ending")
	contact, attempts := s.contact(campaign, "+15550000001")
	s.Require().Len(attempts, 1)
	s.Equal(1, attempts[0].Dial)
	s.Equal(store.Enrolled, contact.Stage)
	s.Require().NotNil(contact.NextAttemptAt)
	s.True(contact.NextAttemptAt.After(s.clock.now()), "dial 1 rang them, so the attempt is used and the next one waits")
}

func (s *RunnerSuite) TestAStoppingRouterRingsNoContactItAlreadyClaimed() {
	campaign := s.launched(s.single(), 1, "UTC", "+15550000001")
	stopped, stop := context.WithCancel(s.ctx)
	stop()
	runner := s.newRunner("router-e")
	s.Require().NoError(runner.Start(stopped))
	s.Require().Eventually(runner.stopping, 5*time.Second, 10*time.Millisecond)
	claimed, err := s.store.ClaimDueContacts(s.ctx, "router-e", s.clock.now(), lease, 10)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)

	runner.attempt(s.ctx, claimed[0])

	s.Empty(s.dialer.dialed())
	contact, attempts := s.contact(campaign, "+15550000001")
	s.Equal(store.Enrolled, contact.Stage, "given back, to be rung by a router that is not stopping")
	s.Equal(0, contact.Attempt)
	s.Empty(attempts)
}

// clock is a time the test moves by hand.
type clock struct {
	mu sync.Mutex
	at time.Time
}

func (c *clock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.at
}

func (c *clock) set(at time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.at = at
}

func (c *clock) advance(by time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.at = c.at.Add(by)
}

// stubDialer ends each dial as results says for its number, in order, and unanswered after
// them. While hold is open, every dial waits for it to close.
type stubDialer struct {
	mu      sync.Mutex
	results map[string][]Dialed
	hold    chan struct{}
	dials   []Dial
	now     int
	highest int
}

func (d *stubDialer) answer(number string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.results[number] = []Dialed{{Result: Answered}}
}

func (d *stubDialer) Dial(ctx context.Context, dial Dial) (Dialed, error) {
	d.mu.Lock()
	d.dials = append(d.dials, dial)
	d.now++
	d.highest = max(d.highest, d.now)
	dialed := Dialed{Result: Unanswered}
	if queued := d.results[dial.Contact.ToNumber]; len(queued) > 0 {
		dialed, d.results[dial.Contact.ToNumber] = queued[0], queued[1:]
	}
	hold := d.hold
	d.mu.Unlock()

	if hold != nil {
		<-hold
	}

	d.mu.Lock()
	d.now--
	d.mu.Unlock()
	if dialed.Result == Answered || dialed.Result == Unanswered {
		dialed.CallID = dial.SessionID
		dialed.VendorCallID = "vendor-" + dial.SessionID
	}
	return dialed, nil
}

func (d *stubDialer) dialed() []Dial {
	d.mu.Lock()
	defer d.mu.Unlock()
	return append([]Dial(nil), d.dials...)
}

func (d *stubDialer) inFlight() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.now
}

func (d *stubDialer) most() int {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.highest
}

// stubTranscripts is what was said on each call, as written so far.
type stubTranscripts struct {
	mu   sync.Mutex
	said map[string][][]chatlog.Spoken
}

func (t *stubTranscripts) write(callID string, said []chatlog.Spoken) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.said[callID] = append(t.said[callID], said)
}

func (t *stubTranscripts) CallTranscript(_ context.Context, _, callID string) ([]chatlog.Spoken, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	written := t.said[callID]
	if len(written) == 0 {
		return nil, nil
	}
	return written[len(written)-1], nil
}

// stubClassifier picks outcome for every call it reads.
type stubClassifier struct {
	outcome string
}

func (c *stubClassifier) Classify(context.Context, store.Campaign, string, Step, []chatlog.Spoken) (string, error) {
	return c.outcome, nil
}
