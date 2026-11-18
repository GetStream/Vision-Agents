//go:build integration

package campaign

import (
	"context"
	"database/sql"
	"errors"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/activity"
)

// dialsCounted is the campaign's app's phone_sms row in the activity record.
type dialsCounted struct {
	requests, refused int64
	code              string
}

// failingDialer could not try any dial.
type failingDialer struct{}

func (failingDialer) Dial(context.Context, Dial) (Dialed, error) {
	return Dialed{}, errors.New("the vendor did not answer")
}

// counting is a runner over dialer, as newRunner makes, that counts in an activity record of its
// own, which writes only when the test calls the write it returns.
func (s *RunnerSuite) counting(dialer Dialer) (runner *Runner, write func()) {
	record, err := activity.New(activity.Options{Store: s.store, Interval: time.Hour, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	runner, err = New(Options{
		Store: s.store, Dialer: dialer, Transcripts: s.transcripts, Classifier: s.classifier,
		Owner: "router-counting", Now: s.clock.now, Activity: record, Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(runner.Close)
	return runner, record.Close
}

// counted is what the activity record holds for the suite's app under phone_sms and client api.
func (s *RunnerSuite) counted() dialsCounted {
	var (
		row  dialsCounted
		code sql.NullString
	)
	err := s.store.DB().QueryRowContext(s.ctx, `
SELECT COALESCE(sum(requests), 0), COALESCE(sum(refused), 0), max(last_refusal_code)
FROM customer_activity WHERE app_id = ? AND area = ? AND client = ?`,
		s.customerID, activity.AreaPhoneSMS, activity.ClientRouter).Scan(&row.requests, &row.refused, &code)
	s.Require().NoError(err)
	row.code = code.String
	return row
}

// Defect: a call nobody answered counted refused, or the second dial of a double dial, which
// rings the person again, left uncounted.
func (s *RunnerSuite) TestEachDialOfADoubleDialIsCountedAndARingNobodyAnsweredIsNotRefused() {
	runner, write := s.counting(s.dialer)
	s.launched(DefaultSequence("config-1"), 1, "UTC", "+15550000001")

	s.tick(runner)
	write()

	s.Require().Len(s.dialer.dialed(), 2)
	s.Equal(dialsCounted{requests: 2}, s.counted())
}

// Defect: a call an answering machine took, which the dial placed, counted refused.
func (s *RunnerSuite) TestADialAMachineAnsweredIsNotCountedRefused() {
	s.dialer.results["+15550000001"] = []Dialed{{Result: Voicemail}}
	runner, write := s.counting(s.dialer)
	s.launched(s.single(), 1, "UTC", "+15550000001")

	s.tick(runner)
	write()

	s.Equal(dialsCounted{requests: 1}, s.counted())
}

// Defect: a dial refused for money, which placed no call, counted as placed, or refused without
// the reason the campaign paused under.
func (s *RunnerSuite) TestADialRefusedForMoneyIsCountedRefusedUnderItsReason() {
	s.dialer.results["+15550000001"] = []Dialed{{Result: Paused, Reason: "insufficient_credits"}}
	runner, write := s.counting(s.dialer)
	s.launched(s.single(), 1, "UTC", "+15550000001")

	s.tick(runner)
	write()

	s.Equal(dialsCounted{requests: 1, refused: 1, code: "insufficient_credits"}, s.counted())
}

// Defect: a dial that placed no call and gave no reason counted refused with no code, so the
// digest cannot say why.
func (s *RunnerSuite) TestADialToANumberThatCannotBeDialedIsCountedRefusedUnderItsResult() {
	s.dialer.results["+15550000001"] = []Dialed{{Result: InvalidNumber}}
	runner, write := s.counting(s.dialer)
	s.launched(s.single(), 1, "UTC", "+15550000001")

	s.tick(runner)
	write()

	s.Equal(dialsCounted{requests: 1, refused: 1, code: string(InvalidNumber)}, s.counted())
}

// Defect: a dial that could not be tried, which may have rung nobody, counted as a call placed.
func (s *RunnerSuite) TestADialThatCouldNotBeTriedIsCountedRefused() {
	runner, write := s.counting(failingDialer{})
	s.launched(s.single(), 1, "UTC", "+15550000001")

	s.tick(runner)
	write()

	s.Equal(dialsCounted{requests: 1, refused: 1, code: codeDialFailed}, s.counted())
}

// Defect: a dial a stopping router never placed counted refused, so the contact, rung by
// another router later, reads as two dials, one of them refused.
func (s *RunnerSuite) TestADialTheRouterDidNotPlaceAsItStopsIsNotCounted() {
	s.launched(s.single(), 1, "UTC", "+15550000001")
	runner, write := s.counting(s.dialer)
	stopped, stop := context.WithCancel(s.ctx)
	stop()
	s.Require().NoError(runner.Start(stopped))
	s.Require().Eventually(runner.stopping, 5*time.Second, 10*time.Millisecond)
	claimed, err := s.store.ClaimDueContacts(s.ctx, "router-counting", s.clock.now(), lease, 10)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)

	dialed, err := runner.ring(s.ctx, claimed[0].Campaign, claimed[0].Contact, s.single().First(), 1)
	s.Require().NoError(err)
	write()

	s.Require().Equal(NotPlaced, dialed.Result)
	s.Equal(dialsCounted{}, s.counted())
}
