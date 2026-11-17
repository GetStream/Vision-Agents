//go:build integration

package session

// TestARealCallIsCountedAndTimedUnderCustom: a call through the dispatcher, of the suite's
// custom connector, is counted once under custom as ok and timed.
func (s *CallLogSuite) TestARealCallIsCountedAndTimedUnderCustom() {
	calls, timed := callsOf("custom", "ok"), timedCalls("custom").GetSampleCount()
	app := s.connection("", "primary")
	d, _ := s.logging(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))

	_, err := s.call(d, "crm__whoami", "{}")

	s.Require().NoError(err)
	s.Empty(s.row(app).ErrorType)
	s.Equal(calls+1, callsOf("custom", "ok"))
	s.Equal(timed+1, timedCalls("custom").GetSampleCount())
}
