//go:build integration

package session

import (
	"cmp"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// TestEachCallIsCountedUnderItsRowsErrorType: real calls through the dispatcher, of the
// suite's custom connector, are each counted once under custom, as ok or their row's
// error_type, and timed.
func (s *CallLogSuite) TestEachCallIsCountedUnderItsRowsErrorType() {
	counted := func(outcome string, call func() string) {
		calls, timed := callsOf("custom", outcome), timedCalls("custom").GetSampleCount()

		connection := call()

		s.Equal(outcome, cmp.Or(s.row(connection).ErrorType, "ok"))
		s.Equal(calls+1, callsOf("custom", outcome))
		s.Equal(timed+1, timedCalls("custom").GetSampleCount())
	}
	s.Run("answered", func() {
		counted("ok", func() string {
			app := s.connection("", "primary")
			d, _ := s.logging(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
			_, _ = s.call(d, "crm__whoami", "{}")
			return app
		})
	})
	s.Run("isError", func() {
		counted(store.InvocationExternalServer, func() string {
			app := s.connection("", "primary")
			d, _ := s.logging(s.spec(s.config(s.fixed("crm", app, "fails")), "", nil))
			_, _ = s.call(d, "crm__fails", "{}")
			return app
		})
	})
	s.Run("401", func() {
		counted(store.InvocationCustomerAuth, func() string {
			app := s.connection("", "primary")
			d, _ := s.logging(s.spec(s.config(s.fixed("crm", app, "whoami")), "", nil))
			s.sendWrongToken(app)
			_, _ = s.call(d, "crm__whoami", "{}")
			return app
		})
	})
	s.Run("refused", func() {
		counted(store.InvocationDenied, func() string {
			app := s.connection("", "primary")
			config := s.config(s.fixed("crm", app, "whoami"))
			d, _ := s.logging(s.spec(config, "", nil))
			s.rebind(config, s.fixed("crm", app))
			_, _ = s.call(d, "crm__whoami", "{}")
			return app
		})
	})
	s.Run("timeout", func() {
		counted(store.InvocationOutcomeUnknown, func() string {
			app := s.connection("", "primary")
			binding := s.fixed("crm", app, "slow")
			binding.TimeoutMs = 200
			d, _ := s.logging(s.spec(s.config(binding), "", nil))
			_, _ = s.call(d, "crm__slow", "{}")
			return app
		})
	})
}
