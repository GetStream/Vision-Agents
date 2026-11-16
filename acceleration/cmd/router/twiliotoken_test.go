package main

import (
	"log/slog"
	"os"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
)

// TwilioTokenStartupSuite starts the router with TWILIO_WEBHOOK_AUTH_TOKEN set, as its serve
// command does, with no database, Redis or Stream app.
type TwilioTokenStartupSuite struct {
	suite.Suite
}

func TestTwilioTokenStartupSuite(t *testing.T) {
	suite.Run(t, new(TwilioTokenStartupSuite))
}

// A token pasted with its newline stops the router at start, rather than leaving it to
// refuse every webhook Twilio sends. The error names the setting and not the token.
func (s *TwilioTokenStartupSuite) TestATokenWithANewlineStopsTheRouter() {
	settings := config.Defaults()
	settings.Auth.Mode = string(auth.NoAuth)
	settings.Limits.Mode = config.LimitsOff
	settings.Addr = "127.0.0.1:0"
	settings.Twilio.WebhookAuthToken = "token-with-a-newline\n"

	stopped := make(chan error, 1)
	go func() { stopped <- run(settings, slog.New(slog.DiscardHandler)) }()

	select {
	case err := <-stopped:
		s.ErrorContains(err, "TWILIO_WEBHOOK_AUTH_TOKEN")
		s.NotContains(err.Error(), "token-with-a-newline")
	case <-time.After(10 * time.Second):
		s.Fail("the router started with a token that cannot be Twilio's")
		s.NoError(syscall.Kill(os.Getpid(), syscall.SIGTERM))
		<-stopped
	}
}
