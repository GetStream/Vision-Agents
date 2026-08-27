//go:build integration

package phone

import (
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/suite"
)

// Event hooks are one setting on the whole app, so this runs against a real one: the thing
// worth proving is that pointing a call hook somewhere leaves the app's other hooks alone,
// and that cannot be shown against a stand-in that has none.
//
// The suite puts back exactly what it found, so an app configured for something else is not
// left changed by a test run.

type CallHookIntegrationSuite struct {
	suite.Suite
	ctx    context.Context
	stream *Stream
	// found is what the app had before the suite touched anything.
	found []CallHook
}

func TestCallHookIntegrationSuite(t *testing.T) {
	suite.Run(t, new(CallHookIntegrationSuite))
}

func (s *CallHookIntegrationSuite) SetupSuite() {
	if os.Getenv(apiKeyEnvVar) == "" || os.Getenv(apiSecretEnvVar) == "" {
		s.T().Skipf("%s and %s must be set", apiKeyEnvVar, apiSecretEnvVar)
	}

	s.ctx = context.Background()
	stream, err := NewStream(StreamOptions{})
	s.Require().NoError(err)
	s.stream = stream

	hooks, err := stream.CallHooks(s.ctx)
	s.Require().NoError(err)
	s.found = hooks
}

// SetupTest takes the suite's own hook back down, so each test starts from an app that has
// whatever it really has and nothing pointing at this suite's url. Otherwise whether a hook
// was added or updated would depend on which test ran first.
func (s *CallHookIntegrationSuite) SetupTest() {
	for _, hook := range s.found {
		if hook.URL == s.url() {
			return
		}
	}
	_, err := s.stream.RemoveCallHook(s.ctx, s.url())
	s.Require().NoError(err)
}

// TearDownSuite removes the hook the suite added, leaving the app as it was found.
func (s *CallHookIntegrationSuite) TearDownSuite() {
	if s.stream == nil {
		return
	}
	for _, hook := range s.found {
		if hook.URL == s.url() {
			return // It was already there, so leaving it is putting it back.
		}
	}
	_, err := s.stream.RemoveCallHook(s.ctx, s.url())
	s.Require().NoError(err)
}

// url is where the suite points a hook. Nothing waits for a delivery there, and the hook is
// removed again in teardown.
//
// It has to be a host that resolves: Stream refuses a hook it cannot reach, so a .invalid
// name or a made-up subdomain fails validation rather than being stored. example.com is
// reserved for exactly this and discards what it is sent, and the path says whose it is.
func (s *CallHookIntegrationSuite) url() string {
	return "https://example.com/vision-agents-hook-test" + CallHookPath
}

func (s *CallHookIntegrationSuite) TestPointingACallHookAsksForTheCallEvents() {
	added, err := s.stream.PointCallHook(s.ctx, s.url())
	s.Require().NoError(err)
	s.False(added, "nothing was pointing at this url yet, so it was added rather than updated")

	hooks, err := s.stream.CallHooks(s.ctx)
	s.Require().NoError(err)

	var ours *CallHook
	for index, hook := range hooks {
		if hook.URL == s.url() {
			ours = &hooks[index]
		}
	}
	s.Require().NotNil(ours, "the hook that was just set has to be readable")
	s.True(ours.Enabled)
	s.Equal("webhook", ours.HookType)
	s.ElementsMatch(callHookEvents, ours.EventTypes,
		"a hook asking for everything would deliver every message in the app here")
}

func (s *CallHookIntegrationSuite) TestPointingTheSameHookTwiceDoesNotAddASecond() {
	_, err := s.stream.PointCallHook(s.ctx, s.url())
	s.Require().NoError(err)

	updated, err := s.stream.PointCallHook(s.ctx, s.url())
	s.Require().NoError(err)
	s.True(updated, "the second call has to update the hook rather than duplicate it")

	hooks, err := s.stream.CallHooks(s.ctx)
	s.Require().NoError(err)

	matching := 0
	for _, hook := range hooks {
		if hook.URL == s.url() {
			matching++
		}
	}
	s.Equal(1, matching, "two hooks on one url would deliver every call twice")
}

func (s *CallHookIntegrationSuite) TestHooksTheAppAlreadyHadAreLeftAlone() {
	_, err := s.stream.PointCallHook(s.ctx, s.url())
	s.Require().NoError(err)

	hooks, err := s.stream.CallHooks(s.ctx)
	s.Require().NoError(err)

	after := map[string]bool{}
	for _, hook := range hooks {
		after[hook.Destination] = true
	}
	for _, hook := range s.found {
		s.True(after[hook.Destination],
			"writing the call hook must not turn off a hook the app was already using")
	}
}

func (s *CallHookIntegrationSuite) TestAHookNeedsSomewhereStreamCanReach() {
	_, err := s.stream.PointCallHook(s.ctx, "")
	s.ErrorContains(err, "needs a url")

	_, err = s.stream.PointCallHook(s.ctx, "example.ngrok.app/v1/phone/hooks/stream")
	s.ErrorContains(err, "is not a url Stream can reach")
}

func (s *CallHookIntegrationSuite) TestRemovingAHookThatIsNotThereChangesNothing() {
	removed, err := s.stream.RemoveCallHook(s.ctx, "https://nothing-points-here.invalid/hook")

	s.Require().NoError(err)
	s.False(removed)
}

func (s *CallHookIntegrationSuite) TestAHookCanBeTakenBackDown() {
	_, err := s.stream.PointCallHook(s.ctx, s.url())
	s.Require().NoError(err)

	removed, err := s.stream.RemoveCallHook(s.ctx, s.url())
	s.Require().NoError(err)
	s.True(removed)

	hooks, err := s.stream.CallHooks(s.ctx)
	s.Require().NoError(err)
	for _, hook := range hooks {
		s.NotEqual(s.url(), hook.URL, "a tunnel that is gone should not still be delivered to")
	}
}
