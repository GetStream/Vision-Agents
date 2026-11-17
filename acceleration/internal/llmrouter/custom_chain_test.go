package llmrouter

import (
	"context"
	"errors"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// ownModels is a customer who serves one model of their own, called own.
type ownModels struct{}

func (ownModels) CustomModel(_ context.Context, _, name string) (routing.ProviderConfig, routing.Endpoint, error) {
	if name != "own" {
		return routing.ProviderConfig{}, routing.Endpoint{}, errors.New("routing: not one of this customer's models")
	}
	return routing.ProviderConfig{Provider: routing.CustomProvider, Model: name},
		routing.Endpoint{BaseURL: "https://8.8.8.8/v1", Model: "acme/tuned-27b"}, nil
}

// newChainRig opens a session on the customer's own model alone, or on it and then beta/steady
// as a priority list, the way an agent config's llm and llm_fallbacks reach the router.
func newChainRig(t *testing.T, hedge time.Duration, providers ...string) *rig {
	t.Helper()
	r := &rig{
		headers: map[string]time.Duration{},
		failing: map[string]error{},
		scripts: map[string][]*llmtest.Script{},
	}
	registry := NewRegistry()
	for _, name := range []string{routing.CustomProvider, "beta"} {
		registry.Register(name, func(spec routing.Spec) (Provider, error) {
			return &pace{rig: r, name: name, model: spec.Model}, nil
		})
	}
	config := routing.ModalityConfig{Providers: []routing.ProviderConfig{
		{Provider: "beta", Model: "steady", Languages: []string{"en"}},
	}}
	router, err := New(Options{Config: config, Registry: registry, Models: ownModels{}, ReplyHedge: hedge})
	require.NoError(t, err)
	r.session, err = router.Start(context.Background(), Request{CustomerID: "acme", Target: "custom/own", Providers: providers})
	require.NoError(t, err)
	t.Cleanup(func() {
		require.NoError(t, r.session.Close())
		router.Close()
	})
	return r
}

func TestACustomModelThatFailsAReplyIsServedByTheNextEntry(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := newChainRig(t, 0, "custom/own", "beta/steady")
		require.Equal(t, routing.CustomProvider, r.session.Provider(), "the customer's own model is the first choice")
		r.fail(routing.CustomProvider, errors.New("custom: 503 the deployment is scaling up"))

		asked := r.ask(context.Background(), r.params())
		require.NoError(t, asked.err)
		r.say("beta", "Hello")

		response := r.finish("beta", asked)
		require.Equal(t, "beta", response.Provider)
		require.Equal(t, "Hello", response.OutputText)
		require.Equal(t, map[string]string{routing.CustomProvider: "create_failed", "beta": "served"}, r.called(),
			"the custom model was asked first")
	})
}

func TestACustomModelAloneHasNothingToFallBackTo(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := newChainRig(t, 0)
		r.fail(routing.CustomProvider, errors.New("custom: 503 the deployment is scaling up"))

		asked := r.ask(context.Background(), r.params())

		require.Error(t, asked.err)
		require.Zero(t, r.requests("beta"), "a single custom target resolves to itself alone")
	})
}

func TestACustomModelSilentPastTheHedgeIsOvertakenByTheNextEntry(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := newChainRig(t, lateAfter, "custom/own", "beta/steady")
		asked := r.ask(context.Background(), r.params())

		pass(lateAfter)
		require.Equal(t, 1, r.requests("beta"), "a cold custom model that has said nothing is asked of beta as well")
		r.say("beta", "Hello")
		require.True(t, asked.answered())
		require.NoError(t, asked.err)

		require.True(t, r.script(routing.CustomProvider).Abandoned(), "the custom model lost the race and was stopped")
		require.Equal(t, "beta", r.finish("beta", asked).Provider)
	})
}

func TestACustomModelThatSpeaksSlowlyIsNeverOvertaken(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := newChainRig(t, lateAfter, "custom/own", "beta/steady")
		asked := r.ask(context.Background(), r.params())

		pass(lateAfter / 2)
		r.say(routing.CustomProvider, "One")
		require.True(t, asked.answered())
		pass(time.Minute)
		r.say(routing.CustomProvider, " token at a time")

		require.Zero(t, r.requests("beta"), "once it has said something, however slowly it goes on, it is not hedged")
		response := r.finish(routing.CustomProvider, asked)
		require.Equal(t, routing.CustomProvider, response.Provider)
		require.Equal(t, "One token at a time", response.OutputText)
	})
}
