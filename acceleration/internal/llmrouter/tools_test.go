package llmrouter

import (
	"context"
	"errors"
	"testing"
	"testing/synctest"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// withTools is a reply offering the agent's tools, as a turn on a call does.
func (r *rig) withTools() llm.ResponseParams {
	params := r.params()
	params.Tools = []llm.Tool{{Name: "transfer", Parameters: map[string]any{"type": "object"}}}
	return params
}

func TestALateReplyOfferingToolsIsNotHedgedOnAModelThatIgnoresThem(t *testing.T) {
	for name, test := range map[string]struct {
		params func(*rig) llm.ResponseParams
		hedge  string
	}{
		"offering tools": {(*rig).withTools, "beta"},
		"offering none":  {(*rig).params, "gamma"},
	} {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				r := newRig(t, lateAfter, "alpha", "gamma", "beta")
				asked := r.ask(context.Background(), test.params(r))

				pass(lateAfter)
				r.say(test.hedge, "Hello")
				require.True(t, asked.answered())
				require.NoError(t, asked.err)
				require.Equal(t, test.hedge, r.finish(test.hedge, asked).Provider)
			})
		})
	}
}

func TestAFailedReplyOfferingToolsFallsBackPastAModelThatIgnoresThem(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := newRig(t, 0, "alpha", "gamma", "beta")
		r.fail("alpha", errors.New("alpha is down"))
		asked := r.ask(context.Background(), r.withTools())

		require.True(t, asked.answered())
		require.NoError(t, asked.err)
		require.Zero(t, r.requests("gamma"), "it would have answered without the tool")
		r.say("beta", "Transferring you now")
		require.Equal(t, "beta", r.finish("beta", asked).Provider)
	})
}

func TestASessionOnAModelThatIgnoresToolsAsksAnotherWhenAReplyOffersThem(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := newRig(t, 0, "gamma", "beta")
		require.Equal(t, "gamma", r.session.Provider())

		asked := r.ask(context.Background(), r.withTools())
		require.NoError(t, asked.err)
		require.Zero(t, r.requests("gamma"))
		r.say("beta", "Transferring you now")
		require.Equal(t, "beta", r.finish("beta", asked).Provider)

		plain := r.ask(context.Background(), r.params())
		require.NoError(t, plain.err)
		r.say("gamma", "It is a cat")
		require.Equal(t, "gamma", r.finish("gamma", plain).Provider, "a reply offering no tools stays on its own model")
	})
}

func TestAReplyOfferingToolsStaysOnAModelThatIgnoresThemWhenNothingElseAnswers(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		r := newRig(t, 0, "gamma", "beta")
		r.fail("beta", errors.New("beta is down"))

		asked := r.ask(context.Background(), r.withTools())
		require.NoError(t, asked.err)
		r.say("gamma", "It is a cat")
		require.Equal(t, "gamma", r.finish("gamma", asked).Provider, "an answer without the tool beats none")
	})
}
