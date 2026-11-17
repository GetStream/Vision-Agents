package sttrouter

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

func TestTheSessionRetunesWithoutHandingOutTheVendor(t *testing.T) {
	provider := &retuningSTT{quietSTT: quietSTT{emitter: stt.NewEmitter(64)}}
	session := newSession(provider, routing.ProviderConfig{Provider: "deepgram", Model: "flux-general-en"}, routing.Owner{}, nil, nil, nil)
	defer session.Close()

	require.NoError(t, session.SetKeyterms([]string{"Chen"}))

	require.Equal(t, []string{"Chen"}, provider.terms)
}

func TestASessionWhoseVendorCannotRetuneIgnoresTheTerms(t *testing.T) {
	session := newSession(&quietSTT{emitter: stt.NewEmitter(64)}, routing.ProviderConfig{Provider: "deepgram", Model: "flux-general-en"}, routing.Owner{}, nil, nil, nil)
	defer session.Close()

	require.NoError(t, session.SetKeyterms([]string{"Chen"}))
}

// retuningSTT is a transcriber that can be told mid-session which words to expect.
type retuningSTT struct {
	quietSTT
	terms []string
}

func (r *retuningSTT) SetKeyterms(terms []string) error {
	r.terms = terms
	return nil
}
