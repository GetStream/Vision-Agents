package agent

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

func TestAnUtteranceThatCameBackWithoutAudioIsToldApartFromOneCutShort(t *testing.T) {
	for _, test := range []struct {
		name   string
		done   tts.SynthesisComplete
		silent bool
	}{
		{"text was billed and no audio came back", tts.SynthesisComplete{Characters: 42}, true},
		{"text came back as speech", tts.SynthesisComplete{Characters: 42, AudioDurationMs: 1800}, false},
		{"the caller cut it short before any audio", tts.SynthesisComplete{Characters: 42, Interrupted: true}, false},
		{"there was no text to say", tts.SynthesisComplete{}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			require.Equal(t, test.silent, spokeNothing(test.done))
		})
	}
}
