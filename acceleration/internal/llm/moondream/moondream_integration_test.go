//go:build integration

package moondream

import (
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// Moondream takes no tools, so it is not run through llmsuite: what it is for is reading a
// picture, and that is what is asked of it.
func TestMoondreamSaysWhatIsInAPicture(t *testing.T) {
	if os.Getenv("MOONDREAM_API_KEY") == "" {
		t.Skip("MOONDREAM_API_KEY not set")
	}
	cat, err := os.ReadFile("../../detection/detectionsuite/testdata/cat.jpg")
	require.NoError(t, err)
	provider, err := New(Options{})
	require.NoError(t, err)
	defer provider.Close()

	stream, err := provider.Create(t.Context(), llm.ResponseParams{Input: []llm.Message{{
		Role: llm.User,
		Parts: []llm.ContentPart{
			{Text: "What animal is this? Answer in one word."},
			{Image: &llm.ImagePart{MIME: "image/jpeg", Data: cat}},
		},
	}}})
	require.NoError(t, err)
	response, err := llm.Collect(stream)
	require.NoError(t, err)

	require.Contains(t, strings.ToLower(response.OutputText), "cat")
	require.Positive(t, response.Usage.InputTokens)
	require.Positive(t, response.Usage.OutputTokens)
}
