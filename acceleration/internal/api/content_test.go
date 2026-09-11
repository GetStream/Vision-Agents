package api

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

func TestParseContentReadsAString(t *testing.T) {
	text, parts, err := parseContent(json.RawMessage(`"hello"`))

	require.NoError(t, err)
	require.Equal(t, "hello", text)
	require.Empty(t, parts)
}

func TestParseContentReadsImageParts(t *testing.T) {
	image := llm.ImagePart{MIME: "image/jpeg", Data: []byte{1, 2, 3}}
	raw, err := json.Marshal([]wirePart{
		{Type: "text", Text: "what is this"},
		{Type: "image_url", ImageURL: &wireImage{URL: image.DataURI(), Detail: "low"}},
	})
	require.NoError(t, err)

	text, parts, err := parseContent(raw)

	require.NoError(t, err)
	require.Empty(t, text)
	require.Len(t, parts, 2)
	require.Equal(t, "what is this", parts[0].Text)
	require.Equal(t, []byte{1, 2, 3}, parts[1].Image.Data)
	require.Equal(t, "low", parts[1].Image.Detail)
}

func TestParseContentAcceptsFiveImages(t *testing.T) {
	image := llm.ImagePart{MIME: "image/jpeg", Data: []byte{1}}
	wires := make([]wirePart, 5)
	for i := range wires {
		wires[i] = wirePart{Type: "image_url", ImageURL: &wireImage{URL: image.DataURI()}}
	}
	raw, err := json.Marshal(wires)
	require.NoError(t, err)

	_, _, err = parseContent(raw)

	require.NoError(t, err)
}

func TestParseContentAcceptsAURL(t *testing.T) {
	raw, err := json.Marshal([]wirePart{
		{Type: "image_url", ImageURL: &wireImage{URL: "https://example.com/cat.jpg"}},
	})
	require.NoError(t, err)

	_, _, err = parseContent(raw)

	require.NoError(t, err)
}
