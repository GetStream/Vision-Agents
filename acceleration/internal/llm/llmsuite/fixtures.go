//go:build integration

package llmsuite

import (
	"bytes"
	"image"
	"image/color"
	"image/draw"
	"image/png"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// The fixtures are what the tests ask. Each has one answer no model gets wrong, so a test
// that fails is the provider failing rather than the model having an opinion. Tests read
// them and never change them.

// capital is a question with a one-word answer.
var capital = llm.ResponseParams{
	Instructions: "Answer with a single word and no punctuation.",
	Input:        []llm.Message{{Role: llm.User, Content: "What is the capital of France?"}},
}

// favouriteNumber is only answerable from the turns before the question.
var favouriteNumber = llm.ResponseParams{
	Instructions: "Answer with a single number and nothing else.",
	Input: []llm.Message{
		{Role: llm.User, Content: "My favourite number is 7. Remember it."},
		{Role: llm.Assistant, Content: "Noted."},
		{Role: llm.User, Content: "What is my favourite number?"},
	},
}

// essay cannot be finished inside the budget it is given.
var essay = llm.ResponseParams{
	Input:           []llm.Message{{Role: llm.User, Content: "Write a long essay about the sea."}},
	MaxOutputTokens: 16,
}

// counting runs long enough to be cut off partway.
var counting = llm.ResponseParams{
	Input:           []llm.Message{{Role: llm.User, Content: "Count slowly from 1 to 200."}},
	MaxOutputTokens: 2048,
}

// weather is a question only the weather tool can answer.
var weather = llm.Message{Role: llm.User, Content: "What is the weather in Paris?"}

// weatherTool is the tool the weather question is asked alongside.
var weatherTool = llm.Tool{
	Name:        "get_weather",
	Description: "Look up the weather somewhere",
	Parameters: map[string]any{
		"type":       "object",
		"properties": map[string]any{"city": map[string]any{"type": "string"}},
		"required":   []string{"city"},
	},
}

// weatherReport is what the weather tool found, with a number the answer has to repeat.
const weatherReport = "It is 20 degrees and sunny in Paris."

// squares are a red and then a blue square. SetupSuite encodes them once for every test
// that shows them.
func squares() ([]llm.ContentPart, error) {
	var parts []llm.ContentPart
	for _, c := range []color.Color{color.RGBA{R: 255, A: 255}, color.RGBA{B: 255, A: 255}} {
		picture := image.NewRGBA(image.Rect(0, 0, 64, 64))
		draw.Draw(picture, picture.Bounds(), image.NewUniform(c), image.Point{}, draw.Src)
		var data bytes.Buffer
		if err := png.Encode(&data, picture); err != nil {
			return nil, err
		}
		parts = append(parts, llm.ContentPart{Image: &llm.ImagePart{MIME: "image/png", Data: data.Bytes()}})
	}
	return parts, nil
}
