package google

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/jpeg"
	"image/png"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
)

const model = "gemini-3.1-flash-image"

// gemini stands in for the Gemini API's generateContent, answering each call with the
// next of its answers.
type gemini struct {
	server *httptest.Server

	mu       sync.Mutex
	answers  []answer
	requests []map[string]any
	keys     []string
}

type answer struct {
	status int
	body   any
}

func newGemini() *gemini {
	g := &gemini{}
	g.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		g.mu.Lock()
		defer g.mu.Unlock()
		if r.Method != http.MethodPost || r.URL.Path != "/models/"+model+":generateContent" {
			http.NotFound(w, r)
			return
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		g.requests = append(g.requests, body)
		g.keys = append(g.keys, r.Header.Get("x-goog-api-key"))

		next := answer{status: http.StatusInternalServerError, body: map[string]any{"error": "no answer left"}}
		if len(g.answers) > 0 {
			next, g.answers = g.answers[0], g.answers[1:]
		}
		w.WriteHeader(next.status)
		_ = json.NewEncoder(w).Encode(next.body)
	}))
	return g
}

func (g *gemini) answer(status int, body any) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.answers = append(g.answers, answer{status: status, body: body})
}

func (g *gemini) asked() []map[string]any {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.requests
}

// drawn is a generateContent answer carrying one inline picture.
func drawn(mimeType string, data []byte) map[string]any {
	return map[string]any{"candidates": []any{map[string]any{
		"content": map[string]any{"parts": []any{
			map[string]any{"text": "Here is your picture."},
			map[string]any{"inlineData": map[string]any{
				"mimeType": mimeType, "data": base64.StdEncoding.EncodeToString(data),
			}},
		}},
		"finishReason": "STOP",
	}}}
}

func picture(format string, width, height int) []byte {
	var encoded bytes.Buffer
	canvas := image.NewRGBA(image.Rect(0, 0, width, height))
	if format == imagegen.FormatJPEG {
		_ = jpeg.Encode(&encoded, canvas, nil)
	} else {
		_ = png.Encode(&encoded, canvas)
	}
	return encoded.Bytes()
}

type GoogleSuite struct {
	suite.Suite
	ctx      context.Context
	gemini   *gemini
	provider *Provider
}

func TestGoogleSuite(t *testing.T) {
	suite.Run(t, new(GoogleSuite))
}

func (s *GoogleSuite) SetupTest() {
	s.ctx = context.Background()
	s.gemini = newGemini()
	s.T().Cleanup(s.gemini.server.Close)

	provider, err := New(Options{APIKey: "test-key", Model: model, BaseURL: s.gemini.server.URL})
	s.Require().NoError(err)
	s.provider = provider
}

func (s *GoogleSuite) TestAKeyIsRequired() {
	s.T().Setenv(apiKeyEnvVar, "")

	_, err := New(Options{Model: model})
	s.ErrorContains(err, apiKeyEnvVar)
}

func (s *GoogleSuite) TestAPictureIsAskedForInTheShapeWantedAndReturned() {
	s.gemini.answer(http.StatusOK, drawn("image/png", picture("png", 1344, 768)))

	result, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "A yellow watering can", AspectRatio: "16:9"})
	s.Require().NoError(err)

	s.Require().Len(result.Images, 1)
	s.Equal("image/png", result.Images[0].MediaType)
	s.Equal(1344, result.Images[0].Width)
	s.Equal(768, result.Images[0].Height)

	asked := s.gemini.asked()
	s.Require().Len(asked, 1)
	config := asked[0]["generationConfig"].(map[string]any)
	s.Equal([]any{"IMAGE"}, config["responseModalities"])
	s.Equal(map[string]any{"aspectRatio": "16:9", "imageSize": "1K"}, config["imageConfig"])
	s.Equal([]any{map[string]any{"parts": []any{map[string]any{"text": "A yellow watering can"}}}}, asked[0]["contents"])
	s.Equal([]string{"test-key"}, s.gemini.keys)
}

func (s *GoogleSuite) TestSeveralPicturesAreOneCallEach() {
	s.gemini.answer(http.StatusOK, drawn("image/png", picture("png", 64, 64)))
	s.gemini.answer(http.StatusOK, drawn("image/jpeg", picture("jpeg", 64, 64)))

	result, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field", N: 2})
	s.Require().NoError(err)

	s.Require().Len(result.Images, 2)
	s.Equal("image/png", result.Images[0].MediaType)
	s.Equal("image/jpeg", result.Images[1].MediaType)
	s.Len(s.gemini.asked(), 2)
}

func (s *GoogleSuite) TestASketchTheModelThoughtWithIsNotTheAnswer() {
	sketch := base64.StdEncoding.EncodeToString(picture("png", 8, 8))
	final := base64.StdEncoding.EncodeToString(picture("png", 64, 64))
	s.gemini.answer(http.StatusOK, map[string]any{"candidates": []any{map[string]any{
		"content": map[string]any{"parts": []any{
			map[string]any{"thought": true, "inlineData": map[string]any{"mimeType": "image/png", "data": sketch}},
			map[string]any{"inlineData": map[string]any{"mimeType": "image/png", "data": final}},
		}},
	}}})

	result, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field"})
	s.Require().NoError(err)

	s.Equal(64, result.Images[0].Width)
}

func (s *GoogleSuite) TestABlockedPromptIsContentFiltered() {
	s.gemini.answer(http.StatusOK, map[string]any{"promptFeedback": map[string]any{"blockReason": "PROHIBITED_CONTENT"}})

	_, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field"})

	s.Equal(imagegen.ContentFiltered, imagegen.CodeOf(err))
	s.ErrorContains(err, "PROHIBITED_CONTENT")
}

func (s *GoogleSuite) TestAPictureStoppedOnASafetyRuleIsContentFiltered() {
	s.gemini.answer(http.StatusOK, map[string]any{"candidates": []any{map[string]any{
		"content":      map[string]any{"parts": []any{}},
		"finishReason": "IMAGE_SAFETY",
	}}})

	_, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field"})

	s.Equal(imagegen.ContentFiltered, imagegen.CodeOf(err))
	s.True(imagegen.Accepted(err))
}

func (s *GoogleSuite) TestAnAnswerWithNoPictureIsAFailure() {
	s.gemini.answer(http.StatusOK, map[string]any{"candidates": []any{map[string]any{
		"content":      map[string]any{"parts": []any{map[string]any{"text": "I would rather describe it."}}},
		"finishReason": "STOP",
	}}})

	_, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field"})

	s.Equal(imagegen.ProviderFailed, imagegen.CodeOf(err))
	s.True(imagegen.Accepted(err), "Gemini answered, so it billed")
	s.ErrorContains(err, "drew nothing")
}

func (s *GoogleSuite) TestAPictureThatDoesNotDecodeIsRefused() {
	s.gemini.answer(http.StatusOK, drawn("image/png", picture("png", 64, 64)[:40]))

	_, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field"})

	s.Equal(imagegen.ProviderFailed, imagegen.CodeOf(err))
	s.ErrorContains(err, "does not decode")
}

func (s *GoogleSuite) TestAPictureLabelledAsSomethingElseIsRefused() {
	s.gemini.answer(http.StatusOK, drawn("image/png", picture("jpeg", 64, 64)))

	_, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field"})

	s.Equal(imagegen.ProviderFailed, imagegen.CodeOf(err))
	s.ErrorContains(err, "labelled image/png is image/jpeg")
}

func (s *GoogleSuite) TestARefusalBeforeAnythingWasDrawnWasNeverAccepted() {
	s.gemini.answer(http.StatusTooManyRequests, map[string]any{"error": map[string]any{"status": "RESOURCE_EXHAUSTED"}})

	_, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field"})

	s.Equal(imagegen.ProviderFailed, imagegen.CodeOf(err))
	s.False(imagegen.Accepted(err), "nothing was drawn, so the request can go elsewhere")
	s.ErrorContains(err, "429")
}

func (s *GoogleSuite) TestARefusalAfterAPictureWasDrawnWasAccepted() {
	s.gemini.answer(http.StatusOK, drawn("image/png", picture("png", 64, 64)))
	s.gemini.answer(http.StatusServiceUnavailable, map[string]any{"error": "busy"})

	_, err := s.provider.Generate(s.ctx, imagegen.Request{Prompt: "a field", N: 2})

	s.Equal(imagegen.ProviderFailed, imagegen.CodeOf(err))
	s.True(imagegen.Accepted(err), "the first picture was billed, so the request is not asked again elsewhere")
}

func (s *GoogleSuite) TestWhatGeminiIsNotAskedForIsRefusedWithoutAsking() {
	seed := int64(7)
	for _, request := range []imagegen.Request{
		{Prompt: "a field", AspectRatio: "7:3"},
		{Prompt: "a field", Width: 1024, Height: 1024},
		{Prompt: "a field", Seed: &seed},
		{Prompt: "a field", NegativePrompt: "text"},
		{Prompt: "a field", Format: imagegen.FormatPNG},
	} {
		_, err := s.provider.Generate(s.ctx, request)

		s.Equal(imagegen.UnsupportedOption, imagegen.CodeOf(err), "%+v", request)
		s.False(imagegen.Accepted(err))
	}
	s.Empty(s.gemini.asked())
}
