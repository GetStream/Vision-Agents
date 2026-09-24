package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"image"
	"image/png"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagerouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// painter stands in for an image provider: it draws what it is told to, or waits for the
// caller to give up, or fails the way it is told to.
type painter struct {
	name    string
	drawn   imagegen.Result
	err     error
	waiting bool

	mu       sync.Mutex
	asked    []imagegen.Request
	deadline time.Time
}

func (p *painter) Generate(ctx context.Context, request imagegen.Request) (imagegen.Result, error) {
	p.mu.Lock()
	p.asked = append(p.asked, request)
	p.deadline, _ = ctx.Deadline()
	p.mu.Unlock()
	if p.waiting {
		<-ctx.Done()
		return imagegen.Result{}, imagegen.Fail(imagegen.CodeOf(ctx.Err()), true, ctx.Err())
	}
	if p.err != nil {
		return imagegen.Result{}, p.err
	}
	return p.drawn, nil
}

func (p *painter) Start(context.Context) error { return nil }
func (p *painter) Close() error                { return nil }
func (p *painter) Provider() string            { return p.name }
func (p *painter) Model() string               { return "stub" }

type ImagesSuite struct {
	suite.Suite
	png     []byte
	quick   *painter
	lush    *painter
	handler http.Handler
}

func TestImagesSuite(t *testing.T) {
	suite.Run(t, new(ImagesSuite))
}

func (s *ImagesSuite) SetupTest() {
	var encoded bytes.Buffer
	s.Require().NoError(png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 32, 32))))
	s.png = encoded.Bytes()
	seed := int64(7)
	s.quick = &painter{name: "quick", drawn: imagegen.Result{Images: []imagegen.Image{
		{Data: s.png, MediaType: "image/png", Width: 1024, Height: 1024, Seed: &seed},
	}}}
	s.lush = &painter{name: "lush", drawn: s.quick.drawn}

	registry := imagerouter.NewRegistry()
	registry.Register("quick", func(routing.Spec) (imagegen.Provider, error) { return s.quick, nil })
	registry.Register("lush", func(routing.Spec) (imagegen.Provider, error) { return s.lush, nil })
	router, err := imagerouter.New(imagerouter.Options{
		Config: routing.ModalityConfig{
			Providers: []routing.ProviderConfig{
				{
					Provider: "quick", Model: "fast", Languages: []string{"en"}, Tier: routing.LowLatency,
					Terms: []options.Term{options.Size, options.AspectRatio, options.Seed, options.NegativePrompt, options.Format},
					Price: routing.Price{PerImage: 0.04},
				},
				{
					Provider: "lush", Model: "best", Languages: []string{"en"}, Tier: routing.HighQuality,
					Terms: []options.Term{options.AspectRatio},
					Price: routing.Price{PerImage: 0.067},
				},
			},
			Aliases: map[string]routing.Alias{
				"image-fast":    {Title: "Fast images", Tier: routing.LowLatency},
				"image-quality": {Title: "Best images", Tier: routing.HighQuality},
			},
		},
		Registry: registry,
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)

	// Behind a proxy the caller says what it is, which is what lets a device be told apart
	// from the customer's own backend.
	authenticator, err := auth.New(auth.Proxy, nil)
	s.Require().NoError(err)
	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{routing.Image: router},
		Streams: &Streams{Image: router},
		Auth:    authenticator,
		Logger:  slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.handler = server.Handler()
}

func (s *ImagesSuite) generate(ctx context.Context, body string, headers ...string) *httptest.ResponseRecorder {
	request := httptest.NewRequestWithContext(ctx, http.MethodPost, "/v1/image/generations", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(CustomerHeader, "acme")
	for i := 0; i+1 < len(headers); i += 2 {
		request.Header.Set(headers[i], headers[i+1])
	}
	recorder := httptest.NewRecorder()
	s.handler.ServeHTTP(recorder, request)
	return recorder
}

func (s *ImagesSuite) generated(recorder *httptest.ResponseRecorder) ImageGeneration {
	s.Require().Equal(http.StatusOK, recorder.Code, recorder.Body.String())
	var generation ImageGeneration
	s.Require().NoError(json.Unmarshal(recorder.Body.Bytes(), &generation))
	return generation
}

func (s *ImagesSuite) TestAPromptComesBackAsPicturesAndWhatTheyCost() {
	recorder := s.generate(context.Background(), `{
		"prompt": "A yellow watering can",
		"options": {"target": "image-fast", "size": "1024x1024", "aspect_ratio": "1:1", "n": 1,
			"seed": 7, "negative_prompt": "text", "output_format": "png"},
		"tags": {"employee": "e1"}
	}`)

	generation := s.generated(recorder)
	s.True(strings.HasPrefix(generation.Id, "img_"))
	s.Equal(ImageGenerationStatusCompleted, generation.Status)
	s.Equal("quick", *generation.Provider)
	s.Equal("fast", *generation.Model)
	s.EqualValues(40_000, generation.CostMicros)
	s.Nil(generation.ErrorCode)
	s.Nil(generation.Error)
	s.Require().Len(generation.Images, 1)
	s.Equal(s.png, generation.Images[0].Data, "the picture itself, not a link to it")
	s.Equal(GeneratedImageMediaType("image/png"), generation.Images[0].MediaType)
	s.Equal(1024, generation.Images[0].Width)
	s.EqualValues(7, *generation.Images[0].Seed)
	s.NotContains(recorder.Body.String(), "error_code", "a picture that was drawn carries no reason it was not")

	s.Require().Len(s.quick.asked, 1)
	asked := s.quick.asked[0]
	s.Equal("A yellow watering can", asked.Prompt)
	s.Equal(1024, asked.Width)
	s.Equal(1024, asked.Height)
	s.Equal("1:1", asked.AspectRatio)
	s.Equal(1, asked.N)
	s.EqualValues(7, *asked.Seed)
	s.Equal("text", asked.NegativePrompt)
	s.Equal("png", asked.Format)
}

func (s *ImagesSuite) TestAGenerationRunsUnderItsOwnDeadline() {
	recorder := s.generate(context.Background(), `{"prompt": "a field"}`)

	s.generated(recorder)
	s.WithinDuration(time.Now().Add(imageDeadline), s.quick.deadline, 5*time.Second,
		"the provider is given 240 seconds, not however long the connection lasts")
}

func (s *ImagesSuite) TestAGenerationThatRanOutOfTimeSaysSo() {
	s.quick.waiting = true
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	generation := s.generated(s.generate(ctx, `{"prompt": "a field"}`))

	s.Equal(ImageGenerationStatusFailed, generation.Status)
	s.Equal(ImageErrorCodeTimeout, *generation.ErrorCode)
	s.Zero(generation.CostMicros)
	s.Empty(generation.Images)
}

func (s *ImagesSuite) TestACallerThatHangsUpCancelsTheGeneration() {
	s.quick.waiting = true
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(30*time.Millisecond, cancel)

	generation := s.generated(s.generate(ctx, `{"prompt": "a field"}`))

	s.Equal(ImageErrorCodeCancelled, *generation.ErrorCode)
}

func (s *ImagesSuite) TestAPictureASafetyFilterRefusedIsAFailedGenerationRatherThanABadRequest() {
	s.quick.err = imagegen.Fail(imagegen.ContentFiltered, true, errors.New("fal: the safety checker flagged the picture"))

	generation := s.generated(s.generate(context.Background(), `{"prompt": "a field"}`))

	s.Equal(ImageGenerationStatusFailed, generation.Status)
	s.Equal(ImageErrorCodeContentFiltered, *generation.ErrorCode)
	s.Contains(*generation.Error, "safety checker")
	s.Equal("quick", *generation.Provider, "the response says who refused")
	s.Zero(generation.CostMicros)
	s.Empty(s.lush.asked)
}

func (s *ImagesSuite) TestAnOptionNothingInTheTargetHonoursIsAnUnsupportedOption() {
	generation := s.generated(s.generate(context.Background(),
		`{"prompt": "a field", "options": {"target": "image-quality", "seed": 7}}`))

	s.Equal(ImageErrorCodeUnsupportedOption, *generation.ErrorCode)
	s.Nil(generation.Provider, "nothing got as far as a provider")
	s.Empty(s.lush.asked)
	s.Empty(s.quick.asked)
}

func (s *ImagesSuite) TestARequestThatCannotBeDrawnIsABadRequest() {
	for _, body := range []string{
		`{"prompt": "  "}`,
		`{"prompt": "a field", "options": {"n": 5}}`,
		`{"prompt": "a field", "options": {"n": 0}}`,
		`{"prompt": "a field", "options": {"size": "big"}}`,
		`{"prompt": "a field", "options": {"size": "1024"}}`,
		`{"prompt": "a field", "options": {"size": "0x1024"}}`,
		`{"prompt": "a field", "options": {"aspect_ratio": "wide"}}`,
		`{"prompt": "a field", "options": {"output_format": "webp"}}`,
		`{"prompt": "a field", "options": {"seed": -1}}`,
		`{"prompt": "a field", "options": {"target": "nowhere"}}`,
		`{"prompt": "a field", "tags": {"not a key": "x"}}`,
	} {
		recorder := s.generate(context.Background(), body)

		s.Equal(http.StatusBadRequest, recorder.Code, body)
	}
	s.Empty(s.quick.asked, "nothing is drawn for a request that was wrong")
}

func (s *ImagesSuite) TestADeviceMayNotDrawOnTheCustomersAccount() {
	recorder := s.generate(context.Background(), `{"prompt": "a field"}`, auth.AuthTypeHeader, auth.AuthTypeJWT)

	s.Equal(http.StatusForbidden, recorder.Code)
	s.Empty(s.quick.asked)
}

func (s *ImagesSuite) TestACallerWithNoCustomerIsRefused() {
	request := httptest.NewRequest(http.MethodPost, "/v1/image/generations", strings.NewReader(`{"prompt": "a field"}`))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	s.handler.ServeHTTP(recorder, request)

	s.Equal(http.StatusUnauthorized, recorder.Code)
}

func (s *ImagesSuite) TestADeploymentWithoutImagesSaysSo() {
	registry := imagerouter.NewRegistry()
	router, err := imagerouter.New(imagerouter.Options{
		Config: routing.ModalityConfig{Providers: []routing.ProviderConfig{
			{Provider: "quick", Model: "fast", Languages: []string{"en"}},
		}},
		Registry: registry,
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	server, err := NewServer(Options{
		Routers: map[routing.Modality]routing.Inspector{routing.STT: router},
		Logger:  slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	request := httptest.NewRequest(http.MethodPost, "/v1/image/generations", strings.NewReader(`{"prompt": "a field"}`))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(CustomerHeader, "acme")
	recorder := httptest.NewRecorder()
	server.Handler().ServeHTTP(recorder, request)

	s.Equal(http.StatusNotFound, recorder.Code)
	s.Contains(recorder.Body.String(), noImages)
}

func (s *ImagesSuite) TestImageModelsAreListedWithTheOtherModalities() {
	request := httptest.NewRequest(http.MethodGet, "/v1/image/routes", nil)
	request.Header.Set(CustomerHeader, "acme")
	recorder := httptest.NewRecorder()
	s.handler.ServeHTTP(recorder, request)

	s.Require().Equal(http.StatusOK, recorder.Code, recorder.Body.String())
	var routes []Route
	s.Require().NoError(json.Unmarshal(recorder.Body.Bytes(), &routes))
	s.Require().Len(routes, 2)
	s.Equal("image-fast", routes[0].Id)
	s.Equal("quick", routes[0].Candidates[0].Provider)
	s.Equal("image-quality", routes[1].Id)
}
