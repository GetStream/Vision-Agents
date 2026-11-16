package imagerouter

import (
	"context"
	"errors"
	"log/slog"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// stubImages stands in for a real provider so routing can be driven without credentials.
type stubImages struct {
	name   string
	drawn  imagegen.Result
	err    error
	asked  []imagegen.Request
	closed bool
}

func (s *stubImages) Generate(_ context.Context, request imagegen.Request) (imagegen.Result, error) {
	s.asked = append(s.asked, request)
	if s.err != nil {
		return imagegen.Result{}, s.err
	}
	return s.drawn, nil
}

func (s *stubImages) Start(context.Context) error { return nil }
func (s *stubImages) Close() error                { s.closed = true; return nil }
func (s *stubImages) Provider() string            { return s.name }
func (s *stubImages) Model() string               { return "stub" }

// pictures is a result of n pictures of one size.
func pictures(n, width, height int) imagegen.Result {
	var result imagegen.Result
	for range n {
		result.Images = append(result.Images, imagegen.Image{MediaType: "image/png", Width: width, Height: height})
	}
	return result
}

type ImageRouterSuite struct {
	suite.Suite
	ctx   context.Context
	quick *stubImages
	spare *stubImages
	lush  *stubImages
}

func TestImageRouterSuite(t *testing.T) {
	suite.Run(t, new(ImageRouterSuite))
}

func (s *ImageRouterSuite) SetupTest() {
	s.ctx = context.Background()
	s.quick = &stubImages{name: "quick", drawn: pictures(1, 1024, 1024)}
	s.spare = &stubImages{name: "spare", drawn: pictures(1, 1024, 1024)}
	s.lush = &stubImages{name: "lush", drawn: pictures(1, 1024, 1024)}
}

// config is two fast models, one billed by the picture and one by the pixel, and a slow
// one, which is what failover and both prices need.
func (s *ImageRouterSuite) config() routing.ModalityConfig {
	return routing.ModalityConfig{
		Providers: []routing.ProviderConfig{
			{
				Provider: "quick", Model: "fast", Languages: []string{"en"}, Tier: routing.LowLatency,
				Terms: []options.Term{options.Size, options.Seed},
				Price: routing.Price{PerImage: 0.04},
			},
			{
				Provider: "spare", Model: "fast", Languages: []string{"en"}, Tier: routing.LowLatency,
				Price: routing.Price{PerMegapixel: 0.02},
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
	}
}

func (s *ImageRouterSuite) newRouter(factories map[string]routing.Factory[imagegen.Provider]) *Router {
	if factories == nil {
		factories = map[string]routing.Factory[imagegen.Provider]{
			"quick": func(routing.Spec) (imagegen.Provider, error) { return s.quick, nil },
			"spare": func(routing.Spec) (imagegen.Provider, error) { return s.spare, nil },
			"lush":  func(routing.Spec) (imagegen.Provider, error) { return s.lush, nil },
		}
	}
	registry := NewRegistry()
	for name, factory := range factories {
		registry.Register(name, factory)
	}
	router, err := New(Options{Config: s.config(), Registry: registry, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

func (s *ImageRouterSuite) TestARequestNamingNowhereGoesToTheFastTier() {
	router := s.newRouter(nil)

	generation, err := router.Generate(s.ctx, Request{CustomerID: "acme", Image: imagegen.Request{Prompt: "a field"}})
	s.Require().NoError(err)

	s.Equal("quick", generation.Provider)
	s.Equal("fast", generation.Model)
	s.Len(generation.Images, 1)
	s.Require().Len(s.quick.asked, 1)
	s.Equal("a field", s.quick.asked[0].Prompt)
	s.True(s.quick.closed, "a provider is closed once it has drawn")
}

func (s *ImageRouterSuite) TestAModelBilledByThePictureCostsTheSameWhateverTheSize() {
	s.quick.drawn = pictures(3, 2048, 2048)
	router := s.newRouter(nil)

	generation, err := router.Generate(s.ctx, Request{
		CustomerID: "acme", Target: "quick/fast", Image: imagegen.Request{Prompt: "a field", N: 3},
	})
	s.Require().NoError(err)

	s.EqualValues(120_000, generation.CostMicros, "three pictures at four cents")
}

func (s *ImageRouterSuite) TestAModelBilledByThePixelCostsWhatCameBack() {
	s.spare.drawn = imagegen.Result{Images: []imagegen.Image{
		{MediaType: "image/png", Width: 1000, Height: 1000},
		{MediaType: "image/png", Width: 2000, Height: 1000},
	}}
	router := s.newRouter(nil)

	generation, err := router.Generate(s.ctx, Request{
		CustomerID: "acme", Target: "spare/fast", Image: imagegen.Request{Prompt: "a field", N: 2},
	})
	s.Require().NoError(err)

	s.EqualValues(60_000, generation.CostMicros, "three megapixels at two cents")
}

func (s *ImageRouterSuite) TestAProviderRefusingAtTheDoorHandsTheRequestOn() {
	s.quick.err = imagegen.Fail(imagegen.ProviderFailed, false, errors.New("fal: the queue answered 503"))
	router := s.newRouter(nil)

	generation, err := router.Generate(s.ctx, Request{CustomerID: "acme", Target: "image-fast", Image: imagegen.Request{Prompt: "a field"}})
	s.Require().NoError(err)

	s.Equal("spare", generation.Provider)
	s.Len(s.quick.asked, 1)
	s.Len(s.spare.asked, 1)
	s.Empty(s.lush.asked, "failover stays inside the tier the caller asked for")
}

func (s *ImageRouterSuite) TestAProviderWithoutAKeyDropsToTheNextCandidate() {
	router := s.newRouter(map[string]routing.Factory[imagegen.Provider]{
		"quick": func(routing.Spec) (imagegen.Provider, error) { return nil, errors.New("fal: FAL_KEY is required") },
		"spare": func(routing.Spec) (imagegen.Provider, error) { return s.spare, nil },
	})

	generation, err := router.Generate(s.ctx, Request{CustomerID: "acme", Image: imagegen.Request{Prompt: "a field"}})
	s.Require().NoError(err)

	s.Equal("spare", generation.Provider)
}

func (s *ImageRouterSuite) TestAJobAProviderAcceptedIsNeverAskedElsewhere() {
	s.quick.err = imagegen.Fail(imagegen.ProviderFailed, true, errors.New("fal: the job is \"FAILED\""))
	router := s.newRouter(nil)

	generation, err := router.Generate(s.ctx, Request{CustomerID: "acme", Image: imagegen.Request{Prompt: "a field"}})

	s.Equal(imagegen.ProviderFailed, imagegen.CodeOf(err))
	s.Equal("quick", generation.Provider, "the failure is reported as the provider's that failed")
	s.Zero(generation.CostMicros, "a failed generation costs nothing")
	s.Empty(s.spare.asked, "a job that may have been billed is not paid for twice")
}

func (s *ImageRouterSuite) TestAPictureASafetyFilterRefusedIsNeverAskedElsewhere() {
	s.quick.err = imagegen.Fail(imagegen.ContentFiltered, false, errors.New("fal: content_policy_violation"))
	router := s.newRouter(nil)

	generation, err := router.Generate(s.ctx, Request{CustomerID: "acme", Image: imagegen.Request{Prompt: "a field"}})

	s.Equal(imagegen.ContentFiltered, imagegen.CodeOf(err))
	s.Equal("quick", generation.Provider)
	s.Empty(s.spare.asked, "asking the next vendor would be shopping for a laxer filter")
}

func (s *ImageRouterSuite) TestAConcreteModelIsNotFailedOverFrom() {
	s.quick.err = imagegen.Fail(imagegen.ProviderFailed, false, errors.New("fal: the queue answered 503"))
	router := s.newRouter(nil)

	_, err := router.Generate(s.ctx, Request{CustomerID: "acme", Target: "quick/fast", Image: imagegen.Request{Prompt: "a field"}})

	s.Equal(imagegen.ProviderFailed, imagegen.CodeOf(err))
	s.ErrorContains(err, "503")
	s.Empty(s.spare.asked, "a caller that named one model asked for that model")
}

func (s *ImageRouterSuite) TestAPriorityListIsWalkedInTheOrderWritten() {
	s.spare.err = imagegen.Fail(imagegen.ProviderFailed, false, errors.New("google: Gemini answered 429"))
	router := s.newRouter(nil)

	generation, err := router.Generate(s.ctx, Request{
		CustomerID: "acme", Providers: []string{"spare", "lush"}, Image: imagegen.Request{Prompt: "a field"},
	})
	s.Require().NoError(err)

	s.Equal("lush", generation.Provider)
	s.Len(s.spare.asked, 1)
	s.Empty(s.quick.asked, "a model the list did not name is not asked")
}

func (s *ImageRouterSuite) TestATermOnlySomeModelsHonourNarrowsTheCandidates() {
	s.quick.err = imagegen.Fail(imagegen.ProviderFailed, false, errors.New("fal: the queue answered 503"))
	seed := int64(7)
	router := s.newRouter(nil)

	_, err := router.Generate(s.ctx, Request{CustomerID: "acme", Image: imagegen.Request{Prompt: "a field", Seed: &seed}})

	s.Equal(imagegen.ProviderFailed, imagegen.CodeOf(err))
	s.Require().Len(s.quick.asked, 1)
	s.Equal(&seed, s.quick.asked[0].Seed)
	s.Empty(s.spare.asked, "a model that cannot read a seed would draw a picture that ignored it")
}

func (s *ImageRouterSuite) TestATermNothingInTheTargetHonoursIsAnUnsupportedOption() {
	seed := int64(7)
	router := s.newRouter(nil)

	_, err := router.Generate(s.ctx, Request{
		CustomerID: "acme", Target: "image-quality", Image: imagegen.Request{Prompt: "a field", Seed: &seed},
	})

	s.Equal(imagegen.UnsupportedOption, imagegen.CodeOf(err))
	s.ErrorContains(err, "seed")
	s.Empty(s.lush.asked)
	s.Empty(s.quick.asked, "a model outside the target is not asked just because it could")
}

func (s *ImageRouterSuite) TestEveryCandidateRefusingTheSizeIsAnUnsupportedOption() {
	s.quick.err = imagegen.Fail(imagegen.UnsupportedOption, false, errors.New("fal: draws from 512 to 2048 pixels a side"))
	router := s.newRouter(nil)

	_, err := router.Generate(s.ctx, Request{
		CustomerID: "acme", Image: imagegen.Request{Prompt: "a field", Width: 4096, Height: 4096},
	})

	s.Equal(imagegen.UnsupportedOption, imagegen.CodeOf(err))
	s.False(imagegen.Accepted(err))
}

func (s *ImageRouterSuite) TestATargetThatNamesNowhereIsNotAGenerationFailure() {
	router := s.newRouter(nil)

	_, err := router.Generate(s.ctx, Request{CustomerID: "acme", Target: "nowhere", Image: imagegen.Request{Prompt: "a field"}})

	var failure *imagegen.Error
	s.Require().Error(err)
	s.False(errors.As(err, &failure), "the request is wrong, rather than the generation")
	s.ErrorContains(err, "nowhere")
}

func (s *ImageRouterSuite) TestADefaultConfigSectionIsServed() {
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	section, ok := config[routing.Image]
	s.Require().True(ok, "the built-in config generates images")

	router, err := New(Options{Config: section, Registry: DefaultRegistry(), Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)

	fast, err := router.Resolve(s.ctx, "image-fast", nil)
	s.Require().NoError(err)
	s.Equal("fal", fast[0].Config.Provider)
	quality, err := router.Resolve(s.ctx, "image-quality", nil)
	s.Require().NoError(err)
	s.Equal("google", quality[0].Config.Provider)
	s.Equal([]string{"image-fast", "image-quality"}, section.Offered())
}
