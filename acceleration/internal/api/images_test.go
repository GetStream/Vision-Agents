//go:build integration

package api

import (
	"net/http"
	"strings"
	"testing"
)

// ImagesSuite covers drawing a picture: what comes back, what it cost, and what a request
// nothing can draw is answered with.
type ImagesSuite struct {
	RouterSuite
}

func TestImagesSuite(t *testing.T) {
	runSuite(t, new(ImagesSuite))
}

func (s *ImagesSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *ImagesSuite) TestAPromptComesBackAsPicturesAndWhatTheyCost() {
	prompt := "a yellow watering can " + s.utils.uuid()

	generation := s.generate(ImageGenerationRequest{
		Prompt: prompt,
		Options: &ImageOptions{
			Target: pointerTo("image-fast"), Size: pointerTo("1024x1024"),
			AspectRatio: pointerTo("1:1"), N: pointerTo(1), Seed: pointerTo(int64(7)),
			NegativePrompt: pointerTo("text"), OutputFormat: pointerTo(ImageOptionsOutputFormat("png")),
		},
		Tags: &map[string]string{"employee": "e1"},
	})

	s.True(strings.HasPrefix(generation.Id, "img_"))
	s.Equal(ImageGenerationStatusCompleted, generation.Status)
	s.Equal("quick", *generation.Provider)
	s.Equal("fast", *generation.Model)
	s.EqualValues(40_000, generation.CostMicros)
	s.Nil(generation.ErrorCode)
	s.Nil(generation.Error)
	s.Require().Len(generation.Images, 1)
	s.Equal(drawn.Images[0].Data, generation.Images[0].Data, "the picture itself, not a link to it")
	s.Equal(GeneratedImageMediaType("image/png"), generation.Images[0].MediaType)
	s.Equal(1024, generation.Images[0].Width)
	s.EqualValues(7, *generation.Images[0].Seed)
}

func (s *ImagesSuite) TestWhatWasAskedForReachesTheProviderAsItWasWritten() {
	prompt := "a yellow watering can " + s.utils.uuid()

	s.generate(ImageGenerationRequest{Prompt: prompt, Options: &ImageOptions{
		Target: pointerTo("image-fast"), Size: pointerTo("1024x1024"),
		AspectRatio: pointerTo("1:1"), N: pointerTo(1), Seed: pointerTo(int64(7)),
		NegativePrompt: pointerTo("text"), OutputFormat: pointerTo(ImageOptionsOutputFormat("png")),
	}})

	asked, drew := painters.quick.drew(prompt)
	s.Require().True(drew)
	s.Equal(1024, asked.Width)
	s.Equal(1024, asked.Height)
	s.Equal("1:1", asked.AspectRatio)
	s.Equal(1, asked.N)
	s.EqualValues(7, *asked.Seed)
	s.Equal("text", asked.NegativePrompt)
	s.Equal("png", asked.Format)
}

func (s *ImagesSuite) TestAPictureASafetyFilterRefusedIsAFailedGenerationRatherThanABadRequest() {
	generation := s.generate(ImageGenerationRequest{Prompt: unsafely + " " + s.utils.uuid()})

	s.Equal(ImageGenerationStatusFailed, generation.Status)
	s.Equal(ImageErrorCodeContentFiltered, *generation.ErrorCode)
	s.Contains(*generation.Error, "safety checker")
	s.Equal("quick", *generation.Provider, "the response says who refused")
	s.Zero(generation.CostMicros)
}

func (s *ImagesSuite) TestAnOptionNothingInTheTargetHonoursIsAnUnsupportedOption() {
	prompt := "a field " + s.utils.uuid()

	generation := s.generate(ImageGenerationRequest{Prompt: prompt, Options: &ImageOptions{
		Target: pointerTo("image-quality"), Seed: pointerTo(int64(7)),
	}})

	s.Equal(ImageErrorCodeUnsupportedOption, *generation.ErrorCode)
	s.Nil(generation.Provider, "nothing got as far as a provider")
	_, drew := painters.lush.drew(prompt)
	s.False(drew)
}

func (s *ImagesSuite) TestAPromptOfNothingIsRefused() {
	s.Equal(http.StatusBadRequest, s.refused(ImageGenerationRequest{Prompt: "  "}))
}

func (s *ImagesSuite) TestMorePicturesThanAreDrawnAtOnceIsRefused() {
	s.Equal(http.StatusBadRequest, s.refused(ImageGenerationRequest{
		Prompt: "a field", Options: &ImageOptions{N: pointerTo(5)}}))
}

func (s *ImagesSuite) TestAskingForNoPictureAtAllIsRefused() {
	s.Equal(http.StatusBadRequest, s.refused(ImageGenerationRequest{
		Prompt: "a field", Options: &ImageOptions{N: pointerTo(0)}}))
}

func (s *ImagesSuite) TestASizeThatIsNotTwoNumbersIsRefused() {
	s.Equal(http.StatusBadRequest, s.refused(ImageGenerationRequest{
		Prompt: "a field", Options: &ImageOptions{Size: pointerTo("big")}}))
}

func (s *ImagesSuite) TestASizeOfOneNumberIsRefused() {
	s.Equal(http.StatusBadRequest, s.refused(ImageGenerationRequest{
		Prompt: "a field", Options: &ImageOptions{Size: pointerTo("1024")}}))
}

func (s *ImagesSuite) TestASizeWithANumberMissingIsRefused() {
	s.Equal(http.StatusBadRequest, s.refused(ImageGenerationRequest{
		Prompt: "a field", Options: &ImageOptions{Size: pointerTo("0x1024")}}))
}

func (s *ImagesSuite) TestAnAspectRatioNothingRecognisesIsRefused() {
	s.Equal(http.StatusBadRequest, s.refused(ImageGenerationRequest{
		Prompt: "a field", Options: &ImageOptions{AspectRatio: pointerTo("wide")}}))
}

func (s *ImagesSuite) TestAFormatNobodyWritesIsRefused() {
	s.Equal(http.StatusBadRequest, s.refused(ImageGenerationRequest{
		Prompt: "a field", Options: &ImageOptions{OutputFormat: pointerTo(ImageOptionsOutputFormat("webp"))}}))
}

func (s *ImagesSuite) TestASeedThatIsNotANumberToStartFromIsRefused() {
	s.Equal(http.StatusBadRequest, s.refused(ImageGenerationRequest{
		Prompt: "a field", Options: &ImageOptions{Seed: pointerTo(int64(-1))}}))
}

func (s *ImagesSuite) TestATargetNothingAnswersToIsRefused() {
	s.Equal(http.StatusBadRequest, s.refused(ImageGenerationRequest{
		Prompt: "a field", Options: &ImageOptions{Target: pointerTo("nowhere")}}))
}

func (s *ImagesSuite) TestACostLabelThatIsNotOneIsRefused() {
	s.Equal(http.StatusBadRequest, s.refused(ImageGenerationRequest{
		Prompt: "a field", Tags: &map[string]string{"not a key": "x"}}))
}

func (s *ImagesSuite) TestOnlyTheCustomersOwnBackendMayDraw() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		status, _ := as.call(http.MethodPost, "/v1/image/generations",
			ImageGenerationRequest{Prompt: "a field " + s.utils.uuid()})
		return status
	})
}

func (s *ImagesSuite) TestImageModelsAreListedWithTheOtherModalities() {
	var routes []Route
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/image/routes", nil, &routes))

	s.Require().Len(routes, 2)
	s.Equal("image-fast", routes[0].Id)
	s.Equal("quick", routes[0].Candidates[0].Provider)
	s.Equal("image-quality", routes[1].Id)
}

// generate draws a picture the endpoint must answer for, whether or not it could be drawn.
func (s *ImagesSuite) generate(request ImageGenerationRequest) ImageGeneration {
	var generation ImageGeneration
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodPost, "/v1/image/generations", request, &generation))
	return generation
}

// refused is the status of a request the endpoint would not take at all.
func (s *ImagesSuite) refused(request ImageGenerationRequest) int {
	status, _ := s.serverClient.call(http.MethodPost, "/v1/image/generations", request)
	return status
}
