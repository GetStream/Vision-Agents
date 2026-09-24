package imagegen

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"image/png"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

type ImagegenSuite struct {
	suite.Suite
}

func TestImagegenSuite(t *testing.T) {
	suite.Run(t, new(ImagegenSuite))
}

func (s *ImagegenSuite) TestARequestNeedsSomethingToDraw() {
	s.ErrorContains(Request{Prompt: "  "}.Validate(), "nothing to draw")
	s.NoError(Request{Prompt: "a field"}.Validate())
}

func (s *ImagegenSuite) TestARequestIsRefusedWhatNoModelCouldDraw() {
	seed := int64(-1)
	for _, request := range []Request{
		{Prompt: "a field", N: MaxImages + 1},
		{Prompt: "a field", Width: 1024},
		{Prompt: "a field", Width: MaxSide + 1, Height: 1024},
		{Prompt: "a field", AspectRatio: "wide"},
		{Prompt: "a field", AspectRatio: "0:1"},
		{Prompt: "a field", Format: "webp"},
		{Prompt: "a field", Seed: &seed},
	} {
		s.Error(request.Validate(), "%+v", request)
	}
}

func (s *ImagegenSuite) TestARequestAsksForTheTermsItNames() {
	seed := int64(7)
	s.Empty(Request{Prompt: "a field", N: 4}.Terms(), "a count is something every model can do")
	s.Equal(
		[]options.Term{options.Size, options.AspectRatio, options.Seed, options.NegativePrompt, options.Format},
		Request{
			Prompt: "a field", Width: 1024, Height: 1024, AspectRatio: "1:1",
			Seed: &seed, NegativePrompt: "text", Format: FormatPNG,
		}.Terms(),
	)
}

func (s *ImagegenSuite) TestACountOfNothingIsOnePicture() {
	s.Equal(1, Request{}.Count())
	s.Equal(3, Request{N: 3}.Count())
}

func (s *ImagegenSuite) TestAPictureIsReadForWhatItIs() {
	var encoded bytes.Buffer
	s.Require().NoError(png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 30, 20))))

	picture, err := Verify(encoded.Bytes())
	s.Require().NoError(err)

	s.Equal("image/png", picture.MediaType)
	s.Equal(30, picture.Width)
	s.Equal(20, picture.Height)
}

func (s *ImagegenSuite) TestWhatIsNotAPNGOrAJPEGIsRefused() {
	var animated bytes.Buffer
	s.Require().NoError(gif.Encode(&animated, image.NewPaletted(image.Rect(0, 0, 4, 4), color.Palette{color.Black, color.White}), nil))

	_, err := Verify(animated.Bytes())
	s.ErrorContains(err, "gif")
	_, err = Verify(nil)
	s.ErrorContains(err, "empty")
}

func (s *ImagegenSuite) TestAPictureClaimingToBeHugeIsRefusedBeforeItIsInflated() {
	var encoded bytes.Buffer
	s.Require().NoError(png.Encode(&encoded, image.NewGray(image.Rect(0, 0, MaxSide+1, 1))))

	_, err := Verify(encoded.Bytes())
	s.ErrorContains(err, fmt.Sprintf("%dx1", MaxSide+1))
}

func (s *ImagegenSuite) TestARunOutContextIsWhatTheCallerDidWhateverTheProviderSaid() {
	s.Equal(Timeout, CodeOf(Fail(ProviderFailed, true, fmt.Errorf("fal: %w", context.DeadlineExceeded))))
	s.Equal(Cancelled, CodeOf(fmt.Errorf("wrapped: %w", context.Canceled)))
	s.Equal(ContentFiltered, CodeOf(fmt.Errorf("wrapped: %w", Fail(ContentFiltered, true, errors.New("no")))))
	s.Equal(ProviderFailed, CodeOf(errors.New("something else")))
	s.True(Accepted(fmt.Errorf("wrapped: %w", Fail(ProviderFailed, true, errors.New("no")))))
	s.False(Accepted(errors.New("something else")))
}
