package ocr

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/png"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

type OCRSuite struct {
	suite.Suite
	page []byte
}

func TestOCRSuite(t *testing.T) {
	suite.Run(t, new(OCRSuite))
}

func (s *OCRSuite) SetupSuite() {
	var encoded bytes.Buffer
	s.Require().NoError(png.Encode(&encoded, image.NewGray(image.Rect(0, 0, 850, 1100))))
	s.page = encoded.Bytes()
}

func (s *OCRSuite) TestAPageSaysWhatKindOfPictureItIsFromThePictureItself() {
	page := Request{Image: s.page, MediaType: "image/jpeg"}

	s.Require().NoError(page.Validate())
	s.Equal("image/png", page.MediaType)
}

func (s *OCRSuite) TestBytesThatAreNotAPictureAreRefused() {
	page := Request{Image: []byte("%PDF-1.7 this is a document, not a page drawn from one")}

	s.ErrorContains(page.Validate(), "not a JPEG or PNG")
}

func (s *OCRSuite) TestAPageOverTheSizeCapIsRefusedBeforeItIsDecoded() {
	page := Request{Image: append(bytes.Clone(s.page), make([]byte, MaxBytes)...)}

	s.ErrorContains(page.Validate(), fmt.Sprint(MaxBytes))
}

func (s *OCRSuite) TestAPageIsBilledAsOnePageAndTheTokensItTook() {
	s.Equal(pricing.Quantities{pricing.Pages: 1}, Result{Markdown: "# Menu"}.Used())
	s.Equal(pricing.Quantities{pricing.Pages: 1, pricing.InputTokens: 1500, pricing.OutputTokens: 900},
		Result{InputTokens: 1500, OutputTokens: 900}.Used())
}

func (s *OCRSuite) TestAContextThatRanOutIsATimeoutWhateverTheProviderSaid() {
	err := Fail(ProviderFailed, fmt.Errorf("mistral: sending the page: %w", context.DeadlineExceeded))

	s.Equal(Timeout, CodeOf(err))
}
