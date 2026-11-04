package mistral

import (
	"context"
	_ "embed"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/ocr"
)

//go:embed testdata/page.json
var pageAnswer []byte

type MistralSuite struct {
	suite.Suite
	server *httptest.Server

	mu     sync.Mutex
	status int
	answer []byte
	asked  []map[string]any
	keys   []string
}

func TestMistralSuite(t *testing.T) {
	suite.Run(t, new(MistralSuite))
}

func (s *MistralSuite) SetupSuite() {
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var sent map[string]any
		_ = json.Unmarshal(body, &sent)
		s.mu.Lock()
		s.asked = append(s.asked, sent)
		s.keys = append(s.keys, r.Header.Get("Authorization"))
		status, answer := s.status, s.answer
		s.mu.Unlock()
		if r.URL.Path != "/v1/ocr" {
			http.NotFound(w, r)
			return
		}
		w.WriteHeader(status)
		_, _ = w.Write(answer)
	}))
}

func (s *MistralSuite) SetupTest() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.status, s.answer, s.asked, s.keys = http.StatusOK, pageAnswer, nil, nil
}

func (s *MistralSuite) provider() *Provider {
	provider, err := New(Options{APIKey: "key-1", Model: "mistral-ocr-4-1", BaseURL: s.server.URL})
	s.Require().NoError(err)
	return provider
}

func (s *MistralSuite) page() ocr.Request {
	return ocr.Request{Image: []byte("jpeg bytes"), MediaType: "image/jpeg"}
}

func (s *MistralSuite) TestThePageGoesAsADataURLWithItsHeaderAndFooterAskedForApart() {
	_, err := s.provider().Read(context.Background(), s.page())

	s.Require().NoError(err)
	s.Require().Len(s.asked, 1)
	s.Equal("Bearer key-1", s.keys[0])
	s.Equal("mistral-ocr-4-1", s.asked[0]["model"])
	s.Equal(map[string]any{"type": "image_url", "image_url": "data:image/jpeg;base64,anBlZyBieXRlcw=="}, s.asked[0]["document"])
	s.Equal(true, s.asked[0]["extract_header"], "a running head repeated on every page is not the page's text")
	s.Equal(true, s.asked[0]["extract_footer"])
	s.Equal(false, s.asked[0]["include_image_base64"], "pictures on the page are not asked for")
}

func (s *MistralSuite) TestThePageIsItsMarkdownWithoutThePicturesMistralFound() {
	read, err := s.provider().Read(context.Background(), s.page())

	s.Require().NoError(err)
	s.Equal("# Fryer cleaning\n\nDrain the oil before it cools below 60 C.\n\n| Step | Who |\n| --- | --- |\n| Drain | Cook |", read.Markdown)
	s.NotContains(read.Markdown, "Kitchen Operations")
}

func (s *MistralSuite) TestThePageIsBilledAsTheOnesMistralSaysItProcessed() {
	read, err := s.provider().Read(context.Background(), s.page())

	s.Require().NoError(err)
	s.Equal(int64(1), read.Pages)
	s.Zero(read.InputTokens, "Mistral OCR bills pages, not tokens")
}

func (s *MistralSuite) TestARefusalSaysWhatMistralSaid() {
	s.status, s.answer = http.StatusUnauthorized, []byte(`{"message":"Unauthorized","request_id":"abc"}`)

	_, err := s.provider().Read(context.Background(), s.page())

	s.Equal(ocr.ProviderFailed, ocr.CodeOf(err))
	s.ErrorContains(err, "401")
	s.ErrorContains(err, "Unauthorized")
}

func (s *MistralSuite) TestAnAnswerWithNoPageIsAFailureNotABlankPage() {
	s.answer = []byte(`{"pages": [], "usage_info": {"pages_processed": 0}}`)

	_, err := s.provider().Read(context.Background(), s.page())

	s.ErrorContains(err, "no page")
}

func (s *MistralSuite) TestWithoutAKeyItRefusesToBeBuilt() {
	s.T().Setenv(APIKeyEnvVar, "")

	_, err := New(Options{Model: "mistral-ocr-4-1"})

	s.ErrorContains(err, APIKeyEnvVar)
}

func (s *MistralSuite) TestABlankLineLeftByAPictureIsClosedUp() {
	s.Equal("One.\n\nTwo.", clean("One.\n\n![img-1.png](img-1.png)\n\n\nTwo.\n"))
	s.False(strings.HasPrefix(clean("![img-0.jpeg](img-0.jpeg)\n\nText."), "\n"))
}
