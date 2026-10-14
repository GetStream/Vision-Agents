package moondream

import (
	"context"
	_ "embed"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/detection"
)

//go:embed testdata/two_objects.json
var twoObjects []byte

type MoondreamSuite struct {
	suite.Suite
	server *httptest.Server

	mu    sync.Mutex
	calls []detectRequest
	keys  []string
}

func TestMoondreamSuite(t *testing.T) {
	suite.Run(t, new(MoondreamSuite))
}

func (s *MoondreamSuite) SetupSuite() {
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/detect" {
			http.NotFound(w, r)
			return
		}
		var body detectRequest
		_ = json.NewDecoder(r.Body).Decode(&body)
		s.mu.Lock()
		s.calls = append(s.calls, body)
		s.keys = append(s.keys, r.Header.Get("X-Moondream-Auth"))
		s.mu.Unlock()
		_, _ = w.Write(twoObjects)
	}))
}

func (s *MoondreamSuite) SetupTest() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls, s.keys = nil, nil
}

func (s *MoondreamSuite) provider() *Provider {
	provider, err := New(Options{APIKey: "md-key", Model: "moondream3-preview", BaseURL: s.server.URL})
	s.Require().NoError(err)
	return provider
}

func (s *MoondreamSuite) TestEachClassIsItsOwnCallAndItsOwnLabel() {
	found, err := s.provider().Detect(context.Background(), detection.Request{
		Image: []byte("jpeg"), MediaType: "image/jpeg", Classes: []string{"package", "person"},
	})

	s.Require().NoError(err)
	s.Equal(2, found.Billed, "two words are two billed calls")
	s.Require().Len(found.Detections, 4)
	s.Equal("package", found.Detections[0].Label)
	s.Equal("person", found.Detections[2].Label)
	s.Equal(detection.Box{XMin: 0.1, YMin: 0.2, XMax: 0.3, YMax: 0.4}, found.Detections[0].Box)
	s.Equal(1.0, found.Detections[0].Confidence, "Moondream scores nothing")

	s.mu.Lock()
	defer s.mu.Unlock()
	s.Require().Len(s.calls, 2)
	s.Equal("package", s.calls[0].Object)
	s.Equal("moondream3-preview", s.calls[0].Variant)
	s.Equal("data:image/jpeg;base64,anBlZw==", s.calls[0].ImageURL)
	s.Equal([]string{"md-key", "md-key"}, s.keys)
}

func (s *MoondreamSuite) TestARequestNamingNothingToLookForIsRefusedWithoutACall() {
	_, err := s.provider().Detect(context.Background(), detection.Request{Image: []byte("jpeg"), MediaType: "image/jpeg"})

	s.Equal(detection.UnsupportedOption, detection.CodeOf(err))
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Empty(s.calls)
}
