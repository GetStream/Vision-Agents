package ultralytics

import (
	"context"
	_ "embed"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/detection"
)

//go:embed testdata/bus.json
var bus []byte

// asked is one call the stub server saw.
type asked struct {
	path          string
	authorization string
	normalize     string
	conf          string
	file          string
}

type UltralyticsSuite struct {
	suite.Suite
	server *httptest.Server

	mu    sync.Mutex
	calls []asked
}

func TestUltralyticsSuite(t *testing.T) {
	suite.Run(t, new(UltralyticsSuite))
}

func (s *UltralyticsSuite) SetupSuite() {
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		call := asked{
			path:          r.URL.Path,
			authorization: r.Header.Get("Authorization"),
			normalize:     r.FormValue("normalize"),
			conf:          r.FormValue("conf"),
		}
		if files := r.MultipartForm.File["file"]; len(files) == 1 {
			opened, _ := files[0].Open()
			data := make([]byte, files[0].Size)
			_, _ = opened.Read(data)
			call.file = string(data)
		}
		s.mu.Lock()
		s.calls = append(s.calls, call)
		s.mu.Unlock()
		_, _ = w.Write(bus)
	}))
}

func (s *UltralyticsSuite) SetupTest() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls = nil
}

func (s *UltralyticsSuite) seen() []asked {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls
}

func frame() detection.Request {
	return detection.Request{Image: []byte("jpeg"), MediaType: "image/jpeg"}
}

func (s *UltralyticsSuite) TestOneOfUltralyticsOwnModelsIsCalledAnonymouslyForFractions() {
	provider, err := New(Options{Model: "yolo26/yolo26n", BaseURL: s.server.URL})
	s.Require().NoError(err)

	found, err := provider.Detect(context.Background(), frame())

	s.Require().NoError(err)
	s.Require().Len(found.Detections, 2)
	s.Equal("bus", found.Detections[0].Label)
	s.Equal(detection.Box{XMin: 0, YMin: 0.21, XMax: 0.99, YMax: 0.69}, found.Detections[0].Box)
	call := s.seen()[0]
	s.Equal("/ultralytics/yolo26/yolo26n/predict", call.path)
	s.Empty(call.authorization)
	s.Equal("true", call.normalize)
	s.Equal("jpeg", call.file)
}

func (s *UltralyticsSuite) TestAKeyAndAConfidenceFloorAreSentWhenThereIsOne() {
	provider, err := New(Options{APIKey: "ul-key", Model: "acme/doors/v2", BaseURL: s.server.URL})
	s.Require().NoError(err)
	request := frame()
	floor := 0.005
	request.MinConfidence = &floor

	_, err = provider.Detect(context.Background(), request)

	s.Require().NoError(err)
	call := s.seen()[0]
	s.Equal("/acme/doors/v2/predict", call.path)
	s.Equal("Bearer ul-key", call.authorization)
	s.Equal("0.01", call.conf, "Ultralytics refuses a floor below 0.01")
}

func (s *UltralyticsSuite) TestADedicatedEndpointReplacesTheSharedAPI() {
	provider, err := New(Options{Model: "yolo26/yolo26n", Endpoint: s.server.URL + "/predict"})
	s.Require().NoError(err)

	_, err = provider.Detect(context.Background(), frame())

	s.Require().NoError(err)
	s.Equal("/predict", s.seen()[0].path)
}
