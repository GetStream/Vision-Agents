package roboflow

import (
	"context"
	"crypto/rand"
	_ "embed"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/detection"
)

//go:embed testdata/model_two_people.json
var modelTwoPeople []byte

//go:embed testdata/workflow_two_faces.json
var workflowTwoFaces []byte

//go:embed testdata/workflow_missing_embedding.json
var workflowMissingEmbedding []byte

// asked is one call the stub server saw.
type asked struct {
	path  string
	query url.Values
	body  []byte
}

type RoboflowSuite struct {
	suite.Suite
	server *httptest.Server

	mu     sync.Mutex
	answer []byte
	calls  []asked
	// limitedKey, when set, is a key the server answers 429 to.
	limitedKey string
}

func TestRoboflowSuite(t *testing.T) {
	suite.Run(t, new(RoboflowSuite))
}

func (s *RoboflowSuite) SetupSuite() {
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		s.mu.Lock()
		defer s.mu.Unlock()
		s.calls = append(s.calls, asked{path: r.URL.Path, query: r.URL.Query(), body: body})
		var sent struct {
			APIKey string `json:"api_key"`
		}
		_ = json.Unmarshal(body, &sent)
		if s.limitedKey != "" && (r.URL.Query().Get("api_key") == s.limitedKey || sent.APIKey == s.limitedKey) {
			http.Error(w, "slow down", http.StatusTooManyRequests)
			return
		}
		_, _ = w.Write(s.answer)
	}))
}

func (s *RoboflowSuite) SetupTest() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.answer = modelTwoPeople
	s.calls = nil
	s.limitedKey = ""
}

func (s *RoboflowSuite) answering(body []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.answer = body
}

func (s *RoboflowSuite) seen() []asked {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.calls
}

func (s *RoboflowSuite) provider(model string) *Provider {
	provider, err := New(Options{APIKey: "rf-key", Model: model, BaseURL: s.server.URL})
	s.Require().NoError(err)
	return provider
}

func frame() detection.Request {
	return detection.Request{Image: []byte("jpeg"), MediaType: "image/jpeg", Width: 800, Height: 400}
}

func (s *RoboflowSuite) TestAHostedModelsCentreBoxesComeBackAsFractions() {
	found, err := s.provider("coco/3").Detect(context.Background(), frame())

	s.Require().NoError(err)
	s.Require().Len(found.Detections, 2)
	person := found.Detections[0]
	s.Equal("person", person.Label)
	s.InDelta(0.9, person.Confidence, 1e-9)
	s.Equal(detection.Box{XMin: 0.1875, YMin: 0.25, XMax: 0.3125, YMax: 0.75}, person.Box)
	s.Equal(detection.Box{XMin: 0.5, YMin: 0, XMax: 1, YMax: 0.5}, found.Detections[1].Box)
}

func (s *RoboflowSuite) TestAHostedModelIsSentTheImageTheKeyAndTheConfidenceFloor() {
	request := frame()
	floor := 0.4
	request.MinConfidence = &floor

	_, err := s.provider("coco/3").Detect(context.Background(), request)

	s.Require().NoError(err)
	call := s.seen()[0]
	s.Equal("/coco/3", call.path)
	s.Equal("rf-key", call.query.Get("api_key"))
	s.Equal("0.4", call.query.Get("confidence"))
	s.Equal("anBlZw==", string(call.body), "the body is the image, base64")
}

func (s *RoboflowSuite) TestTheFaceWorkflowGivesEveryFaceItsOwnEmbedding() {
	s.answering(workflowTwoFaces)
	request := frame()
	request.Embeddings = true

	found, err := s.provider(ModelFaces).Detect(context.Background(), request)

	s.Require().NoError(err)
	s.Require().Len(found.Detections, 2)
	s.Equal([]float32{0.1, 0.2, 0.3}, found.Detections[0].Embedding)
	s.Equal([]float32{0.4, 0.5, 0.6}, found.Detections[1].Embedding)
	s.Equal(detection.Box{XMin: 0.2, YMin: 0.2, XMax: 0.3, YMax: 0.3}, found.Detections[0].Box)
	var sent map[string]any
	s.Require().NoError(json.Unmarshal(s.seen()[0].body, &sent))
	s.Equal("/workflows/run", s.seen()[0].path)
	s.Contains(sent, "specification", "the face Workflow is sent whole")
	s.Equal("rf-key", sent["api_key"])
}

func (s *RoboflowSuite) TestARateLimitedKeyHandsTheWorkflowToTheNextKey() {
	s.answering(workflowTwoFaces)
	first, second := "rf-"+rand.Text(), "rf-"+rand.Text()
	s.limitedKey = first
	provider, err := New(Options{APIKey: first + "," + second, Model: ModelFaces, BaseURL: s.server.URL})
	s.Require().NoError(err)

	found, err := provider.Detect(context.Background(), frame())

	s.Require().NoError(err)
	s.Len(found.Detections, 2)
	var keys []any
	for _, call := range s.seen() {
		var sent map[string]any
		s.Require().NoError(json.Unmarshal(call.body, &sent))
		keys = append(keys, sent["api_key"])
	}
	s.Equal([]any{first, second}, keys)
}

func (s *RoboflowSuite) TestASavedWorkflowIsCalledByItsWorkspaceAndId() {
	s.answering(workflowTwoFaces)
	request := frame()
	request.Overwrite = json.RawMessage(`{"workspace": "acme", "workflow_id": "doorbell"}`)

	_, err := s.provider(ModelWorkflow).Detect(context.Background(), request)

	s.Require().NoError(err)
	s.Equal("/infer/workflows/acme/doorbell", s.seen()[0].path)
}

func (s *RoboflowSuite) TestAWorkflowNamingNoWorkflowIsRefusedWithoutACall() {
	_, err := s.provider(ModelWorkflow).Detect(context.Background(), frame())

	s.Equal(detection.UnsupportedOption, detection.CodeOf(err))
	s.Empty(s.seen())
}

func (s *RoboflowSuite) TestAWorkflowMissingAnEmbeddingForABoxIsRefused() {
	s.answering(workflowMissingEmbedding)
	request := frame()
	request.Embeddings = true

	_, err := s.provider(ModelFaces).Detect(context.Background(), request)

	s.Equal(detection.UnsupportedOption, detection.CodeOf(err))
}

func (s *RoboflowSuite) TestAModelThatIsNotAProjectAndVersionIsRefused() {
	_, err := New(Options{APIKey: "rf-key", Model: "rfdetr-nano"})

	s.ErrorContains(err, "project/version")
}
