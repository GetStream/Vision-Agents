package detectionrouter

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/detection"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// stubDetector stands in for a real provider so routing can be driven without credentials.
type stubDetector struct {
	name  string
	found detection.Result
	err   error
	asked []detection.Request
}

func (s *stubDetector) Detect(_ context.Context, request detection.Request) (detection.Result, error) {
	s.asked = append(s.asked, request)
	if s.err != nil {
		return detection.Result{}, s.err
	}
	return s.found, nil
}

func (s *stubDetector) Start(context.Context) error { return nil }
func (s *stubDetector) Close() error                { return nil }
func (s *stubDetector) Provider() string            { return s.name }
func (s *stubDetector) Model() string               { return "stub" }

type DetectionRouterSuite struct {
	suite.Suite
	ctx context.Context
	// Provider names are unique to each test so health from one does not rank the next.
	yolo  *stubDetector
	faces *stubDetector
}

func TestDetectionRouterSuite(t *testing.T) {
	suite.Run(t, new(DetectionRouterSuite))
}

func (s *DetectionRouterSuite) SetupTest() {
	s.ctx = context.Background()
	stamp := time.Now().Format("150405.000000000")
	s.yolo = &stubDetector{name: "yolo-" + stamp, found: detection.Result{Detections: []detection.Detection{
		{Label: "Person", Confidence: 0.9, Box: detection.Box{XMin: -0.1, YMin: 0.2, XMax: 0.5, YMax: 1.2}},
		{Label: "car", Confidence: 0.3, Box: detection.Box{XMin: 0.1, YMin: 0.1, XMax: 0.2, YMax: 0.2}},
	}}}
	s.faces = &stubDetector{name: "faces-" + stamp, found: detection.Result{Detections: []detection.Detection{
		{Label: "face", Confidence: 0.8, Embedding: []float32{1, 0}},
	}}}
}

func (s *DetectionRouterSuite) router() *Router {
	registry := NewRegistry()
	registry.Register(s.yolo.name, func(routing.Spec) (detection.Provider, error) { return s.yolo, nil })
	registry.Register(s.faces.name, func(routing.Spec) (detection.Provider, error) { return s.faces, nil })
	router, err := New(Options{
		Config: routing.ModalityConfig{
			Providers: []routing.ProviderConfig{
				{Provider: s.yolo.name, Model: "coco", Languages: []string{"en"}, Terms: []options.Term{options.Confidence}},
				{Provider: s.faces.name, Model: "faces", Languages: []string{"en"}, Tier: routing.HighQuality,
					Terms: []options.Term{options.Confidence, options.Embeddings}},
			},
			Aliases: map[string]routing.Alias{
				DefaultTarget: {Tier: routing.LowLatency},
				"any":         {},
			},
		},
		Registry: registry,
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	return router
}

func (s *DetectionRouterSuite) TestOnlyTheClassesAndConfidenceAskedForComeBackInsideTheImage() {
	floor := 0.5
	found, err := s.router().Detect(s.ctx, Request{
		CustomerID: "c1",
		Detection:  detection.Request{Classes: []string{"person"}, MinConfidence: &floor},
	})

	s.Require().NoError(err)
	s.Equal(s.yolo.name, found.Provider)
	s.Require().Len(found.Detections, 1)
	s.Equal("Person", found.Detections[0].Label, "a class matches its label whatever the case")
	s.Equal(detection.Box{XMin: 0, YMin: 0.2, XMax: 0.5, YMax: 1}, found.Detections[0].Box)
}

func (s *DetectionRouterSuite) TestAskingForEmbeddingsGoesOnlyToAModelThatReturnsThem() {
	found, err := s.router().Detect(s.ctx, Request{
		CustomerID: "c1", Target: "any", Detection: detection.Request{Embeddings: true},
	})

	s.Require().NoError(err)
	s.Equal(s.faces.name, found.Provider)
	s.Empty(s.yolo.asked)
	s.Equal([]float32{1, 0}, found.Detections[0].Embedding)
}

func (s *DetectionRouterSuite) TestAnEmbeddingNobodyAskedForIsDropped() {
	found, err := s.router().Detect(s.ctx, Request{CustomerID: "c1", Target: s.faces.name + "/faces"})

	s.Require().NoError(err)
	s.Nil(found.Detections[0].Embedding)
}

func (s *DetectionRouterSuite) TestAFailedProviderHandsTheImageToTheNext() {
	s.yolo.err = detection.Fail(detection.ProviderFailed, errors.New("down"))

	found, err := s.router().Detect(s.ctx, Request{CustomerID: "c1", Target: "any"})

	s.Require().NoError(err)
	s.Equal(s.faces.name, found.Provider)
}

func (s *DetectionRouterSuite) TestAnOverwriteGoesOnlyToTheProviderItNames() {
	overwrite := json.RawMessage(`{"workflow_id": "doorbell"}`)

	_, err := s.router().Detect(s.ctx, Request{
		CustomerID: "c1",
		Target:     s.faces.name + "/faces",
		Overwrites: map[string]json.RawMessage{s.faces.name: overwrite},
	})

	s.Require().NoError(err)
	s.JSONEq(string(overwrite), string(s.faces.asked[0].Overwrite))
}

func (s *DetectionRouterSuite) TestAnOverwriteForAProviderThatIsNotACandidateIsRefused() {
	_, err := s.router().Detect(s.ctx, Request{
		CustomerID: "c1",
		Overwrites: map[string]json.RawMessage{"roboflw": json.RawMessage(`{}`)},
	})

	s.Equal(detection.UnsupportedOption, detection.CodeOf(err))
	s.Empty(s.yolo.asked)
}

func (s *DetectionRouterSuite) TestATermNothingDeclaresIsRefusedWithoutACall() {
	_, err := s.router().Detect(s.ctx, Request{CustomerID: "c1", Detection: detection.Request{Embeddings: true}})

	s.Equal(detection.UnsupportedOption, detection.CodeOf(err))
	s.Empty(s.yolo.asked)
}
