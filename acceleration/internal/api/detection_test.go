//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/detection"
	"github.com/GetStream/Vision-Agents/acceleration/internal/detectionrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// DetectionSuite covers finding objects in an image: what comes back, what it cost, and
// what a request nothing can serve is answered with.
type DetectionSuite struct {
	RouterSuite
}

func TestDetectionSuite(t *testing.T) {
	runSuite(t, new(DetectionSuite))
}

func (s *DetectionSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *DetectionSuite) TestAnImageComesBackAsBoxesAndWhatTheyCost() {
	found := s.detect(DetectionRequest{
		Image:   drawn.Images[0].Data,
		Options: &DetectionOptions{Target: pointerTo("detection-fast"), MinConfidence: pointerTo(0.5)},
		Tags:    &map[string]string{"camera": "porch"},
	})

	s.True(strings.HasPrefix(found.Id, "det_"))
	s.Equal(DetectionStatusCompleted, found.Status)
	s.Equal("coco", *found.Provider)
	s.EqualValues(750, found.CostMicros)
	s.Require().Len(found.Detections, 1, "the car under the confidence floor is dropped")
	s.Equal("person", found.Detections[0].Label)
	s.Equal(DetectionBox{XMin: 0.1, YMin: 0.2, XMax: 0.3, YMax: 0.4}, found.Detections[0].Box)
	s.Nil(found.Detections[0].Embedding)
}

func (s *DetectionSuite) TestEmbeddingsAreAskedOfTheModelThatReturnsThemWithItsOverwrite() {
	found := s.detect(DetectionRequest{
		Image: drawn.Images[0].Data,
		Options: &DetectionOptions{
			Target:     pointerTo("detection-any"),
			Embeddings: pointerTo(true),
			Overwrites: &map[string]json.RawMessage{"faces": json.RawMessage(`{"workflow_id": "doorbell"}`)},
		},
	})

	s.Equal("faces", *found.Provider)
	s.Require().Len(found.Detections, 1)
	s.Equal([]float32{1, 0}, *found.Detections[0].Embedding)
	s.JSONEq(`{"workflow_id": "doorbell"}`, string(detectors.faces.last().Overwrite))
	s.Equal(32, detectors.faces.last().Width, "the image is measured before it is sent")
}

func (s *DetectionSuite) TestAnOptionNothingInTheTargetHonoursIsRefused() {
	s.Equal(http.StatusBadRequest, s.refused(DetectionRequest{
		Image: drawn.Images[0].Data, Options: &DetectionOptions{Target: pointerTo("detection-fast"), Embeddings: pointerTo(true)},
	}))
}

func (s *DetectionSuite) TestAnOverwriteForAProviderThatIsNotACandidateIsRefused() {
	s.Equal(http.StatusBadRequest, s.refused(DetectionRequest{
		Image: drawn.Images[0].Data,
		Options: &DetectionOptions{
			Target: pointerTo("detection-fast"), Overwrites: &map[string]json.RawMessage{"faces": json.RawMessage(`{}`)},
		},
	}))
}

func (s *DetectionSuite) TestSomethingThatIsNotAnImageIsRefused() {
	s.Equal(http.StatusBadRequest, s.refused(DetectionRequest{Image: []byte("not a picture")}))
}

func (s *DetectionSuite) TestAModelNamedOutsideTheOptionsIsRefused() {
	for _, model := range []any{"nobody/nothing", "", 5, map[string]any{}, []string{"nobody/nothing"}} {
		request := map[string]any{"image": drawn.Images[0].Data}
		request["model"] = model
		s.assertTopLevelModelRefused("/v1/detection", request)
	}
}

func (s *DetectionSuite) TestANullModelIsServedAsNoModel() {
	request := map[string]any{"image": drawn.Images[0].Data}
	request["model"] = nil
	status, payload := s.serverClient.call(http.MethodPost, "/v1/detection", request)
	s.Equal(http.StatusOK, status, string(payload))
}

func (s *DetectionSuite) TestOnlyTheCustomersOwnBackendMayDetect() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		status, _ := as.call(http.MethodPost, "/v1/detection", DetectionRequest{Image: drawn.Images[0].Data})
		return status
	})
}

// detect sends an image the endpoint must answer for, whether or not anything was found.
func (s *DetectionSuite) detect(request DetectionRequest) DetectionResult {
	var found DetectionResult
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/detection", request, &found))
	return found
}

// refused is the status of a request the endpoint would not take at all.
func (s *DetectionSuite) refused(request DetectionRequest) int {
	status, _ := s.serverClient.call(http.MethodPost, "/v1/detection", request)
	return status
}

// detector stands in for a detection provider: it finds the same things every time.
type detector struct {
	name  string
	found detection.Result

	mu    sync.Mutex
	asked []detection.Request
}

var detectors = struct {
	coco, faces *detector
}{
	coco: &detector{name: "coco", found: detection.Result{Detections: []detection.Detection{
		{Label: "person", Confidence: 0.9, Box: detection.Box{XMin: 0.1, YMin: 0.2, XMax: 0.3, YMax: 0.4}},
		{Label: "car", Confidence: 0.2, Box: detection.Box{XMin: 0.5, YMin: 0.5, XMax: 0.6, YMax: 0.6}},
	}}},
	faces: &detector{name: "faces", found: detection.Result{Detections: []detection.Detection{
		{Label: "face", Confidence: 0.8, Embedding: []float32{1, 0}},
	}}},
}

func (d *detector) Detect(_ context.Context, request detection.Request) (detection.Result, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.asked = append(d.asked, request)
	return d.found, nil
}

func (d *detector) Start(context.Context) error { return nil }
func (d *detector) Close() error                { return nil }
func (d *detector) Provider() string            { return d.name }
func (d *detector) Model() string               { return "stub" }

// last is the most recent request this provider was asked.
func (d *detector) last() detection.Request {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.asked[len(d.asked)-1]
}

func detectionConfig() routing.ModalityConfig {
	return routing.ModalityConfig{
		Providers: []routing.ProviderConfig{
			{
				Provider: "coco", Model: "fast", Languages: []string{"en"}, Tier: routing.LowLatency,
				Terms: []options.Term{options.Confidence}, Price: routing.Price{PerImage: 0.00075},
			},
			{
				Provider: "faces", Model: "faces", Languages: []string{"en"}, Tier: routing.HighQuality,
				Terms: []options.Term{options.Confidence, options.Embeddings}, Price: routing.Price{PerImage: 0.0015},
			},
		},
		Aliases: map[string]routing.Alias{
			"detection-fast": {Title: "Fast detection", Tier: routing.LowLatency},
			"detection-any":  {Title: "Any detection"},
		},
	}
}

func detectorRegistry() *detectionrouter.Registry {
	registry := detectionrouter.NewRegistry()
	registry.Register("coco", func(routing.Spec) (detection.Provider, error) { return detectors.coco, nil })
	registry.Register("faces", func(routing.Spec) (detection.Provider, error) { return detectors.faces, nil })
	return registry
}
