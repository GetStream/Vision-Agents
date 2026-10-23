//go:build integration

// Package detectionsuite is what every detection provider is held to against its real API:
// the cat in the fixture is found, its box is a fraction of the image and covers the cat,
// and a provider that returns embeddings returns one for every face.
//
// A provider suite embeds Suite, says how to build its provider and what it finds, and
// inherits those tests.
package detectionsuite

import (
	"context"
	_ "embed"
	"os"
	"strings"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/detection"
	// The providers need their credentials, which live in the repository's .env rather
	// than in the environment an editor happens to run a test with.
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// cat is a cat filling most of a 1024 pixel square, which every COCO model knows and an
// open-vocabulary one can be asked for by name.
//
//go:embed testdata/cat.jpg
var cat []byte

// face is one person looking at the camera.
//
//go:embed testdata/face.jpg
var face []byte

// defaultTimeout bounds one detection, looser than the providers' own.
const defaultTimeout = 30 * time.Second

// Suite is the shared behaviour. The fields are set where the suite is constructed.
type Suite struct {
	suite.Suite

	// New builds a provider.
	New func() detection.Provider
	// Requires are the environment variables whose absence skips rather than fails.
	Requires []string
	// FacesOnly marks a provider that finds faces and nothing else, which has no cat to
	// find.
	FacesOnly bool
	// Embeds marks a provider that returns a vector on every box.
	Embeds bool

	// Provider is built once and never closed: a detection provider holds no connection.
	Provider detection.Provider
}

func (s *Suite) SetupSuite() {
	s.Require().NotNil(s.New, "a provider suite has to say how to build its provider")
	for _, name := range s.Requires {
		if os.Getenv(name) == "" {
			s.T().Skipf("%s not set", name)
		}
	}
	s.Provider = s.New()
}

// Detect runs one request on the suite's provider and fails the test unless it succeeds.
func (s *Suite) Detect(request detection.Request) detection.Result {
	s.Require().NoError(request.Validate())
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()
	found, err := s.Provider.Detect(ctx, request)
	s.Require().NoError(err)
	return found
}

func (s *Suite) TestTheCatIsFoundAndItsBoxCoversIt() {
	if s.FacesOnly {
		s.T().Skip("finds faces only")
	}
	found := s.Detect(detection.Request{Image: cat, Classes: []string{"cat"}})

	var cats []detection.Detection
	for _, detected := range found.Detections {
		if strings.EqualFold(detected.Label, "cat") {
			cats = append(cats, detected)
		}
	}
	s.Require().NotEmpty(cats, "the cat should be found, got %+v", found.Detections)
	box := cats[0].Box
	for _, edge := range []float64{box.XMin, box.YMin, box.XMax, box.YMax} {
		s.GreaterOrEqual(edge, 0.0)
		s.LessOrEqual(edge, 1.0)
	}
	s.Greater((box.XMax-box.XMin)*(box.YMax-box.YMin), 0.1, "the cat fills most of the picture, so its box should too")
	s.Greater(cats[0].Confidence, 0.0)
}

// TestTheWorkIsBilled: what came back says what it used, at least the image it looked at,
// which is what the detection is billed by.
func (s *Suite) TestTheWorkIsBilled() {
	image := cat
	if s.FacesOnly {
		image = face
	}
	found := s.Detect(detection.Request{Image: image, Classes: []string{"cat"}})

	s.False(found.Used().Zero(), "the detection should say what it used")
}

func (s *Suite) TestEveryFaceCarriesAnEmbedding() {
	if !s.Embeds {
		s.T().Skip("returns no embeddings")
	}
	found := s.Detect(detection.Request{Image: face, Embeddings: true})

	s.Require().NotEmpty(found.Detections, "the face should be found")
	for _, detected := range found.Detections {
		s.NotEmpty(detected.Embedding)
	}
}
