//go:build integration

// Package imagegensuite is what every image generation provider is held to against its real
// API: a prompt comes back drawn, as a picture that decodes, and the result says what it used,
// which is what it is billed by.
//
// A provider suite embeds Suite, says how to build its provider, and inherits those tests.
// Anything only one provider does stays in that provider's own file.
package imagegensuite

import (
	"context"
	"os"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
	// The providers need their credentials, which live in the repository's .env rather
	// than in the environment an editor happens to run a test with.
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// defaultTimeout bounds one picture, looser than the providers' own.
const defaultTimeout = 2 * time.Minute

// apple is what every test draws: one picture of something any model can draw.
var apple = imagegen.Request{Prompt: "A red apple on a white table, photographed from above."}

// Suite is the shared behaviour. The fields are set where the suite is constructed rather
// than in a SetupSuite of the provider's own, which would shadow this one.
type Suite struct {
	suite.Suite

	// New builds a provider.
	New func() imagegen.Provider
	// Requires are the environment variables whose absence skips rather than fails.
	Requires []string
	// Timeout bounds one picture.
	Timeout time.Duration

	// Provider is built once and never closed: a picture is one request.
	Provider imagegen.Provider

	// drawn is the apple, drawn once for every test, since each picture is billed.
	drawn *imagegen.Result
}

func (s *Suite) SetupSuite() {
	s.Require().NotNil(s.New, "a provider suite has to say how to build its provider")
	for _, name := range s.Requires {
		if os.Getenv(name) == "" {
			s.T().Skipf("%s not set", name)
		}
	}
	if s.Timeout == 0 {
		s.Timeout = defaultTimeout
	}
	s.Provider = s.New()
}

// Drawn is the apple, drawn on the first call and kept for the rest.
func (s *Suite) Drawn() imagegen.Result {
	if s.drawn == nil {
		ctx, cancel := context.WithTimeout(context.Background(), s.Timeout)
		defer cancel()
		drawn, err := s.Provider.Generate(ctx, apple)
		s.Require().NoError(err)
		s.drawn = &drawn
	}
	return *s.drawn
}

// TestAPromptIsDrawn is the one that says the provider works at all: one picture comes back,
// and it is a picture.
func (s *Suite) TestAPromptIsDrawn() {
	drawn := s.Drawn()

	s.Require().Len(drawn.Images, apple.Count())
	picture := drawn.Images[0]
	s.NotEmpty(picture.Data)
	s.Contains([]string{"image/png", "image/jpeg"}, picture.MediaType)
	s.Positive(picture.Width)
	s.Positive(picture.Height)
}

// TestTheWorkIsBilled: the result says what it used, the pictures and their pixels, which is
// what the generation is billed by.
func (s *Suite) TestTheWorkIsBilled() {
	used := s.Drawn().Used()

	s.False(used.Zero(), "the picture should say what it used")
}
