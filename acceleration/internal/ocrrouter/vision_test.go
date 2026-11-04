package ocrrouter

import (
	"context"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ocr"
)

// visionModel stands in for a language model, answering every request with response.
type visionModel struct {
	accepts  []string
	response llm.Response
	asked    []llm.ResponseParams
	closed   bool
}

func (m *visionModel) Create(_ context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	m.asked = append(m.asked, params)
	return llm.Replay(llm.ResponseCompleted{Response: m.response}), nil
}

func (m *visionModel) Provider() string { return "openai" }
func (m *visionModel) Model() string    { return "gpt-6-luna" }
func (m *visionModel) Capabilities() llm.Capabilities {
	return llm.Capabilities{InputModalities: m.accepts}
}
func (m *visionModel) Close() error { m.closed = true; return nil }

type VisionSuite struct {
	suite.Suite
	page ocr.Request
}

func TestVisionSuite(t *testing.T) {
	suite.Run(t, new(VisionSuite))
}

func (s *VisionSuite) SetupTest() {
	s.page = ocr.Request{Image: []byte("jpeg bytes"), MediaType: "image/jpeg"}
}

func (s *VisionSuite) seeing(answer string) *visionModel {
	return &visionModel{accepts: []string{llm.ModalityImage}, response: llm.Response{
		Status:     llm.StatusCompleted,
		OutputText: answer,
		Usage:      llm.Usage{InputTokens: 1480, OutputTokens: 612},
	}}
}

func (s *VisionSuite) TestAModelThatCannotSeeIsRefusedWhenItIsBuilt() {
	// A page failed over to a model that cannot see it would come back as an apology.
	blind := &visionModel{}

	_, err := newVision(blind)

	s.ErrorContains(err, "does not accept an image")
	s.True(blind.closed)
}

func (s *VisionSuite) TestThePageIsSentAsAPictureWithWhatATranscriptionIs() {
	seeing := s.seeing("# Menu")
	provider, err := newVision(seeing)
	s.Require().NoError(err)

	_, err = provider.Read(context.Background(), s.page)

	s.Require().NoError(err)
	s.Require().Len(seeing.asked, 1)
	asked := seeing.asked[0]
	s.Contains(asked.Instructions, "Markdown")
	s.Contains(asked.Instructions, "Do not describe pictures")
	s.Require().NotNil(asked.Input[0].Parts[0].Image)
	s.Equal(llm.ImagePart{MIME: "image/jpeg", Data: []byte("jpeg bytes")}, *asked.Input[0].Parts[0].Image)
	s.Empty(asked.Tools, "a model offered a tool will reach for it")
}

func (s *VisionSuite) TestAnAnswerFencedAsCodeIsTheTextInside() {
	provider, err := newVision(s.seeing("```markdown\n# Menu\n\n- Fries\n```"))
	s.Require().NoError(err)

	read, err := provider.Read(context.Background(), s.page)

	s.Require().NoError(err)
	s.Equal("# Menu\n\n- Fries", read.Markdown)
}

func (s *VisionSuite) TestThePageIsBilledTheTokensItTook() {
	provider, err := newVision(s.seeing("Text."))
	s.Require().NoError(err)

	read, err := provider.Read(context.Background(), s.page)

	s.Require().NoError(err)
	s.Equal(ocr.Result{Markdown: "Text.", Pages: 1, InputTokens: 1480, OutputTokens: 612}, read)
}

func (s *VisionSuite) TestATranscriptionCutShortIsAFailureNotHalfAPage() {
	cut := s.seeing("# Menu\n\n- Fr")
	cut.response.Status = llm.StatusIncomplete
	provider, err := newVision(cut)
	s.Require().NoError(err)

	_, err = provider.Read(context.Background(), s.page)

	s.Equal(ocr.ProviderFailed, ocr.CodeOf(err))
	s.ErrorContains(err, "token limit")
}
