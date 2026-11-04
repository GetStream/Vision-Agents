package ocrrouter

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ocr"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// A vision language model reads a page by being asked to write out what it sees as
// Markdown. It is the fallback for an OCR model rather than a model of its own: any
// language model the LLM router can build and that accepts an image reads a page this way,
// billed by the tokens it took. One that does not accept an image is refused when it is
// built, so a page is never failed over to a model that cannot see it. The router holds
// it, as it holds every model it meters.

// visionInstructions say what a transcription is. A model left to itself describes the
// picture, summarises the page or fences the answer as code, none of which is the text.
const visionInstructions = `You transcribe one page of a document from a picture of it.

Write out every word printed on the page, exactly as printed, in reading order: a page in columns is read one column after another. Write it as Markdown: headings as #, ## and ###, lists as - or 1., and tables as Markdown tables with a header row.

Leave out the running header and footer and the page number. Do not describe pictures, summarise, translate, correct or comment. Do not wrap the answer in a code block. If there is no text on the page, answer with nothing at all.`

// visionOutputTokens caps one page. A dense page of small print is about 1,500 words, which
// is around 2,500 tokens, and a reasoning model spends some before it writes.
const visionOutputTokens = 8192

// vision reads pages with one vision language model. It satisfies ocr.Provider.
type vision struct {
	model llm.LLM
}

// newVision wraps a language model that accepts images.
func newVision(model llm.LLM) (*vision, error) {
	if !model.Capabilities().Accepts(llm.ModalityImage) {
		model.Close()
		return nil, stack.Wrap(fmt.Errorf("ocrrouter: %s/%s does not accept an image, so it cannot read a page", model.Provider(), model.Model()))
	}
	return &vision{model: model}, nil
}

// Provider is the language model's provider.
func (p *vision) Provider() string { return p.model.Provider() }

// Model is the language model.
func (p *vision) Model() string { return p.model.Model() }

// Start opens nothing: a language model is asked a request at a time.
func (p *vision) Start(context.Context) error { return nil }

// Close releases the language model.
func (p *vision) Close() error { return p.model.Close() }

// Read implements ocr.Provider.
func (p *vision) Read(ctx context.Context, page ocr.Request) (ocr.Result, error) {
	stream, err := p.model.Create(ctx, llm.ResponseParams{
		Purpose:      "ocr",
		Instructions: visionInstructions,
		Input: []llm.Message{{
			Role: llm.User,
			Parts: []llm.ContentPart{
				{Image: &llm.ImagePart{MIME: page.MediaType, Data: page.Image}},
				{Text: "Transcribe this page."},
			},
		}},
		MaxOutputTokens: visionOutputTokens,
	})
	if err != nil {
		return ocr.Result{}, ocr.Fail(ocr.CodeOf(err), stack.Wrap(fmt.Errorf("ocrrouter: asking %s: %w", p.model.Model(), err)))
	}
	response, err := llm.Collect(stream)
	if err != nil {
		return ocr.Result{}, ocr.Fail(ocr.CodeOf(err), stack.Wrap(fmt.Errorf("ocrrouter: reading %s's answer: %w", p.model.Model(), err)))
	}
	if response.Status == llm.StatusIncomplete {
		return ocr.Result{}, ocr.Fail(ocr.ProviderFailed, stack.Wrap(errors.New("ocrrouter: the transcription ran past its token limit")))
	}
	return ocr.Result{
		Markdown:     unfence(response.OutputText),
		Pages:        1,
		InputTokens:  response.Usage.InputTokens,
		OutputTokens: response.Usage.OutputTokens,
	}, nil
}

// unfence strips the code block a model puts around an answer it was told not to fence.
func unfence(answer string) string {
	trimmed := strings.TrimSpace(answer)
	for _, fence := range []string{"```markdown", "```md", "```"} {
		if rest, ok := strings.CutPrefix(trimmed, fence); ok {
			if body, closed := strings.CutSuffix(strings.TrimSpace(rest), "```"); closed {
				return strings.TrimSpace(body)
			}
		}
	}
	return trimmed
}
