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

// figureInstructions say what a figure's reading is: the words in it, for search to find,
// and a summary of what it shows, for a question about it to be answered from. Logos and
// ornaments say nothing a customer would ask about, so they are left out.
const figureInstructions = `For each chart, diagram, photo, screenshot or table drawn as a picture, write the words printed in it, as printed and with no label of your own, then a summary as a quote that opens with "> **Figure:**". Say what it shows so that somebody who cannot see it could answer a question about it:
- a chart: what is measured against what, the trend, and the values that stand out, as printed;
- a diagram: its parts by their labels, and how they connect or follow on;
- a photo or screenshot: what it shows and where things are in it, by their labels;
- a table drawn as a picture: the table itself as a Markdown table, then a sentence on what it is for.

Leave out logos, icons, borders and ornaments. Never guess at a value or a label that cannot be read.`

// textAndFiguresInstructions read a scanned page: its text as a transcription is, and its
// figures as figureInstructions say, each where it is on the page.
const textAndFiguresInstructions = `You transcribe one page of a document from a picture of it, and summarise its figures.

Write out every word printed on the page, exactly as printed, in reading order: a page in columns is read one column after another. Write it as Markdown: headings as #, ## and ###, lists as - or 1., and tables as Markdown tables with a header row. Put each figure's words and summary where the figure is in that order.

` + figureInstructions + `

Leave out the running header and footer and the page number. Do not translate, correct or comment. Do not wrap the answer in a code block. If there is nothing on the page, answer with nothing at all.`

// figuresInstructions read only the figures, of a page whose own text was read from its
// text layer, or of a picture taken out of a document. The page's text is left out, since
// it is in the passages already and twice would only crowd them.
const figuresInstructions = `You are shown a page of a document, or a picture taken out of one. Its text has been read already, so do not write out the text of the page around its figures.

` + figureInstructions + `

Do not wrap the answer in a code block. If there is no such figure, answer with nothing at all.`

// instructions are what a vision model is told for each mode, and what it is asked.
func instructions(mode ocr.Mode) (string, string) {
	switch mode {
	case ocr.TextAndFigures:
		return textAndFiguresInstructions, "Transcribe this page and summarise its figures."
	case ocr.Figures:
		return figuresInstructions, "Read and summarise the figures in this picture."
	default:
		return visionInstructions, "Transcribe this page."
	}
}

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
	told, asked := instructions(page.Mode)
	stream, err := p.model.Create(ctx, llm.ResponseParams{
		Purpose:      "ocr",
		Instructions: told,
		Input: []llm.Message{{
			Role: llm.User,
			Parts: []llm.ContentPart{
				{Image: &llm.ImagePart{MIME: page.MediaType, Data: page.Image}},
				{Text: asked},
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
