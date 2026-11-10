// Package mistral reads pages with Mistral OCR, Mistral's model built for documents.
//
// A page is sent as a data URL and comes back as Markdown, tables included, billed by the
// page. The page's header and footer are asked for apart from its text and dropped, since
// a running head repeated on every page of a manual is what a search would otherwise
// match. So are the pictures on it, which Mistral writes as links to images it would
// otherwise return.
package mistral

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ocr"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ProviderName is how this provider is named in stats.
const ProviderName = "mistral"

// APIKeyEnvVar is where the key is read from.
const APIKeyEnvVar = "MISTRAL_API_KEY"

// DefaultBaseURL is Mistral's API.
const DefaultBaseURL = "https://api.mistral.ai"

const (
	// requestTimeout bounds one page. Mistral reads a page in a few seconds; the router
	// gives each candidate less than this, so it is only the floor under a caller with no
	// deadline of its own.
	requestTimeout = 90 * time.Second
	// responseLimit caps the answer. A page of text is a few kilobytes, and the images
	// that would make it large are not asked for.
	responseLimit  = 4 << 20
	errorBodyLimit = 512
)

// Options configures a Provider.
type Options struct {
	// APIKey defaults to MISTRAL_API_KEY.
	APIKey string
	// Model is a Mistral OCR model, such as mistral-ocr-4-1.
	Model string
	// BaseURL replaces the API host, which is what a test needs.
	BaseURL string
}

// Provider reads pages with one Mistral OCR model. It satisfies ocr.Provider.
type Provider struct {
	keys    *keypool.Pool
	model   string
	baseURL string
	client  *http.Client
}

// New validates the options and returns a Provider.
func New(options Options) (*Provider, error) {
	keys := keypool.FromEnv(options.APIKey, APIKeyEnvVar)
	if keys.Len() == 0 {
		return nil, stack.Wrap(errors.New("mistral: " + APIKeyEnvVar + " is required"))
	}
	if options.Model == "" {
		return nil, stack.Wrap(errors.New("mistral: a model is required"))
	}
	if options.BaseURL == "" {
		options.BaseURL = DefaultBaseURL
	}
	return &Provider{
		keys:    keys,
		model:   options.Model,
		baseURL: strings.TrimSuffix(options.BaseURL, "/"),
		client: &http.Client{
			Timeout:       requestTimeout,
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
		},
	}, nil
}

// Provider implements ocr.Provider.
func (p *Provider) Provider() string { return ProviderName }

// Model is the Mistral OCR model.
func (p *Provider) Model() string { return p.model }

// Start opens nothing: the key was checked when this was built.
func (p *Provider) Start(context.Context) error { return nil }

// Close releases nothing.
func (p *Provider) Close() error { return nil }

type document struct {
	Type     string `json:"type"`
	ImageURL string `json:"image_url"`
}

type request struct {
	Model              string   `json:"model"`
	Document           document `json:"document"`
	IncludeImageBase64 bool     `json:"include_image_base64"`
	IncludeBlocks      bool     `json:"include_blocks"`
	ExtractHeader      bool     `json:"extract_header"`
	ExtractFooter      bool     `json:"extract_footer"`
}

type response struct {
	Pages []struct {
		Index    int    `json:"index"`
		Markdown string `json:"markdown"`
	} `json:"pages"`
	UsageInfo struct {
		PagesProcessed int64 `json:"pages_processed"`
	} `json:"usage_info"`
}

// Read implements ocr.Provider.
func (p *Provider) Read(ctx context.Context, page ocr.Request) (ocr.Result, error) {
	// Mistral OCR transcribes and does not describe, so a page whose text was read already
	// would come back as that text again. A scan asked for with its figures is read for its
	// text, which is more than nothing.
	if page.Mode == ocr.Figures {
		return ocr.Result{}, ocr.Fail(ocr.ProviderFailed, stack.Wrap(errors.New("mistral: OCR transcribes text and cannot summarise pictures")))
	}
	body, err := json.Marshal(request{
		Model: p.model,
		Document: document{
			Type:     "image_url",
			ImageURL: "data:" + page.MediaType + ";base64," + base64.StdEncoding.EncodeToString(page.Image),
		},
		ExtractHeader: true,
		ExtractFooter: true,
	})
	if err != nil {
		return ocr.Result{}, ocr.Fail(ocr.ProviderFailed, stack.Wrap(fmt.Errorf("mistral: encoding the request: %w", err)))
	}
	sent, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/v1/ocr", bytes.NewReader(body))
	if err != nil {
		return ocr.Result{}, ocr.Fail(ocr.ProviderFailed, stack.Wrap(fmt.Errorf("mistral: building the request: %w", err)))
	}
	sent.Header.Set("Authorization", "Bearer "+p.keys.Next())
	sent.Header.Set("Content-Type", "application/json")
	sent.Header.Set("Accept", "application/json")

	answer, err := p.client.Do(sent)
	if err != nil {
		return ocr.Result{}, ocr.Fail(ocr.CodeOf(err), stack.Wrap(fmt.Errorf("mistral: sending the page: %w", err)))
	}
	defer answer.Body.Close()
	if answer.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(answer.Body, errorBodyLimit))
		return ocr.Result{}, ocr.Fail(ocr.ProviderFailed, stack.Wrap(fmt.Errorf("mistral: %s: %s", answer.Status, strings.TrimSpace(string(detail)))))
	}
	raw, err := io.ReadAll(io.LimitReader(answer.Body, responseLimit+1))
	if err != nil {
		return ocr.Result{}, ocr.Fail(ocr.CodeOf(err), stack.Wrap(fmt.Errorf("mistral: reading the answer: %w", err)))
	}
	if len(raw) > responseLimit {
		return ocr.Result{}, ocr.Fail(ocr.ProviderFailed, stack.Wrap(fmt.Errorf("mistral: the answer is over %d bytes", responseLimit)))
	}
	var read response
	if err := json.Unmarshal(raw, &read); err != nil {
		return ocr.Result{}, ocr.Fail(ocr.ProviderFailed, stack.Wrap(fmt.Errorf("mistral: decoding the answer: %w", err)))
	}
	if len(read.Pages) == 0 {
		return ocr.Result{}, ocr.Fail(ocr.ProviderFailed, stack.Wrap(errors.New("mistral: the answer has no page")))
	}

	texts := make([]string, 0, len(read.Pages))
	for _, page := range read.Pages {
		if text := clean(page.Markdown); text != "" {
			texts = append(texts, text)
		}
	}
	return ocr.Result{
		Markdown: strings.Join(texts, "\n\n"),
		Pages:    max(read.UsageInfo.PagesProcessed, int64(len(read.Pages))),
	}, nil
}

// pictureLink is a picture Mistral found on the page, written as a link to an image it
// returns only when asked to.
var pictureLink = regexp.MustCompile(`!\[[^\]]*\]\([^)]*\)`)

// clean drops the links to pictures that were not asked for, and the blank lines they
// leave.
func clean(markdown string) string {
	markdown = pictureLink.ReplaceAllString(markdown, "")
	var kept []string
	blank := false
	for _, line := range strings.Split(markdown, "\n") {
		if strings.TrimSpace(line) == "" {
			blank = len(kept) > 0
			continue
		}
		if blank {
			kept = append(kept, "")
			blank = false
		}
		kept = append(kept, strings.TrimRight(line, " \t"))
	}
	return strings.Join(kept, "\n")
}
