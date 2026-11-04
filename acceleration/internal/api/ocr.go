package api

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/danielgtaylor/huma/v2"
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/ocr"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ocrrouter"
)

// ocrDeadline bounds one page, failover included: each candidate has
// ocrrouter.DefaultAttemptTimeout, so this is enough for the first to time out and two more
// to answer.
const ocrDeadline = 150 * time.Second

// maxOCRBody is the largest page, base64, and room for the rest of the request.
const maxOCRBody = ocr.MaxBytes*4/3 + 64<<10

// errNoOCR is what the OCR path says on a deployment that does not read pages.
var errNoOCR = notConfigured("this deployment does not read pages")

// readPage reads the text of one page and returns it as Markdown.
//
// It is answered inline and nothing is kept. A reading that reached a provider is a 200
// whether or not it read anything, the way a detection is, since a failure is part of the
// answer rather than something wrong with the request.
func (s *Server) readPage(ctx context.Context, request *readPageRequest) (*readPageResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.streams == nil || s.streams.OCR == nil {
		return nil, errNoOCR
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	if request.Body.Model != nil {
		return nil, errTopLevelModel
	}
	body := request.Body
	page := ocr.Request{Image: body.Image}
	if err := page.Validate(); err != nil {
		return nil, invalidRequest(err.Error())
	}
	tags := tagsSent(body.Tags)
	if err := tags.Validate(); err != nil {
		return nil, invalidRequest(err.Error())
	}

	sent := value(body.Options)
	ctx, cancel := context.WithTimeout(ctx, ocrDeadline)
	defer cancel()
	reading, err := s.streams.OCR.Read(ctx, ocrrouter.Request{
		CustomerID: customerID,
		Tags:       tags,
		DataPolicy: dataPolicyOf(sent.DataPolicy),
		Target:     value(sent.Target),
		Providers:  value(sent.Providers),
		Page:       page,
	})
	// Every candidate refused for money is a refusal, not a page that failed.
	if answer, refused := moneyError(err); refused {
		return nil, answer
	}
	var failure *ocr.Error
	if err != nil && !errors.As(err, &failure) {
		return nil, invalidRequest(err.Error())
	}
	return &readPageResponse{Body: ocrResultOf(reading, err)}, nil
}

// ocrResultOf renders a reading for the wire, and why nothing was read when it failed.
func ocrResultOf(reading ocrrouter.Reading, failure error) OcrResult {
	rendered := OcrResult{
		Id:         "ocr_" + uuid.NewString(),
		Status:     OcrStatusCompleted,
		Provider:   optional(reading.Provider),
		Model:      optional(reading.Model),
		Markdown:   reading.Markdown,
		CostMicros: reading.CostMicros,
	}
	if failure != nil {
		code := OcrErrorCode(ocr.CodeOf(failure))
		rendered.Status = OcrStatusFailed
		rendered.ErrorCode = &code
		rendered.Error = optional(failure.Error())
		rendered.Markdown = ""
	}
	return rendered
}

// registerOCR declares the operations served in ocr.go.
func (s *Server) registerOCR(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "readPage",
		Method:      http.MethodPost,
		Path:        "/v1/ocr",
		Summary:     "Read the text of one page, and return it as Markdown",
		Description: "Routed like detection: a target or a priority list picks the model, failover and billing " +
			"work as they do everywhere else, and one page is one stat row. ocr-default asks Mistral OCR and " +
			"fails over to vision language models that transcribe the page, and to nothing that cannot read " +
			"one; ocr-vision asks only those. Mistral OCR is billed per page and a vision model by the " +
			"tokens it took.\n" +
			"A candidate that fails, or takes longer than 45 seconds, hands the page to the next one. A " +
			"reading that got as far as a provider answers 200 whether it read anything or not: a failed one " +
			"carries status failed, an error_code and the error, and costs nothing. A page with no text on " +
			"it is completed with empty markdown. Nothing is stored, so the id cannot be fetched again.",
		Responses: map[string]*huma.Response{
			"200": {Description: "What the page says, or why nothing was read"},
		},
		MaxBodyBytes: maxOCRBody,
		Errors:       []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.readPage)
}

type readPageRequest struct {
	Body *OcrRequest `required:"true"`
}

type readPageResponse struct {
	Body OcrResult
}

// OcrRequest is the OcrRequest schema.
type OcrRequest struct {
	Image   []byte             `json:"image" doc:"The page, base64. A JPEG or PNG of at most 10 MiB and 10,000 pixels a side; a page drawn at 150 to 200 dpi reads well." format:"byte" nullable:"false"`
	Options *OcrOptions        `json:"options,omitempty"`
	Tags    *map[string]string `json:"tags,omitempty"`

	// Model is never accepted: it is read only to refuse it (errTopLevelModel).
	Model any `json:"model,omitempty" hidden:"true"`
}

// OcrOptions is the OcrOptions schema.
type OcrOptions struct {
	DataPolicy *DataPolicy `json:"data_policy,omitempty"`
	Providers  *[]string   `json:"providers,omitempty" doc:"A priority list of where to try, in the order given, which wins over target when it holds anything."`
	Target     *string     `json:"target,omitempty" doc:"A provider/model or a shortcut. Defaults to ocr-default." example:"ocr-vision"`
}

func (*OcrOptions) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Who reads the page, and what may happen to it afterwards."
	return schema
}

// OcrResult is the OcrResult schema.
type OcrResult struct {
	CostMicros int64         `json:"cost_micros" doc:"Millionths of a dollar, priced per page or per token from what the model billed. Zero when it failed."`
	Error      *string       `json:"error,omitempty" doc:"What went wrong, in words. Absent when the page was read."`
	ErrorCode  *OcrErrorCode `json:"error_code,omitempty"`
	Id         string        `json:"id" doc:"This response's own id, for logs. Nothing is stored under it." example:"ocr_1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed"`
	Markdown   string        `json:"markdown" doc:"The page's text as Markdown, headings, lists and tables kept. Empty for a page with no text on it, or when the reading failed."`
	Model      *string       `json:"model,omitempty" example:"mistral-ocr-4-1"`
	Provider   *string       `json:"provider,omitempty" doc:"Who read it, or who failed to. Absent when nothing got as far as a provider." example:"mistral"`
	Status     OcrStatus     `json:"status"`
}

// OcrStatus is the OcrStatus schema.
type OcrStatus string

// Defines values for OcrStatus.
const (
	OcrStatusCompleted OcrStatus = "completed"
	OcrStatusFailed    OcrStatus = "failed"
)

func (OcrStatus) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "OcrStatus", "Whether the page was read. A failed reading says why in error_code and error.", "completed", "failed")
}

// OcrErrorCode is the OcrErrorCode schema.
type OcrErrorCode string

func (OcrErrorCode) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "OcrErrorCode", "Why a page was not read, absent when it was. provider_failed is every candidate failing; timeout the 150 seconds running out; cancelled the caller hanging up.", "provider_failed", "timeout", "cancelled")
}
