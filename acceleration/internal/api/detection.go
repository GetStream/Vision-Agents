package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"time"

	"github.com/danielgtaylor/huma/v2"
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/detection"
	"github.com/GetStream/Vision-Agents/acceleration/internal/detectionrouter"
)

// detectionDeadline bounds one detection. A worker sampling a camera sends the next frame
// a second later, so an answer that took longer than this is one nobody is waiting for.
const detectionDeadline = 30 * time.Second

// errNoDetection is what the detection path says on a deployment that does not detect.
var errNoDetection = notConfigured("this deployment does not detect objects")

// detect finds objects in one image and returns their boxes.
//
// It is answered inline and nothing is kept. A detection that reached a provider is a 200
// whether or not it found anything, the way an image generation is, since a failure is
// part of the answer rather than something wrong with the request.
func (s *Server) detect(ctx context.Context, request *detectRequest) (*detectResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.streams == nil || s.streams.Detection == nil {
		return nil, errNoDetection
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	if request.Body.Model != nil {
		return nil, errTopLevelModel
	}
	body := request.Body
	sent := value(body.Options)
	looking := detection.Request{
		Image:         body.Image,
		Classes:       value(sent.Classes),
		MinConfidence: sent.MinConfidence,
		Embeddings:    value(sent.Embeddings),
	}
	if err := looking.Validate(); err != nil {
		return nil, invalidRequest(err.Error())
	}
	tags := tagsSent(body.Tags)
	if err := tags.Validate(); err != nil {
		return nil, invalidRequest(err.Error())
	}

	ctx, cancel := context.WithTimeout(ctx, detectionDeadline)
	defer cancel()
	found, err := s.streams.Detection.Detect(ctx, detectionrouter.Request{
		CustomerID: customerID,
		Tags:       tags,
		Target:     value(sent.Target),
		Providers:  value(sent.Providers),
		Overwrites: value(sent.Overwrites),
		Detection:  looking,
	})
	// Every candidate refused for money is a refusal, not a detection that failed.
	if answer, refused := moneyError(err); refused {
		return nil, answer
	}
	var failure *detection.Error
	if err != nil && !errors.As(err, &failure) {
		return nil, invalidRequest(err.Error())
	}
	// A request no candidate could honour is the caller's to change, so it is a 400 like
	// any other the router refused before asking anybody.
	if err != nil && found.Provider == "" && detection.CodeOf(err) == detection.UnsupportedOption {
		return nil, invalidRequest(err.Error())
	}
	return &detectResponse{Body: detectionOf(found, err)}, nil
}

// detectionOf renders what was found for the wire, and why nothing was when it failed.
func detectionOf(found detectionrouter.Found, failure error) DetectionResult {
	rendered := DetectionResult{
		Id:         "det_" + uuid.NewString(),
		Status:     DetectionStatusCompleted,
		Provider:   optional(found.Provider),
		Model:      optional(found.Model),
		Detections: make([]DetectedObject, 0, len(found.Detections)),
		CostMicros: found.CostMicros,
	}
	for _, detected := range found.Detections {
		object := DetectedObject{
			Label:      detected.Label,
			Confidence: detected.Confidence,
			Box: DetectionBox{
				XMin: detected.Box.XMin, YMin: detected.Box.YMin, XMax: detected.Box.XMax, YMax: detected.Box.YMax,
			},
		}
		if detected.Embedding != nil {
			embedding := detected.Embedding
			object.Embedding = &embedding
		}
		rendered.Detections = append(rendered.Detections, object)
	}
	if failure != nil {
		code := DetectionErrorCode(detection.CodeOf(failure))
		rendered.Status = DetectionStatusFailed
		rendered.ErrorCode = &code
		rendered.Error = optional(failure.Error())
	}
	return rendered
}

// registerDetection declares the operations served in detection.go.
func (s *Server) registerDetection(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "detect",
		Method:      http.MethodPost,
		Path:        "/v1/detection",
		Summary:     "Find objects in one image, and return their boxes",
		Description: "Routed like image generation: a target or a priority list picks the model, failover and " +
			"billing work as they do everywhere else, and one request is one stat row counting the images " +
			"the vendor billed. Every box comes back as fractions of the image, whatever the vendor sent.\n" +
			"detection-fast finds COCO's 80 classes quickly; detection-open finds whatever is named in " +
			"classes; detection-faces finds faces with an embedding each. A confidence floor or embeddings " +
			"narrow the candidates to the models that declared them, so they are honoured or the request is " +
			"a 400.\n" +
			"A detection that got as far as a provider answers 200 whether it found anything or not: a " +
			"failed one carries status failed, an error_code and the error, and costs nothing. Nothing is " +
			"stored, so the id cannot be fetched again.",
		Responses: map[string]*huma.Response{
			"200": {Description: "What was found, or why nothing was"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.detect)
}

type detectRequest struct {
	Body *DetectionRequest `required:"true"`
}

type detectResponse struct {
	Body DetectionResult
}

// DetectionRequest is the DetectionRequest schema.
type DetectionRequest struct {
	Image   []byte             `json:"image" doc:"The image, base64. A JPEG or PNG of at most 4 MB and 4096 pixels a side; a video frame scaled to 1024 wide is plenty." format:"byte" nullable:"false"`
	Options *DetectionOptions  `json:"options,omitempty"`
	Tags    *map[string]string `json:"tags,omitempty"`

	// Model is never accepted: it is read only to refuse it (errTopLevelModel).
	Model any `json:"model,omitempty" hidden:"true"`
}

// DetectionOptions is the DetectionOptions schema.
type DetectionOptions struct {
	Classes       *[]string                   `json:"classes,omitempty" doc:"The labels to keep. An open-vocabulary model (detection-open) looks for these words and has to be given at least one; a fixed-class model returns only these of what it found. Matched without regard to case." maxItems:"8"`
	Embeddings    *bool                       `json:"embeddings,omitempty" doc:"Return a vector on every box, for matching what was found against what was seen before. Only a model that computes one is asked."`
	MinConfidence *float64                    `json:"min_confidence,omitempty" doc:"Drop boxes scored below this. Only a model that scores its boxes is asked." minimum:"0" maximum:"1"`
	Overwrites    *map[string]json.RawMessage `json:"overwrites,omitempty" doc:"Settings for one provider that the options above have no word for, keyed by provider. roboflow takes workspace and workflow_id, or specification, for the roboflow/workflow model. An overwrite for a provider that is not a candidate is refused."`
	Providers     *[]string                   `json:"providers,omitempty" doc:"A priority list of where to try, in the order given, which wins over target when it holds anything."`
	Target        *string                     `json:"target,omitempty" doc:"A provider/model or a capability shortcut. Defaults to detection-fast." example:"detection-open"`
}

func (*DetectionOptions) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Where to look and what to keep."
	return schema
}

// DetectionResult is the DetectionResult schema.
type DetectionResult struct {
	CostMicros int64               `json:"cost_micros" doc:"Millionths of a dollar, priced per image the vendor billed. Zero when it failed."`
	Detections []DetectedObject    `json:"detections" doc:"What was found. Empty when nothing was, or when it failed." nullable:"false"`
	Error      *string             `json:"error,omitempty" doc:"What went wrong, in words. Absent when the detection completed."`
	ErrorCode  *DetectionErrorCode `json:"error_code,omitempty"`
	Id         string              `json:"id" doc:"This response's own id, for logs. Nothing is stored under it." example:"det_1b9d6bcd-bbfd-4b2d-9b5d-ab8dfbbd4bed"`
	Model      *string             `json:"model,omitempty" example:"moondream3-preview"`
	Provider   *string             `json:"provider,omitempty" doc:"Who looked, or who failed to. Absent when nothing got as far as a provider." example:"moondream"`
	Status     DetectionStatus     `json:"status"`
}

// DetectedObject is the DetectedObject schema.
type DetectedObject struct {
	Box        DetectionBox `json:"box"`
	Confidence float64      `json:"confidence" doc:"The model's score from 0 to 1. A model that scores nothing, such as Moondream, reports 1."`
	Embedding  *[]float32   `json:"embedding,omitempty" doc:"A vector describing what is in the box. Present only when embeddings was asked for."`
	Label      string       `json:"label" example:"person"`
}

// DetectionBox is the DetectionBox schema.
type DetectionBox struct {
	XMax float64 `json:"x_max" minimum:"0" maximum:"1"`
	XMin float64 `json:"x_min" minimum:"0" maximum:"1"`
	YMax float64 `json:"y_max" minimum:"0" maximum:"1"`
	YMin float64 `json:"y_min" minimum:"0" maximum:"1"`
}

func (*DetectionBox) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A rectangle as fractions of the image: 0,0 is the top left corner and 1,1 the bottom right."
	return schema
}

// DetectionStatus is the DetectionStatus schema.
type DetectionStatus string

// Defines values for DetectionStatus.
const (
	DetectionStatusCompleted DetectionStatus = "completed"
	DetectionStatusFailed    DetectionStatus = "failed"
)

func (DetectionStatus) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "DetectionStatus", "Whether the image was looked at. A failed detection says why in error_code and error.", "completed", "failed")
}

// DetectionErrorCode is the DetectionErrorCode schema.
type DetectionErrorCode string

func (DetectionErrorCode) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "DetectionErrorCode", "Why a detection found nothing, absent when it completed. unsupported_option is a class, setting or overwrite no candidate could honour; provider_failed anything else a provider did wrong; timeout the 30 seconds running out; cancelled the caller hanging up.", "unsupported_option", "provider_failed", "timeout", "cancelled")
}
