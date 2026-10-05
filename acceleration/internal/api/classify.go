package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/lcm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/lcmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/danielgtaylor/huma/v2"
)

// classify puts a caller's questions about a piece of text to a routed classifier.
//
// It is search's shape for the same reason: one request and one set of answers, with
// nothing arriving in pieces, so there is no socket. Every question goes in one request,
// because they are answered independently and share the state's tokens between them.
func (s *Server) classify(ctx context.Context, request *classifyRequest) (*classifyResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.streams == nil || s.streams.LCM == nil {
		return nil, huma.Error404NotFound("this deployment does not route classification")
	}
	if request.Body == nil {
		return nil, huma.Error400BadRequest("a request body is required")
	}
	if text, ok := request.Body.State.(string); request.Body.State == nil || ok && strings.TrimSpace(text) == "" {
		return nil, huma.Error400BadRequest("there is nothing to judge")
	}

	asked, err := lcmRequestOf(request.Body)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	if err := asked.Validate(); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	tags := tagsSent(request.Body.Tags)
	if err := tags.Validate(); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}

	held := options.Classifier{Target: value(request.Body.Target)}
	if _, err := s.streams.LCM.Resolve(ctx, held.Route(), nil); err != nil {
		return nil, huma.Error404NotFound(err.Error())
	}

	session, err := s.streams.LCM.Start(ctx, lcmrouter.Request{
		CustomerID: customerID,
		Tags:       tags,
		Options:    held,
	})
	if err != nil {
		return nil, classifyFailed(err)
	}
	defer session.Close()

	result, err := session.Classify(ctx, asked)
	if err != nil {
		return nil, classifyFailed(err)
	}

	answers := make(map[string]ClassifyAnswer, len(result.Answers))
	for id, answer := range result.Answers {
		answers[id] = classifyAnswerOf(asked.Questions[id].Type, answer)
	}
	model := session.Model()
	if result.Model != "" {
		model = result.Model
	}
	return &classifyResponse{Body: ClassifyResult{Provider: session.Provider(),
		Model:   model,
		Answers: answers,
		Usage: ClassifyUsage{
			InputTokens:  result.Usage.InputTokens,
			OutputTokens: result.Usage.OutputTokens,
		}}}, nil
}

// classifyFailed tells a caller whether waiting is worth it. A rate limit and an overloaded
// or unreachable provider come back on their own, so they are not reported as the
// request's fault; anything else is.
func classifyFailed(err error) error {
	switch {
	case errors.Is(err, lcm.ErrRateLimited):
		return huma.Error429TooManyRequests(err.Error())
	case errors.Is(err, lcm.ErrUnavailable):
		return huma.Error503ServiceUnavailable(err.Error())
	}
	return huma.Error400BadRequest(err.Error())
}

// lcmRequestOf builds the questions through lcm's constructors, which are what keep each
// type paired with the criteria it means. A choice with nothing to choose from or a score
// with nowhere to land is refused here, since a classifier would answer it with nonsense.
func lcmRequestOf(body *ClassifyRequest) (lcm.Request, error) {
	questions := make(map[string]lcm.Question, len(body.Questions))
	for id, question := range body.Questions {
		switch question.Type {
		case Noul:
			questions[id] = lcm.Noul(question.Instructions, value(question.Yes), value(question.No))
		case Choice:
			options := value(question.Options)
			if len(options) < 2 {
				return lcm.Request{}, fmt.Errorf("question %q is a choice with fewer than two options", id)
			}
			questions[id] = lcm.Choice(question.Instructions, options)
		case Score:
			levels := value(question.Levels)
			if len(levels) < 2 {
				return lcm.Request{}, fmt.Errorf("question %q is a score with fewer than two levels", id)
			}
			questions[id] = lcm.Score(question.Instructions, levels)
		default:
			return lcm.Request{}, fmt.Errorf("question %q asks for %q, which is not a question type", id, question.Type)
		}
	}
	return lcm.Request{State: body.State, Questions: questions}, nil
}

// classifyAnswerOf fills only the fields the question's type carries, so a noul does not
// come back with a confidence of zero that means nothing.
func classifyAnswerOf(asked lcm.QuestionType, answer lcm.Answer) ClassifyAnswer {
	switch asked {
	case lcm.TypeNoul:
		return ClassifyAnswer{Type: Noul, Yes: &answer.Yes}
	case lcm.TypeChoice:
		return ClassifyAnswer{
			Type:          Choice,
			Chosen:        &answer.Chosen,
			Probabilities: &answer.Probabilities,
			Confidence:    &answer.Confidence,
		}
	default:
		return ClassifyAnswer{
			Type:          Score,
			Level:         &answer.Level,
			Legend:        &answer.Legend,
			Probabilities: &answer.Probabilities,
			Confidence:    &answer.Confidence,
		}
	}
}

// registerClassify declares the operations served in classify.go.
func (s *Server) registerClassify(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "classify",
		Method:      http.MethodPost,
		Path:        "/v1/classify",
		Summary:     "Ask a classifier typed questions about a piece of text",
		Description: "The lcm modality, reachable on its own rather than only inside a guardrail. Every " +
			"question is put to the classifier at once and each comes back as a typed answer with " +
			"the distribution behind it: the probability a noul is true, which option of a choice " +
			"fits, where a score lands. There is no generated text, so there is nothing to stream: " +
			"routed, failed over and billed like search, one request one stat row.\n" +
			"Questions are answered independently and share the state's tokens between them, so ask " +
			"everything that might matter in one request. A question that comes back unanswered " +
			"fails the request rather than reading as a zero.\n" +
			"A target nobody routes is a 404. A provider that is rate limiting is a 429 and one that " +
			"is overloaded or cannot be reached is a 503; both are worth asking again after a wait, " +
			"and nothing else is.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The answers, under the ids they were asked under"},
			"429": errorResponse("The classifier is rate limiting. Ask again after a wait."),
			"503": errorResponse("The classifier is overloaded or could not be reached. Ask again after a wait."),
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound},
	}, s.classify)
}

type classifyRequest struct {
	Body *ClassifyRequest `required:"true"`
}

type classifyResponse struct {
	Body ClassifyResult
}

// ClassifyAnswer Which fields carry the answer depends on the type. A noul fills yes alone. A choice fills chosen, probabilities and confidence. A score fills level, legend, probabilities and confidence.
type ClassifyAnswer struct {
	Chosen        *string              `json:"chosen,omitempty" doc:"The likeliest option of a choice."`
	Confidence    *float64             `json:"confidence,omitempty" doc:"How peaked the distribution is, not whether acting on it is safe."`
	Legend        *map[string]string   `json:"legend,omitempty" doc:"A score's levels by index, as decimal strings."`
	Level         *float64             `json:"level,omitempty" doc:"Where a score landed, which may be between two of its levels."`
	Probabilities *map[string]float64  `json:"probabilities,omitempty" doc:"The distribution the answer came from: options for a choice, level indices for a score. They sum to one."`
	Type          ClassifyQuestionType `json:"type"`
	Yes           *float64             `json:"yes,omitempty" doc:"The probability a noul is true, from 0 to 1."`
}

func (*ClassifyAnswer) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Which fields carry the answer depends on the type. A noul fills yes alone. A choice fills chosen, probabilities and confidence. A score fills level, legend, probabilities and confidence."
	return schema
}

// ClassifyRequest is the ClassifyRequest schema.
type ClassifyRequest struct {
	Questions map[string]ClassifyQuestion `json:"questions" doc:"Keyed by ids of the caller's own choosing, which is how the answers come back. An id is not part of what is asked, so a question carries its whole meaning in its instructions." nullable:"false"`
	State     interface{}                 "json:\"state\" doc:\"What the questions are about: a string for plain text, or a JSON object whose parts a question can name, such as `message`.\" example:\"\\\"I was charged twice this month and nobody has answered my email.\\\"\""
	Tags      *map[string]string          `json:"tags,omitempty"`
	Target    *string                     `json:"target,omitempty" doc:"A provider/model or a capability shortcut. Empty takes classify-fast." example:"classify-fast"`
}

// ClassifyResult is the ClassifyResult schema.
type ClassifyResult struct {
	Answers  map[string]ClassifyAnswer `json:"answers" nullable:"false"`
	Model    string                    `json:"model" doc:"The version that answered, which is worth recording when the target was an alias."`
	Provider string                    `json:"provider"`
	Usage    ClassifyUsage             `json:"usage"`
}

// ClassifyUsage What the request read and wrote. The state's tokens are counted once however many questions shared them.
type ClassifyUsage struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
}

func (*ClassifyUsage) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What the request read and wrote. The state's tokens are counted once however many questions shared them."
	return schema
}
