package api

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/lcm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/lcmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

// Classify puts a caller's questions about a piece of text to a routed classifier.
//
// It is search's shape for the same reason: one request and one set of answers, with
// nothing arriving in pieces, so there is no socket. Every question goes in one request,
// because they are answered independently and share the state's tokens between them.
func (s *Server) Classify(ctx context.Context, request ClassifyRequestObject) (ClassifyResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return Classify401JSONResponse{missingCustomer()}, nil
	}
	if s.streams == nil || s.streams.LCM == nil {
		return Classify404JSONResponse{NotFoundJSONResponse{Error: "this deployment does not route classification"}}, nil
	}
	if request.Body == nil {
		return Classify400JSONResponse{badRequest("a request body is required")}, nil
	}
	if text, ok := request.Body.State.(string); request.Body.State == nil || ok && strings.TrimSpace(text) == "" {
		return Classify400JSONResponse{badRequest("there is nothing to judge")}, nil
	}

	asked, err := lcmRequestOf(request.Body)
	if err != nil {
		return Classify400JSONResponse{badRequest(err.Error())}, nil
	}
	if err := asked.Validate(); err != nil {
		return Classify400JSONResponse{badRequest(err.Error())}, nil
	}
	tags := tagsSent(request.Body.Tags)
	if err := tags.Validate(); err != nil {
		return Classify400JSONResponse{badRequest(err.Error())}, nil
	}

	held := options.Classifier{Target: value(request.Body.Target)}
	if _, err := s.streams.LCM.Resolve(ctx, held.Route(), nil); err != nil {
		return Classify404JSONResponse{NotFoundJSONResponse{Error: err.Error()}}, nil
	}

	session, err := s.streams.LCM.Start(ctx, lcmrouter.Request{
		CustomerID: customerID,
		Tags:       tags,
		Options:    held,
	})
	if err != nil {
		return classifyFailed(err), nil
	}
	defer session.Close()

	result, err := session.Classify(ctx, asked)
	if err != nil {
		return classifyFailed(err), nil
	}

	answers := make(map[string]ClassifyAnswer, len(result.Answers))
	for id, answer := range result.Answers {
		answers[id] = classifyAnswerOf(asked.Questions[id].Type, answer)
	}
	model := session.Model()
	if result.Model != "" {
		model = result.Model
	}
	return Classify200JSONResponse{
		Provider: session.Provider(),
		Model:    model,
		Answers:  answers,
		Usage: ClassifyUsage{
			InputTokens:  result.Usage.InputTokens,
			OutputTokens: result.Usage.OutputTokens,
		},
	}, nil
}

// classifyFailed tells a caller whether waiting is worth it. A rate limit and an overloaded
// or unreachable provider come back on their own, so they are not reported as the
// request's fault; anything else is.
func classifyFailed(err error) ClassifyResponseObject {
	switch {
	case errors.Is(err, lcm.ErrRateLimited):
		return Classify429JSONResponse{Error: err.Error()}
	case errors.Is(err, lcm.ErrUnavailable):
		return Classify503JSONResponse{Error: err.Error()}
	}
	return Classify400JSONResponse{badRequest(err.Error())}
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
