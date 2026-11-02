package api

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const (
	// notGoodScore is the highest review score a call that went badly can have: a 3 is a
	// caller who was passed on or told to call back, neither helped nor harmed.
	notGoodScore = 3
	// analysedCalls is how many of those calls one analysis reads, newest first. A
	// hundred reviews are a few thousand words, which one model call reads whole.
	analysedCalls = 100
	// analysisTarget is a quality tier rather than a fast one, as the review itself is:
	// somebody is waiting, but for a judgement, not for a turn.
	analysisTarget   = "multilingual-high-accuracy"
	analysisTokens   = 3000
	analysisDeadline = 2 * time.Minute
	// maxCauses keeps the answer to what a person would act on. The instructions ask for
	// as many.
	maxCauses = 6
)

// analysisInstructions is what the analyst is told to be.
const analysisInstructions = `You are reading the reviews of calls an automated voice agent handled badly. Each
review was written right after its call, from the transcript: a summary of what happened,
a score from 1 (wrong or unhelpful) to 3 (neither helped nor harmed), and notes on what to act on.

Find what caused them. Group the calls by cause, at most 6 causes, the one behind the most
calls first. A cause is something the agent's builder can change: a gap in what the agent
knows, a tool that failed or was missing, instructions it misread, a handover that came too
late, speech it misheard. A call can have more than one cause. Leave out a cause only one
call shows unless it is serious.

For each cause write a short title, an explanation of what went wrong in those calls, the fix
the builder should make, and the ids of the calls it explains, exactly as given. Then sum up
the whole in two sentences.

Answer with JSON only:
{"summary": "...", "causes": [{"title": "...", "explanation": "...", "fix": "...", "call_ids": ["..."]}]}`

// analyzeReviews reads the latest badly reviewed calls and asks a model what caused them.
//
// It is answered inline and nothing is kept: it costs a model call, so it runs when somebody
// asks rather than on every read, and an answer kept would be stale as soon as the next call
// is reviewed.
func (s *Server) analyzeReviews(ctx context.Context, request *analyzeReviewsRequest) (*analyzeReviewsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoCalls
	}
	if s.streams == nil || s.streams.LLM == nil {
		return nil, notConfigured("reviews cannot be analysed: this deployment routes no language model")
	}
	if request.Body == nil {
		return nil, invalidRequest("a request body is required")
	}
	if !request.Body.To.After(request.Body.From) {
		return nil, invalidRequest("to must be after from")
	}

	calls, err := s.store.CustomerCalls(ctx, customerID, store.CallFilter{
		From:           request.Body.From,
		To:             request.Body.To,
		MaxReviewScore: notGoodScore,
		Limit:          analysedCalls,
	})
	if err != nil {
		return nil, err
	}
	analysis := ReviewAnalysis{CallsRead: len(calls), Causes: []ReviewCause{}}
	if len(calls) == 0 {
		return &analyzeReviewsResponse{Body: analysis}, nil
	}

	ctx, cancel := context.WithTimeout(ctx, analysisDeadline)
	defer cancel()
	target := value(request.Body.Target)
	if target == "" {
		target = analysisTarget
	}
	session, err := s.streams.LLM.Start(ctx, llmrouter.Request{
		CustomerID: customerID,
		Target:     target,
	})
	if err != nil {
		return nil, unavailable("the reviews could not be analysed: " + err.Error())
	}
	defer session.Close()

	stream, err := session.Create(ctx, llm.ResponseParams{
		ID:              "review-analysis",
		Instructions:    analysisInstructions,
		Input:           []llm.Message{{Role: llm.User, Content: reviewsRead(calls)}},
		MaxOutputTokens: analysisTokens + llm.ThinkingRoom,
		Text:            llm.TextParams{Format: llm.FormatJSONObject},
	})
	if err != nil {
		return nil, unavailable("the reviews could not be analysed: " + err.Error())
	}
	response, err := llm.Collect(stream)
	if err != nil {
		return nil, unavailable("the reviews could not be analysed: " + err.Error())
	}

	var answer struct {
		Summary string `json:"summary"`
		Causes  []struct {
			Title       string   `json:"title"`
			Explanation string   `json:"explanation"`
			Fix         string   `json:"fix"`
			CallIDs     []string `json:"call_ids"`
		} `json:"causes"`
	}
	if err := json.Unmarshal([]byte(llm.Unfence(response.OutputText)), &answer); err != nil {
		return nil, unavailable("the reviews could not be analysed: the model's answer was not JSON")
	}

	read := make(map[string]bool, len(calls))
	for _, call := range calls {
		read[call.ID] = true
	}
	analysis.Summary = strings.TrimSpace(answer.Summary)
	for _, cause := range answer.Causes {
		// An id the model made up explains nothing, and a cause left explaining nothing
		// is one it made up.
		var ids []string
		for _, id := range cause.CallIDs {
			if read[id] && !slices.Contains(ids, id) {
				ids = append(ids, id)
			}
		}
		if len(ids) == 0 || strings.TrimSpace(cause.Title) == "" {
			continue
		}
		analysis.Causes = append(analysis.Causes, ReviewCause{
			Title:       strings.TrimSpace(cause.Title),
			Explanation: strings.TrimSpace(cause.Explanation),
			Fix:         strings.TrimSpace(cause.Fix),
			CallIds:     ids,
		})
	}
	slices.SortStableFunc(analysis.Causes, func(a, b ReviewCause) int {
		return cmp.Compare(len(b.CallIds), len(a.CallIds))
	})
	if len(analysis.Causes) > maxCauses {
		analysis.Causes = analysis.Causes[:maxCauses]
	}
	return &analyzeReviewsResponse{Body: analysis}, nil
}

// reviewsRead is the reviews as the analyst reads them, one block a call.
func reviewsRead(calls []store.Call) string {
	var written strings.Builder
	written.WriteString(strconv.Itoa(len(calls)) + " reviews, newest first:\n")
	for _, call := range calls {
		fmt.Fprintf(&written, "\nCall %s", call.ID)
		if call.ConfigID != "" {
			fmt.Fprintf(&written, " (agent %s)", call.ConfigID)
		}
		if call.ReviewScore != nil {
			fmt.Fprintf(&written, ", score %d", *call.ReviewScore)
		}
		fmt.Fprintf(&written, "\nSummary: %s\nNotes: %s\n", call.Summary, call.ReviewNotes)
	}
	return written.String()
}

// registerReviews declares the operations served in reviews.go.
func (s *Server) registerReviews(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "analyzeReviews",
		Method:      http.MethodPost,
		Path:        "/v1/agents/calls/reviews/analysis",
		Summary:     "What caused the calls the agents handled badly",
		Description: "Reads the latest " + strconv.Itoa(analysedCalls) + " calls in the window whose review " +
			"scored them " + strconv.Itoa(notGoodScore) + " or lower, and asks a model to group them " +
			"by cause, each with the fix to make and the calls it explains.\n" +
			"Costs one language model call, billed as any other, and keeps nothing: ask again for " +
			"a fresh answer.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The causes, the one behind the most calls first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusServiceUnavailable},
	}, s.analyzeReviews)
}

type analyzeReviewsRequest struct {
	Body *AnalyzeReviewsRequest `required:"true"`
}

type analyzeReviewsResponse struct {
	Body ReviewAnalysis
}

// AnalyzeReviewsRequest is the window the reviews are read from.
type AnalyzeReviewsRequest struct {
	From   time.Time `json:"from" doc:"Only calls that started at or after this, inclusive."`
	To     time.Time `json:"to" doc:"Only calls that started before this, exclusive."`
	Target *string   `json:"target,omitempty" doc:"The model that analyses, a \"provider/model\" name or a shortcut. Left out, multilingual-high-accuracy."`
}

// ReviewAnalysis is what caused the calls the agents handled badly.
type ReviewAnalysis struct {
	CallsRead int           `json:"calls_read" doc:"How many badly reviewed calls were read. Zero means there were none, and nothing was asked."`
	Summary   string        `json:"summary" doc:"The whole, in two sentences."`
	Causes    []ReviewCause `json:"causes" nullable:"false"`
}

// ReviewCause is one thing that went wrong across several calls.
type ReviewCause struct {
	Title       string   `json:"title"`
	Explanation string   `json:"explanation" doc:"What went wrong in those calls."`
	Fix         string   `json:"fix" doc:"What to change in the agent."`
	CallIds     []string `json:"call_ids" nullable:"false" doc:"The calls it explains, which are the sessions that ran them. Only ids that were read."`
}
