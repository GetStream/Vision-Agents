package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// suiteVersion names the task set. A change to the tasks, their checks or the token budget
// is a new version, so the inventory never averages two different tests.
const suiteVersion = "v2"

// maxOutputTokens leaves room for the longest thinking the suite has seen (about 10k
// tokens at Gemini's xhigh) without letting a runaway answer cost dollars.
const maxOutputTokens = 16384

const callTimeout = 3 * time.Minute

// kind groups tasks by what they say about a model, which is how the report reads them.
type kind string

const (
	kindVoice     kind = "voice"
	kindTool      kind = "tool"
	kindJSON      kind = "json"
	kindReasoning kind = "reasoning"
	kindImage     kind = "image"
)

type task struct {
	name   string
	kind   kind
	params llm.ResponseParams
	check  func(llm.Response) bool
}

var weather = llm.Tool{
	Name:        "get_weather",
	Description: "Look up the current weather in a city",
	Parameters: map[string]any{
		"type":       "object",
		"properties": map[string]any{"city": map[string]any{"type": "string"}},
		"required":   []string{"city"},
	},
}

func user(text string) []llm.Message { return []llm.Message{{Role: llm.User, Content: text}} }

func answers(want string) func(llm.Response) bool {
	return func(r llm.Response) bool {
		return strings.Contains(strings.ReplaceAll(r.OutputText, ",", ""), "ANSWER: "+want)
	}
}

var tasks = []task{
	{"voice_reply", kindVoice, llm.ResponseParams{
		Instructions: "You are a friendly voice assistant on a phone call. Answer in one short spoken sentence.",
		Input:        user("Hi! What's a good month to visit Lisbon?"),
	}, func(r llm.Response) bool {
		text := strings.TrimSpace(r.OutputText)
		return text != "" && len(text) < 300
	}},
	{"tool_call", kindTool, llm.ResponseParams{
		Instructions: "Use tools when they help.",
		Input:        user("Do I need an umbrella in Paris right now?"),
		Tools:        []llm.Tool{weather},
	}, func(r llm.Response) bool {
		return len(r.ToolCalls) == 1 && r.ToolCalls[0].Name == "get_weather" &&
			strings.Contains(r.ToolCalls[0].Arguments, "Paris")
	}},
	{"json_extract", kindJSON, llm.ResponseParams{
		Instructions: "Reply with only a JSON object with keys name (string), date (YYYY-MM-DD) and party_size (integer).",
		Input:        user("Hey, it's Maria Rossi, I'd like a table for four on the 3rd of March 2027, around eight."),
	}, func(r llm.Response) bool {
		var v struct {
			Name      string `json:"name"`
			Date      string `json:"date"`
			PartySize int    `json:"party_size"`
		}
		return json.Unmarshal([]byte(llm.Unfence(r.OutputText)), &v) == nil &&
			strings.Contains(v.Name, "Rossi") && v.Date == "2027-03-03" && v.PartySize == 4
	}},
	{"math_word", kindReasoning, llm.ResponseParams{
		Instructions: "Solve it. End your answer with a line 'ANSWER: <number>'.",
		Input:        user("A shop sells pens at 3 for $4 and notebooks at $2.50 each. Ana buys 12 pens and 7 notebooks and pays with a $50 note. How many cents of change does she get?"),
	}, answers("1650")},
	{"digit_sum", kindReasoning, llm.ResponseParams{
		Instructions: "End your answer with a line 'ANSWER: <number>'.",
		Input:        user("What is the sum of the decimal digits of 2^50?"),
	}, answers("76")},
	{"number_theory", kindReasoning, llm.ResponseParams{
		Instructions: "End your answer with a line 'ANSWER: <number>'.",
		Input:        user("What is the smallest positive integer n such that the decimal representation of n^2 ends in the digits 444?"),
	}, answers("38")},
	{"image_colour", kindImage, llm.ResponseParams{
		Input: []llm.Message{{Role: llm.User, Parts: []llm.ContentPart{
			{Text: "What colour fills this whole image? Answer with one word."},
			{Image: &llm.ImagePart{MIME: "image/png", Data: redSquare()}},
		}}},
	}, func(r llm.Response) bool { return strings.Contains(strings.ToLower(r.OutputText), "red") }},
}

func redSquare() []byte {
	picture := image.NewRGBA(image.Rect(0, 0, 64, 64))
	for x := range 64 {
		for y := range 64 {
			picture.Set(x, y, color.RGBA{R: 220, A: 255})
		}
	}
	var out bytes.Buffer
	_ = png.Encode(&out, picture)
	return out.Bytes()
}

// tasksFor leaves out what the model cannot be asked: tools it would ignore, images it
// cannot read.
func tasksFor(s subject) []task {
	var chosen []task
	for _, t := range tasks {
		if t.kind == kindTool && s.config.IgnoresTools {
			continue
		}
		if t.kind == kindImage && !slices.Contains(s.config.InputModalities, "image") {
			continue
		}
		chosen = append(chosen, t)
	}
	return chosen
}

// call is one request and what came of it.
type call struct {
	subject   int
	kind      kind
	ttftMs    int64
	totalMs   int64
	input     int64
	cached    int64
	output    int64
	reasoning int64
	correct   bool
	err       string
}

func ask(s subject, t task) call {
	result := call{kind: t.kind}
	ctx, cancel := context.WithTimeout(context.Background(), callTimeout)
	defer cancel()
	params := t.params
	params.MaxOutputTokens = maxOutputTokens
	if s.asked != "" {
		params.Reasoning.Effort = s.asked
	}
	start := time.Now()
	stream, err := s.llm.Create(ctx, params)
	if err != nil {
		result.err = err.Error()
		return result
	}
	defer stream.Close()
	for stream.Next() {
		switch stream.Current().(type) {
		case llm.OutputTextDelta, llm.FunctionCallArgumentsDelta:
			if result.ttftMs == 0 {
				result.ttftMs = time.Since(start).Milliseconds()
			}
		}
	}
	result.totalMs = time.Since(start).Milliseconds()
	if err := stream.Err(); err != nil {
		result.err = err.Error()
		return result
	}
	if result.ttftMs == 0 {
		result.ttftMs = result.totalMs
	}
	response := stream.Response()
	usage := response.Usage
	result.input = usage.InputTokens
	result.cached = usage.InputTokensDetails.CachedTokens
	result.output = usage.OutputTokens
	result.reasoning = usage.OutputTokensDetails.ReasoningTokens
	result.correct = t.check(response)
	return result
}

// run interleaves every model's calls so no one vendor sees the whole concurrency at once.
func run(subjects []subject, reps, concurrency int) []measurement {
	type job struct {
		subject int
		task    task
	}
	var jobs []job
	for range reps {
		for _, t := range tasks {
			for i, s := range subjects {
				if slices.ContainsFunc(tasksFor(s), func(c task) bool { return c.name == t.name }) {
					jobs = append(jobs, job{i, t})
				}
			}
		}
	}

	queue := make(chan job)
	calls := make([]call, 0, len(jobs))
	var mu sync.Mutex
	var wg sync.WaitGroup
	for range concurrency {
		wg.Go(func() {
			for j := range queue {
				c := ask(subjects[j.subject], j.task)
				c.subject = j.subject
				mu.Lock()
				calls = append(calls, c)
				done := len(calls)
				mu.Unlock()
				status := "ok"
				if c.err != "" {
					status = "error: " + firstLine(c.err)
				} else if !c.correct {
					status = "wrong"
				}
				s := subjects[j.subject]
				fmt.Fprintf(os.Stderr, "[%d/%d] %s@%s %s %dms %s\n",
					done, len(jobs), s.config.Name(), s.effort, j.task.name, c.ttftMs, status)
			}
		})
	}
	for _, j := range jobs {
		queue <- j
	}
	close(queue)
	wg.Wait()

	measurements := make([]measurement, len(subjects))
	for i, s := range subjects {
		var own []call
		for _, c := range calls {
			if c.subject == i {
				own = append(own, c)
			}
		}
		measurements[i] = summarise(s, own)
	}
	return measurements
}

func firstLine(s string) string {
	line, _, _ := strings.Cut(s, "\n")
	if len(line) > 160 {
		line = line[:160]
	}
	return line
}
