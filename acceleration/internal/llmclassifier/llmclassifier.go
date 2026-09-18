// Package llmclassifier is what a routed classifier can be asked and what it answers.
//
// A classifier is not a language model with a narrow prompt. It is asked named questions
// about a piece of state and answers each with a typed value and the distribution behind
// it: the probability a condition holds, which of a set of options fits, where something
// falls on a described scale. There is no generated text, so there is nothing to stream
// and nothing to parse, and the number that comes back is the answer rather than a
// sentence about the answer.
//
// That is why this is a modality of its own rather than a corner of internal/llm. What a
// caller wants from it - a threshold it can act on - is not what a caller wants from a
// model that writes prose, and a request shaped for one is the wrong shape for the other.
package llmclassifier

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

// QuestionType is the shape of the answer a question asks for.
type QuestionType string

const (
	// TypeChoice picks one of a named set of options.
	TypeChoice QuestionType = "choice"
	// TypeScore places the state along ordered levels.
	TypeScore QuestionType = "score"
	// TypeNoul answers yes or no, as the probability of yes.
	TypeNoul QuestionType = "noul"
)

// Question is one judgement to make about the state.
//
// Build one with Choice, Score or Noul. Criteria means something different for each of the
// three, and the constructors are what keep the pairing honest.
type Question struct {
	Type         QuestionType
	Instructions string
	// Criteria is the possible answers: the options of a choice, the ordered levels of a
	// score, or what yes and no mean for a noul. Empty where a noul needs no gloss.
	Criteria any
}

// Choice asks which of options fits. Each option carries a description of what it covers,
// or an empty string where the name says it. Include something for "none of these"
// whenever the options may not cover an input, because a classifier can only answer with
// what it was given.
func Choice(instructions string, options map[string]string) Question {
	criteria := make(map[string]*string, len(options))
	for option, described := range options {
		if described == "" {
			criteria[option] = nil
			continue
		}
		criteria[option] = &described
	}
	return Question{Type: TypeChoice, Instructions: instructions, Criteria: criteria}
}

// Score asks where the state falls along levels, given in order. Each level has to
// describe a concrete situation and stand on its own, because the answer can land between
// two of them.
func Score(instructions string, levels []string) Question {
	return Question{Type: TypeScore, Instructions: instructions, Criteria: levels}
}

// Noul asks a yes or no question. Yes and no describe what each end means, and either may
// be empty when the question says it plainly enough.
func Noul(instructions, yes, no string) Question {
	question := Question{Type: TypeNoul, Instructions: instructions}
	if yes != "" || no != "" {
		question.Criteria = map[string]string{"true": yes, "false": no}
	}
	return question
}

// Answer is one judgement. Which fields carry it depends on Type: Choice fills Chosen,
// Probabilities and Confidence, Score fills Level, Legend, Probabilities and Confidence,
// and Noul fills Yes alone, because the probability is the answer and has no confidence
// beside it.
type Answer struct {
	Type QuestionType
	// Chosen is the likeliest option of a choice.
	Chosen string
	// Yes is the probability a noul is true, from 0 to 1.
	Yes float64
	// Level is where a score landed, which may be between two of the levels asked about.
	Level float64
	// Legend names a score's levels by their index, as decimal strings.
	Legend map[string]string
	// Probabilities is the distribution the answer came from: options for a choice, level
	// indices for a score. They sum to one.
	Probabilities map[string]float64
	// Confidence says how peaked that distribution is, not whether acting on it is safe.
	Confidence float64
}

// Usage is what one request cost.
type Usage struct {
	InputTokens  int64
	OutputTokens int64
}

// Request is a piece of state and everything worth asking about it.
//
// Every question is asked at once on purpose. They are answered independently and cannot
// see each other, so they cost the state's tokens once between them rather than once each.
// That makes asking a question whose answer may turn out to be irrelevant close to free,
// which is why a caller should ask everything it might need and let its own code decide
// what applies.
type Request struct {
	// State is what the questions are about: a string for plain text, or anything that
	// marshals to JSON for something with parts a question can name.
	State any
	// Questions are keyed by ids of the caller's own choosing, which is how the answers
	// come back. An id is for code and is not part of what is asked, so a question has to
	// carry its whole meaning in its instructions.
	Questions map[string]Question
}

// Validate reports what would make a request meaningless, so a provider does not have to.
func (r Request) Validate() error {
	if len(r.Questions) == 0 {
		return errors.New("llmclassifier: at least one question is required")
	}
	for id, question := range r.Questions {
		if strings.TrimSpace(question.Instructions) == "" {
			return fmt.Errorf("llmclassifier: question %q has no instructions", id)
		}
		switch question.Type {
		case TypeChoice, TypeScore, TypeNoul:
		default:
			return fmt.Errorf("llmclassifier: question %q asks for %q, which is not a question type",
				id, question.Type)
		}
	}
	return nil
}

// Result is the answers to one request, under the ids they were asked under.
type Result struct {
	// Model is the version that answered, which is worth recording when the request named
	// an alias.
	Model   string
	Answers map[string]Answer
	Usage   Usage
}

// Provider is one classifier this service can ask.
type Provider interface {
	// Classify puts every question to the provider at once and hands back the answers.
	Classify(ctx context.Context, request Request) (Result, error)
	Start(ctx context.Context) error
	Close() error
	// Provider is the stable provider name used in stats, e.g. "typesafe".
	Provider() string
	// Model is which of a provider's models is in use, e.g. "jev-latest".
	Model() string
}
