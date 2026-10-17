package harness

import (
	"context"
	"fmt"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// Judge is a decision model: it answers named questions about a state with distributions,
// and writes no prose, so there is nothing to parse and nothing to stream.
type Judge interface {
	Classify(ctx context.Context, request decisionmodel.Request) (decisionmodel.Result, error)
	Close() error
}

// judgeNeutral is where a probability stops leaning one way.
const judgeNeutral = 0.5

// judgeHistory is how much of the conversation the judge is shown. Accuracy falls as the
// state fills with what the decision does not need, and who holds the floor now is settled
// by the last exchange.
const judgeHistory = 2

// judgeState is the turn with its parts named, so a question can point at one in backticks
// rather than describing it again.
func judgeState(turn FlowTurn) map[string]any {
	history := turn.History
	if len(history) > judgeHistory {
		history = history[len(history)-judgeHistory:]
	}
	conversation := make([]map[string]string, 0, len(history))
	for _, message := range history {
		speaker := "caller"
		if message.Role == llm.Assistant {
			speaker = "agent"
		}
		conversation = append(conversation, map[string]string{"speaker": speaker, "text": message.Content})
	}
	state := map[string]any{
		"agent_was_told":       turn.Instructions,
		"conversation":         conversation,
		"speaker":              turn.Participant,
		"heard":                turn.Text,
		"agent_speaking":       turn.Speaking,
		"speaker_has_finished": !turn.Unfinished,
		"different_voice":      turn.AnotherVoice,
	}
	if turn.Reply != "" {
		state["agent_has_said"] = turn.Reply
	}
	return state
}

// judgeQuestions are the facts the conversation's policy turns on, each a yes or no asked
// on its own. None asks what to do: judged decides that in code.
func judgeQuestions() map[string]decisionmodel.Question {
	return map[string]decisionmodel.Question{
		"addressed_to_agent": decisionmodel.Noul(
			"Were the words in `heard` meant for the agent described in `agent_was_told`?",
			"Spoken to the agent, whether or not they are finished.",
			"Spoken to somebody else in the room, to a pet or a child, read off a television, "+
				"or otherwise not meant for the agent."),
		"finished": decisionmodel.Noul(
			"Is `heard` a complete thought that now waits for the agent to reply?",
			"The speaker has said what they meant to say and expects an answer.",
			"The words stop part way through a sentence, a list, a number or a name, so "+
				"more is coming."),
		"still_growing": decisionmodel.Noul(
			"Does `heard` end part way through a number, an identifier or a time that the "+
				"speaker is still reading out?",
			"It ends mid-sequence, so more digits or words are still coming.",
			"Whatever number it contains is complete, or it contains none."),
		"recorded_menu": decisionmodel.Noul(
			"Is `heard` a recording reading out its options rather than a person talking?",
			"An automated menu, hold message or greeting.",
			"A person speaking, however stilted."),
		"non_speech": decisionmodel.Noul(
			"Is `heard` a noise rather than words: a cough, a sneeze, a laugh, a door, "+
				"static, or something else in the room?",
			"A noise, or a transcriber's description of one.",
			"Words the speaker meant to say, however short."),
		"ambiguous": decisionmodel.Noul(
			"Is what `heard` asks the agent to do impossible to act on without asking which "+
				"thing the speaker means?",
			"It points at something `conversation` does not pin down, such as \"the usual\", "+
				"\"change it\" or \"put it back\" with nothing saying what it is.",
			"What is wanted is clear from `heard` and `conversation`, or it asks for nothing."),
		"echo": decisionmodel.Noul(
			"Is `heard` the agent's own words from `agent_has_said` coming back, word for word "+
				"or nearly?",
			"The same words the agent just said, as a line echoing back.",
			"Words of the speaker's own."),
		"acknowledgement": decisionmodel.Noul(
			"Is `heard` only a brief sign that the speaker is listening, such as \"yeah\", "+
				"\"okay\" or \"right\"?",
			"A listening noise that asks for nothing.",
			"It says or asks something."),
		"adds_to_request": decisionmodel.Noul(
			"Does `heard` add to what the speaker asked for, without contradicting what the "+
				"agent is saying in `agent_has_said`?",
			"An addition such as \"and Saturday as well\" or \"and put us on the patio\".",
			"It corrects the agent, asks something unrelated, or adds nothing."),
		"objects": decisionmodel.Noul(
			"Does `heard` correct the agent, tell it to stop or wait, or ask it something new, "+
				"while it is saying `agent_has_said`?",
			"A correction such as \"no, make it six\", \"that's the wrong date\", \"wait\", "+
				"or a new question.",
			"It agrees, acknowledges, repeats the agent, or is not for the agent at all."),
	}
}

// judged is the conversation's policy over the facts. The order is the order of precedence:
// whether the words were speech, then whether they were for the agent, then what they ask.
func judged(turn FlowTurn, answers map[string]decisionmodel.Answer) (flowAnswer, error) {
	for id := range judgeQuestions() {
		if _, ok := answers[id]; !ok {
			return flowAnswer{}, fmt.Errorf("harness: the judge did not answer %q", id)
		}
	}
	yes := func(fact string) bool { return answers[fact].Yes >= judgeNeutral }

	if !turn.Speaking {
		switch {
		case yes("non_speech"), !yes("addressed_to_agent"):
			return flowAnswer{Disposition: Ignore, Floor: Continue}, nil
		case yes("recorded_menu"), yes("still_growing"), !yes("finished"):
			return flowAnswer{Disposition: Wait, Floor: Continue}, nil
		case yes("ambiguous"):
			return flowAnswer{Disposition: Clarify, Floor: Continue}, nil
		default:
			return flowAnswer{Disposition: Respond, Floor: Continue}, nil
		}
	}

	settled := Respond
	if turn.Unfinished {
		settled = Wait
	}
	switch {
	case yes("non_speech"):
		return flowAnswer{Disposition: Ignore, Floor: Continue}, nil
	case yes("echo"), yes("acknowledgement"):
		return flowAnswer{Disposition: settled, Floor: Continue}, nil
	case !yes("addressed_to_agent"):
		return flowAnswer{Disposition: Ignore, Floor: Continue}, nil
	case yes("adds_to_request"):
		return flowAnswer{Disposition: settled, Floor: Shorten}, nil
	case yes("objects"):
		return flowAnswer{Disposition: settled, Floor: Stop}, nil
	default:
		return flowAnswer{Disposition: settled, Floor: Continue}, nil
	}
}

// runJudge asks the decision model about one candidate and reports the ruling.
func (f *flow) runJudge(asked *candidate) {
	turn := asked.turn
	answered, err := f.judge.Classify(asked.ctx, decisionmodel.Request{
		State:     judgeState(turn),
		Questions: judgeQuestions(),
	})
	if f.forget(turn.ID) == nil {
		return
	}
	took := millis(time.Since(asked.askedAt))
	if f.yieldExpiredOverlap(asked) {
		return
	}
	if err != nil {
		f.logger.Error("flow judge failed", "error", err)
		f.emitter.Send(Decided{CandidateID: turn.ID, TookMs: took, Err: fmt.Errorf("harness: judge flow: %w", err)})
		return
	}
	answer, err := judged(turn, answered.Answers)
	if err != nil {
		f.logger.Error("flow judge failed", "error", err)
		f.emitter.Send(Decided{CandidateID: turn.ID, TookMs: took, Err: err})
		return
	}
	f.logger.Debug("flow judge answered", "candidate", turn.ID, "text", turn.Text,
		"speaking", turn.Speaking, "unfinished", turn.Unfinished,
		"disposition", answer.Disposition, "floor", answer.Floor, "took_ms", took)
	f.emitter.Send(Decided{
		CandidateID: turn.ID,
		Disposition: answer.Disposition,
		Floor:       answer.Floor,
		TookMs:      took,
	})
}
