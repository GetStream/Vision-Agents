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

// judgeAmbiguous is higher than neutral because a clarifying question costs the caller a turn,
// and Jev puts plain requests such as "move my Thursday appointment to Tuesday" at 0.5 to 0.6.
const judgeAmbiguous = 0.7

// judgeFinished is higher than neutral because answering a caller part way through talks over
// the rest, and Jev puts a pause after a name or a date of birth at 0.5 to 0.7. judgeGrowing
// is lower for the same reason: a number Jev half suspects is unfinished usually is.
const (
	judgeFinished = 0.7
	judgeGrowing  = 0.3
)

// judgeAddressed is higher than neutral because asking whether any part of the words is for
// the agent puts side talk such as "I'll call you back, Mum" at 0.5 to 0.55.
const judgeAddressed = 0.6

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
			"Is any part of `heard` meant for the agent described in `agent_was_told`?",
			"Some or all of it is spoken to the agent, whether or not it is finished, even if "+
				"the speaker also turned aside to somebody else in the same words.",
			"All of it is spoken to somebody else in the room, to a pet or a child, read off a "+
				"television, or otherwise not meant for the agent."),
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
		"listing": decisionmodel.Noul(
			"Has the speaker said, earlier in `conversation` or in `heard` itself, that they will "+
				"give several details in one go, and is `heard` that promise or one more of those "+
				"details, rather than a question for the agent or a plain sign they have finished?",
			"They promised several details and are still going: the promise itself, a name, "+
				"a date, a booking, a preference.",
			"No such promise was made, or `heard` asks the agent something, or says they are done."),
		"greeting_only": decisionmodel.Noul(
			"Is `heard` only a greeting, such as \"hi\" or \"hello there\", with nothing asked "+
				"or said yet?",
			"A greeting and nothing more, so what they called about is still coming.",
			"It says or asks something beyond a greeting."),
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
			"An addition such as \"and the other one too\" or \"also send it by email\".",
			"It corrects the agent, asks something unrelated, or adds nothing."),
		"objects": decisionmodel.Noul(
			"Does `heard` correct the agent, tell it to stop or wait, or ask it something new, "+
				"while it is saying `agent_has_said`?",
			"A correction such as \"no, the other one\", \"that's not what I said\", \"wait\", "+
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
		case yes("recorded_menu"):
			return flowAnswer{Disposition: Wait, Floor: Continue}, nil
		case yes("non_speech"), answers["addressed_to_agent"].Yes < judgeAddressed:
			return flowAnswer{Disposition: Ignore, Floor: Continue}, nil
		case answers["still_growing"].Yes >= judgeGrowing, yes("listing"), yes("greeting_only"),
			answers["finished"].Yes < judgeFinished:
			return flowAnswer{Disposition: Wait, Floor: Continue}, nil
		case !turn.Waited && !turn.Ended:
			// The cadence asks after 350ms of silence and Flux marks the end of a turn about
			// 570ms after the last word, so answering on the silence alone answered callers
			// in the middle of what they were asking.
			return flowAnswer{Disposition: Wait, Floor: Continue}, nil
		case answers["ambiguous"].Yes >= judgeAmbiguous:
			return flowAnswer{Disposition: Clarify, Floor: Continue}, nil
		default:
			return flowAnswer{Disposition: Respond, Floor: Continue}, nil
		}
	}

	settled := Respond
	if turn.Unfinished {
		settled = Wait
	}
	// Words said over the agent settle on the same short pause, and "actually, no" answered
	// on its own had the agent reply "No."
	thought := settled
	if answers["finished"].Yes < judgeFinished || answers["still_growing"].Yes >= judgeGrowing {
		thought = Wait
	}
	switch {
	case yes("non_speech"):
		return flowAnswer{Disposition: Ignore, Floor: Continue}, nil
	case yes("echo"), yes("acknowledgement"):
		return flowAnswer{Disposition: settled, Floor: Continue}, nil
	case answers["addressed_to_agent"].Yes < judgeAddressed:
		return flowAnswer{Disposition: Ignore, Floor: Continue}, nil
	case yes("adds_to_request"):
		return flowAnswer{Disposition: thought, Floor: Shorten}, nil
	case yes("objects"):
		return flowAnswer{Disposition: thought, Floor: Stop}, nil
	default:
		return flowAnswer{Disposition: thought, Floor: Continue}, nil
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
		"disposition", answer.Disposition, "floor", answer.Floor, "took_ms", took,
		"finished", answered.Answers["finished"].Yes, "addressed", answered.Answers["addressed_to_agent"].Yes,
		"growing", answered.Answers["still_growing"].Yes, "listing", answered.Answers["listing"].Yes)
	f.emitter.Send(Decided{
		CandidateID: turn.ID,
		Disposition: answer.Disposition,
		Floor:       answer.Floor,
		TookMs:      took,
		Ruler:       Primary,
	})
}
