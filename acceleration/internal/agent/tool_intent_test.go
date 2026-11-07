package agent

import (
	"fmt"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audiofloor"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// sure and unsure are how likely chorus-asr makes the tool a caller asks for: past both bars,
// and short of them.
const (
	sure   = 0.9995
	unsure = 0.9
)

// heardAskingFor has chorus-asr's check of the caller's pause end their turn on words, ruling
// that they ask for tool, with probability p, in language, and the utterance's final follow.
func (s *AgentSuite) heardAskingFor(participant stt.Participant, utterance int64, words, tool string, p float64, language string) {
	pEOT := 0.9999
	pauseAsker{s.agent}.Ruled(participant, audiofloor.Ruled{
		Check: audiofloor.Check{Ended: true, Ruling: harness.Decided{Disposition: harness.Respond, Floor: harness.Continue},
			Answer: audiofloor.Answer{Answer: tool, AnswerProbability: p, PEOT: &pEOT,
				Ruling:  &audiofloor.Ruling{Disposition: string(harness.Respond), Floor: string(harness.Continue), Tool: tool},
				Options: []audiofloor.Option{{Name: "respond", P: 1 - p - 0.0001}, {Name: "wait", P: 0.0001}, {Name: tool, P: p}}}},
		Words: words, Language: language, Utterance: utterance,
	})
	s.ears.emitter.Send(stt.Transcript{Participant: participant, Mode: stt.ModeFinal, Utterance: utterance, Text: words,
		Provider: audiofloor.ProviderName, Language: "en", Confidence: 1})
}

// replyNote is the note the reply to the turn asked at index i was sent with, empty for none.
func (s *AgentSuite) replyNote(i int) string {
	requests := s.model.turns()
	s.Require().Greater(len(requests), i, "the reply was never asked for")
	var notes []string
	for _, message := range requests[i].Input {
		if message.Role == llm.System {
			notes = append(notes, message.Content)
		}
	}
	return strings.Join(notes, "\n")
}

// spokenTexts is what the voice was given, in order.
func (s *AgentSuite) spokenTexts() []string {
	var texts []string
	for _, request := range s.voice.spoken() {
		if request.Text != "" {
			texts = append(texts, request.Text)
		}
	}
	return texts
}

func (s *AgentSuite) TestATurnChorusASRIsSureAsksForAToolIsAcknowledgedBeforeTheReply() {
	s.actsOnChorusASR()
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"I have order twelve for you."}
	s.model.then = []string{"It ships tomorrow."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.heardAskingFor(participant, 1, "Where is my order twelve?", "lookup_order", sure, "English")

	s.eventually(func() bool { return s.spokenText("ships tomorrow") }, "the answer never came")
	spoken := s.spokenTexts()
	s.Require().NotEmpty(spoken)
	s.Equal("One moment.", spoken[0], "the acknowledgment is the first thing the caller hears, ahead of the reply")
	s.Contains(spoken, "I have order twelve for you.", "the reply still says what it was going to")
	s.Contains(s.replyNote(0), `"One moment."`, "the reply is told what was said for it, so it says no hold phrase of its own")
	s.True(slices.ContainsFunc(s.history(), func(m llm.Message) bool {
		return m.Role == llm.Assistant && m.Content == "One moment. I have order twelve for you."
	}), "the acknowledgment is part of the reply the conversation keeps: %v", s.history())
}

func (s *AgentSuite) TestATurnChorusASRIsUnsureAsksForAToolIsNotAcknowledged() {
	s.actsOnChorusASR()
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"I have order twelve for you."}
	s.model.then = []string{"It ships tomorrow."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.heardAskingFor(participant, 1, "Where is my order twelve?", "lookup_order", unsure, "English")

	s.eventually(func() bool { return s.spokenText("ships tomorrow") }, "the answer never came")
	s.Equal("I have order twelve for you.", s.spokenTexts()[0], "below the bar the reply speaks first, as it always has")
	s.NotContains(s.replyNote(0), "hold phrase", "the reply was told of an acknowledgment nobody said")
}

func (s *AgentSuite) TestAnAcknowledgedTurnIsGivenNoFillerOfItsOwn() {
	// The acknowledgment is the wait's hold phrase. A reply that reaches for the tool without
	// a word would otherwise be given a line the model writes, a second hold phrase.
	s.actsOnChorusASR()
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{}
	s.model.then = []string{"It ships tomorrow."}
	s.model.composes = []string{"Pulling up order twelve now."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.heardAskingFor(participant, 1, "Where is my order twelve?", "lookup_order", sure, "English")

	s.eventually(func() bool { return s.spokenText("ships tomorrow") }, "the answer never came")
	s.Empty(s.model.composeRequests(), "the agent wrote a hold line on top of the acknowledgment")
	s.Equal([]string{"One moment.", "It ships tomorrow."}, s.spokenTexts(), "one hold phrase to the wait")
}

func (s *AgentSuite) TestAnAcknowledgedTurnIsGivenNoPreSpeechEither() {
	s.actsOnChorusASR()
	s.ownsTools("order 12 ships tomorrow")
	s.toolPolicy = func(string) ToolPolicy { return ToolPolicy{PreSpeech: "Let me pull up your order."} }
	s.join(false)
	s.model.reply = []string{}
	s.model.then = []string{"It ships tomorrow."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.heardAskingFor(participant, 1, "Where is my order twelve?", "lookup_order", sure, "English")

	s.eventually(func() bool { return s.spokenText("ships tomorrow") }, "the answer never came")
	s.False(s.spokenText("Let me pull up your order."), "the binding's phrase was said on top of the acknowledgment")
}

func (s *AgentSuite) TestAnAcknowledgmentIsNotSaidOverACallerWhoTalksOn() {
	// The caller goes on before the reply's request is accepted: the acknowledgment would be
	// said over them.
	s.actsOnChorusASR()
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	hold := make(chan struct{})
	s.model.holdCreate = hold
	s.model.reply = []string{"I have order twelve for you."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.heardAskingFor(participant, 1, "Where is my order twelve?", "lookup_order", sure, "English")
	s.eventually(func() bool { return len(s.model.turns()) == 1 }, "the reply was never asked for")
	s.mutters(participant, "and also my other")
	s.eventually(func() bool { return s.agent.converse.Listening() }, "the caller was not heard talking on")
	close(hold)

	s.eventually(func() bool {
		return countOf[Responded](s.reported()) > 0 || countOf[Interrupted](s.reported()) > 0
	}, "the reply neither finished nor was dropped")
	s.NotContains(s.spokenTexts(), "One moment.", "the acknowledgment was said over the caller")
}

func (s *AgentSuite) TestAnAcknowledgmentIsNotSaidToACallerHeardInAnUtteranceSinceTheTurnEnded() {
	// The caller is voiced again, with no words yet, before the reply's request is accepted:
	// the reply stands until their words come, but an acknowledgment now would talk over them.
	s.actsOnChorusASR()
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	hold := make(chan struct{})
	s.model.holdCreate = hold
	s.model.reply = []string{"I have order twelve for you."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.heardAskingFor(participant, 1, "Where is my order twelve?", "lookup_order", sure, "English")
	s.eventually(func() bool { return len(s.model.turns()) == 1 }, "the reply was never asked for")
	pauseAsker{s.agent}.Hearing(participant)
	close(hold)

	s.eventually(func() bool { return s.spokenText("I have order twelve for you.") }, "the reply was not said")
	s.NotContains(s.spokenTexts(), "One moment.", "the acknowledgment was said over the caller")
}

func (s *AgentSuite) TestAnAcknowledgmentIsDroppedWithAReplyTheCallerWentOnBeforeHearing() {
	// The caller adds to their turn before the reply's request is accepted, and the reply is
	// dropped for one that answers all of it. Its acknowledgment goes with it.
	s.actsOnChorusASR()
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	hold := make(chan struct{})
	s.model.holdCreate = hold
	s.model.reply = []string{"I have order twelve for you."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.heardAskingFor(participant, 1, "Where is my order twelve?", "lookup_order", sure, "English")
	s.eventually(func() bool { return len(s.model.turns()) == 1 }, "the reply was never asked for")
	first := s.model.turns()[0].ID
	s.ears.emitter.Send(stt.Transcript{Participant: participant, Mode: stt.ModeFinal, Utterance: 2, Text: "Actually, never mind.",
		Provider: audiofloor.ProviderName, Language: "en", Confidence: 1})
	s.eventually(func() bool { return len(s.model.turns()) == 2 }, "what the caller went on with was not answered")
	s.True(s.model.abandoned(first) || !s.agent.speaking(first) || s.model.turns()[1].ID == first,
		"the reply the caller went on over was kept")
	close(hold)

	s.eventually(func() bool { return countOf[Responded](s.reported()) > 0 }, "the second reply never finished")
	s.NotContains(s.spokenTexts(), "One moment.", "the dropped reply's acknowledgment was said")
}

func (s *AgentSuite) TestAReplyThatFailsAfterItsAcknowledgmentStillSaysTheAnswerIsLost() {
	// "One moment." and then nothing is the answer lost, as silence is.
	s.actsOnChorusASR()
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = nil
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.heardAskingFor(participant, 1, "Where is my order twelve?", "lookup_order", sure, "English")
	s.eventually(func() bool { return s.spokenText("One moment.") }, "the acknowledgment was not said")
	s.model.opened(s.model.turns()[0].ID).Fail(fmt.Errorf("upstream went away"), "llm")

	s.eventually(func() bool { return s.spokenText(lostReply) }, "the caller was left waiting after the acknowledgment")
}

func (s *AgentSuite) TestTheAcknowledgmentIsInTheLanguageTheCallerWasHeardIn() {
	s.actsOnChorusASR()
	s.ownsTools("el pedido 12 llega mañana")
	s.join(false)
	s.model.reply = []string{"Tengo el pedido doce."}
	s.model.then = []string{"Llega mañana."}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.heardAskingFor(participant, 1, "¿Dónde está mi pedido doce?", "lookup_order", sure, "Spanish")

	s.eventually(func() bool { return s.spokenText("Llega mañana.") }, "the answer never came")
	s.Equal("Un momento.", s.spokenTexts()[0])
}

func (s *AgentSuite) TestNoAcknowledgmentIsSaidInALanguageItHasNoWordsFor() {
	s.actsOnChorusASR()
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"注文番号十二ですね。"}
	s.model.then = []string{"明日発送します。"}
	s.asksFor("lookup_order", `{"order":"12"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.heardAskingFor(participant, 1, "注文十二はどこですか", "lookup_order", sure, "Japanese")

	s.eventually(func() bool { return s.spokenText("明日発送します。") }, "the answer never came")
	s.Equal("注文番号十二ですね。", s.spokenTexts()[0], "an acknowledgment in the wrong language was said")
}

func (s *AgentSuite) TestTheModelsWordsAfterAnAcknowledgmentAreWhereTheMessageHasThem() {
	// What the caller is counted as having heard of a reply is reckoned from where each
	// sentence is in its message, which now opens with the acknowledgment.
	s.actsOnChorusASR()
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"I have order twelve for you. ", "It is a big one."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.heardAskingFor(participant, 1, "Where is my order twelve?", "lookup_order", sure, "English")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 && countOf[Spoke](s.reported()) == 3 }, "the reply was not all said")
	responded, _ := firstOf[Responded](s.reported())
	s.Equal("One moment. I have order twelve for you. It is a big one.", responded.Text)
	runes := []rune(responded.Text)
	var said []string
	for _, event := range s.reported() {
		if spoke, ok := event.(Spoke); ok {
			said = append(said, strings.TrimSpace(string(runes[spoke.Message.Start:spoke.Message.End])))
		}
	}
	s.Equal([]string{"One moment.", "I have order twelve for you.", "It is a big one."}, said)
}

func (s *AgentSuite) TestEachAcknowledgmentIsNotTheSameWordsAsTheLast() {
	s.actsOnChorusASR()
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"Sure."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.heardAskingFor(participant, 1, "Where is my order twelve?", "lookup_order", sure, "English")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the first reply never finished")
	s.speak(participant)
	s.heardAskingFor(participant, 2, "And where is order thirteen?", "lookup_order", sure, "English")
	s.eventually(func() bool { return countOf[Responded](s.reported()) == 2 }, "the second reply never finished")

	s.Equal([]string{"One moment.", "Sure.", "Just a moment.", "Sure."}, s.spokenTexts())
}

func (s *AgentSuite) TestAnAgentThatDoesNotActOnChorusASRSaysNoAcknowledgment() {
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"I have order twelve for you."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "Where is my order twelve?")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")
	s.Equal([]string{"I have order twelve for you."}, s.spokenTexts())
	s.NotContains(s.replyNote(0), "hold phrase")
}

// steers has the agent joined next act on chorus-asr at a deployment that turns steering on.
func (s *AgentSuite) steers() {
	s.actsOnChorusASR()
	s.floor = audiofloor.Open(config.Agent{AudioFloorURL: "http://127.0.0.1:1", AudioFloorToolSteering: "on"})
}

func (s *AgentSuite) TestASteeredReplyIsToldTheToolButNotMadeToCallIt() {
	// Every provider's named tool choice forces the call, and the tool the caller asks for
	// is often not the one to call first, so the reply is told and still chooses.
	s.steers()
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"I have order twelve for you."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.heardAskingFor(participant, 1, "Where is my order twelve?", "lookup_order", sure, "English")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")
	s.Contains(s.replyNote(0), "the lookup_order tool", "the reply was not told which tool the caller asked for")
	s.Empty(s.model.turns()[0].ToolChoice, "the reply was made to call the tool")
	s.NotEmpty(s.model.turns()[0].Tools, "the reply was offered nothing else to call")
}

func (s *AgentSuite) TestASteeredReplyGoesToAModelThatHonoursNoToolChoice() {
	// A model that takes only auto refuses a request naming a tool to call: steering names
	// none, so such a model is steered all the same.
	s.steers()
	s.ownsTools("order 12 ships tomorrow")
	s.model.capabilities = llm.Capabilities{ToolChoices: []string{"auto"}}
	s.join(false)
	s.model.reply = []string{"I have order twelve for you."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.heardAskingFor(participant, 1, "Where is my order twelve?", "lookup_order", sure, "English")

	s.eventually(func() bool { return s.spokenText("I have order twelve for you.") }, "the steered reply was refused")
	s.Contains(s.replyNote(0), "the lookup_order tool")
	s.NoError(s.model.capabilities.Validate(s.model.turns()[0]), "the steered request asks for a tool choice the model does not honour")
}

func (s *AgentSuite) TestAnUnsureRulingSteersNothing() {
	s.steers()
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"I have order twelve for you."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.heardAskingFor(participant, 1, "Where is my order twelve?", "lookup_order", unsure, "English")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")
	s.NotContains(s.replyNote(0), "lookup_order", "a ruling short of the bar steered the reply")
}

func (s *AgentSuite) TestSteeringIsOffUnlessTheDeploymentTurnsItOn() {
	s.actsOnChorusASR()
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"I have order twelve for you."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.heardAskingFor(participant, 1, "Where is my order twelve?", "lookup_order", sure, "English")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")
	s.NotContains(s.replyNote(0), "lookup_order", "steering was on by default")
	s.Contains(s.replyNote(0), `"One moment."`, "the acknowledgment is on by default")
}

func (s *AgentSuite) TestAcknowledgmentsAreOffWhenTheDeploymentTurnsThemOff() {
	s.actsOnChorusASR()
	s.floor = audiofloor.Open(config.Agent{AudioFloorURL: "http://127.0.0.1:1", AudioFloorToolAcknowledgment: "off"})
	s.ownsTools("order 12 ships tomorrow")
	s.join(false)
	s.model.reply = []string{"I have order twelve for you."}
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.heardAskingFor(participant, 1, "Where is my order twelve?", "lookup_order", sure, "English")

	s.eventually(func() bool { return countOf[Responded](s.reported()) == 1 }, "the reply never finished")
	s.Equal([]string{"I have order twelve for you."}, s.spokenTexts())
}
