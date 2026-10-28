package agent

import (
	"cmp"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// Why a reply was abandoned, the cause label of router_agent_interruptions_total.
const (
	// causeSettled is the controller giving the floor to words the caller had finished.
	causeSettled = "settled"
	// causeTalking is the controller giving the floor to a caller still talking.
	causeTalking = "talking"
	// causeUnheard is the caller saying more before any of the reply went out.
	causeUnheard = "unheard"
	// causeAPI is Agent.Interrupt, a caller outside the call.
	causeAPI = "api"
	// causeProvider is a speech-to-speech model stopping on its own hearing of the caller.
	causeProvider = "provider"
	// causeVoiceLost is the voice dropping mid-reply.
	causeVoiceLost = "voice_lost"
	// causeVoiceRefused is the voice refusing a line said as written.
	causeVoiceRefused = "voice_refused"
)

// echo buckets: how much of the words that took the floor the reply had itself said.
const (
	echoNoWords = "no_words"
	echoNone    = "none"
	echoSome    = "some"
	echoMost    = "most"
)

// echoMostShare is where an echo starts. The agent's own line heard back through a
// caller's speaker is transcribed near word for word, so even with one word in three
// misheard it lands at or above this; a caller's own words share only a few common ones
// ("you", "the") with the reply, and land below.
const echoMostShare = 2.0 / 3

// interruption is why a reply was abandoned, for the log line, the counter and the turn's
// span.
type interruption struct {
	cause string
	// ruler is whose ruling took the floor, empty when no ruling did.
	ruler harness.Ruler
	// tookMs is what the ruling took, zero when nothing was ruled.
	tookMs float64
	// words is what the caller said that took the floor, empty when nobody said anything
	// the router heard.
	words string
	// reply is the text the reply had handed to the voice. It is compared with words and
	// never logged.
	reply string
	// echo is the bucket of how much of words the reply had said.
	echo string
}

// interrupted closes a turn whose reply was abandoned, and says why: one line in the log,
// router_agent_interruptions_total and the turn's span, all from the first report. A turn
// already told about is left to the report that told it, so a second cause never reaches the
// span.
func (a *Agent) interrupted(turnID string, participant stt.Participant, why interruption) {
	share, compared := echoShare(why.words, why.reply)
	why.echo = echoBucket(share, compared)
	a.mu.Lock()
	_, told := a.toldInterrupts[turnID]
	a.toldInterrupts[turnID] = struct{}{}
	a.mu.Unlock()
	if told {
		return
	}
	playedMs := a.turns.interrupt(turnID, why)
	ruler := cmp.Or(string(why.ruler), "none")
	interruptions.WithLabelValues(why.cause, ruler, why.echo).Inc()
	fields := []any{
		"turn", turnID, "participant", participant.ID, "cause", why.cause, "ruler", ruler,
		"took_ms", why.tookMs, "words", why.words, "played_ms", playedMs, "echo", why.echo,
	}
	if compared {
		fields = append(fields, "echo_share", share)
	}
	a.logger.Info("the agent was interrupted", fields...)
}

// echoShare is the share of the caller's words that are also words of the reply, both
// lowercased with their punctuation gone. False when either has no words to compare.
func echoShare(said, reply string) (float64, bool) {
	heard, replied := wordList(said), wordList(reply)
	if len(heard) == 0 || len(replied) == 0 {
		return 0, false
	}
	known := make(map[string]bool, len(replied))
	for _, word := range replied {
		known[word] = true
	}
	found := 0
	for _, word := range heard {
		if known[word] {
			found++
		}
	}
	return float64(found) / float64(len(heard)), true
}

func echoBucket(share float64, compared bool) string {
	switch {
	case !compared:
		return echoNoWords
	case share == 0:
		return echoNone
	case share < echoMostShare:
		return echoSome
	}
	return echoMost
}
