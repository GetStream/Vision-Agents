package audiofloor

import "github.com/GetStream/Vision-Agents/acceleration/internal/stt"

// Unwritten is an Asker that is told of each utterance that ended with no final: its words
// came back empty, or were never written. A check over the agent may have written some of
// them, which the agent acted on (Overheard), and no final will carry those words. why is
// the reason, as far as the transcriber knows it: "language" for words in a language the
// callers don't speak, "behind" for a final dropped while the server was behind, "failed" for
// one the server did not write, the server's other noise, or empty for no words at all. It is
// optional: the transcriber tells an asker that implements it.
type Unwritten interface {
	Unwritten(participant stt.Participant, utterance int64, why string)
}

// emptyWords is why the words of an utterance last came back empty.
type emptyWords struct {
	utterance int64
	why       string
}

// noted keeps why the words of utterance came back empty, the server's noise, for the asker
// to be told if it ends with no final.
func (t *Transcriber) noted(utterance int64, noise string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.noise = emptyWords{utterance: utterance, why: noise}
}

// unwritten tells the asker, if it listens (Unwritten), that utterance ended with no final,
// for why, or else for why its words last came back empty.
func (t *Transcriber) unwritten(participant stt.Participant, utterance int64, why string) {
	t.mu.Lock()
	asker := t.asker
	if why == "" && t.noise.utterance == utterance {
		why = t.noise.why
	}
	t.mu.Unlock()
	if told, ok := asker.(Unwritten); ok {
		told.Unwritten(participant, utterance, why)
	}
}
