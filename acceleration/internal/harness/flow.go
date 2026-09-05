package harness

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
)

// Disposition says what to do with a stable transcript revision.
type Disposition string

const (
	Wait    Disposition = "wait"
	Ignore  Disposition = "ignore"
	Respond Disposition = "respond"
	Clarify Disposition = "clarify"
)

func (d Disposition) Valid() bool {
	return d == Wait || d == Ignore || d == Respond || d == Clarify
}

// Floor says how the agent should handle speech arriving while it is talking.
type Floor string

const (
	Stop     Floor = "stop"
	Shorten  Floor = "shorten"
	Continue Floor = "continue"
)

func (f Floor) Valid() bool { return f == Stop || f == Shorten || f == Continue }

// FlowTurn is one transcript candidate for the flow controller.
type FlowTurn struct {
	ID           string
	Instructions string
	History      []llm.Message
	Participant  string
	Text         string
	Speaking     bool
	// AnotherVoice says the words came from someone other than the participant whose
	// microphone they arrived on, which the transcribers that tell voices apart can hear.
	// It is evidence rather than a verdict: somebody else in the room is usually talking
	// to the room, but they may equally have leaned in to answer for the caller.
	AnotherVoice bool
}

type flow struct {
	model   *llmrouter.Session
	emitter *Emitter
	logger  *slog.Logger

	mu sync.Mutex
	// pending holds each candidate still with the controller: when it was asked about, so
	// the wait it cost the caller can be reported with the answer, and the stream it is
	// being answered on, so a candidate whose transcript changed can be abandoned.
	pending map[string]*candidate
	running sync.WaitGroup
}

// candidate is one transcript revision the controller is deciding about.
type candidate struct {
	askedAt time.Time
	stream  *llm.Stream
}

type flowAnswer struct {
	Disposition Disposition `json:"disposition"`
	Floor       Floor       `json:"floor"`
}

// flowHistory is how much of the conversation the controller is shown. What it decides is
// whether the latest words belong to that conversation, which the last few turns settle.
const flowHistory = 6

const flowInstructions = `You control the floor of a live voice conversation. You never talk
to the caller and you never answer their question: another model does that. Your only output
is one JSON object and no other text:
{"disposition":"wait|ignore|respond|clarify","floor":"stop|shorten|continue"}

Choose wait when the words are probably incomplete.
Choose ignore only when the words are clearly background speech or addressed to somebody else.
Words in a different voice come from somebody else at the caller's microphone, who is
usually talking to the room rather than to the agent, so lean towards ignore unless they
plainly address it.
Choose clarify when the caller addressed the agent but their request is ambiguous.
Choose respond for a complete, relevant thought.
A recorded menu reading out its options is one thought and not several, however long the
pauses between them: choose wait until it has asked for a choice, and never interrupt one,
because it is not listening and starts again from the top if it is talked over.
If the agent is speaking, stop for a correction or direct interruption, shorten for a related
addition that makes the current answer too long, and continue for a brief acknowledgement or
clearly unrelated background speech. If the agent is not speaking, choose continue.`

func newFlow(model *llmrouter.Session, emitter *Emitter, logger *slog.Logger) *flow {
	return &flow{
		model:   model,
		emitter: emitter,
		logger:  logger,
		pending: map[string]*candidate{},
	}
}

func (f *flow) Decide(turn FlowTurn) error {
	if strings.TrimSpace(turn.ID) == "" {
		return errors.New("harness: a flow candidate id is required")
	}
	if strings.TrimSpace(turn.Text) == "" {
		return errors.New("harness: flow candidate text is required")
	}

	asked := &candidate{askedAt: time.Now()}
	f.mu.Lock()
	f.pending[turn.ID] = asked
	f.mu.Unlock()

	stream, err := f.model.Create(context.Background(), llm.ResponseParams{
		ID:           turn.ID,
		Instructions: flowInstructions + "\n\nThe agent has been told:\n" + turn.Instructions,
		Input:        []llm.Message{{Role: llm.User, Content: flowQuestion(turn)}},
		// A decision is one small JSON object, so there is nothing to think about and
		// nothing to be verbose with. Both are latency the caller waits through.
		MaxOutputTokens: 32,
		Text:            llm.TextParams{Format: llm.FormatJSONObject},
	})
	if err != nil {
		f.forget(turn.ID)
		return fmt.Errorf("harness: decide flow: %w", err)
	}

	f.mu.Lock()
	asked.stream = stream
	abandoned := f.pending[turn.ID] != asked
	f.mu.Unlock()
	if abandoned {
		stream.Close()
	}

	f.running.Add(1)
	go f.consume(turn.ID, stream)
	return nil
}

// flowQuestion is the whole of what the controller is asked.
//
// The conversation is quoted inside the question rather than replayed as turns, because a
// model handed a conversation to continue continues it: primed with a caller's question in
// the user role, it answers the question instead of classifying it, and the caller's turn
// is lost to a model they never hear.
func flowQuestion(turn FlowTurn) string {
	var asked strings.Builder
	asked.WriteString("Conversation so far:\n")
	history := turn.History
	if len(history) > flowHistory {
		history = history[len(history)-flowHistory:]
	}
	if len(history) == 0 {
		asked.WriteString("(nothing said yet)\n")
	}
	for _, message := range history {
		speaker := "Caller"
		if message.Role == llm.Assistant {
			speaker = "Agent"
		}
		fmt.Fprintf(&asked, "%s: %s\n", speaker, message.Content)
	}

	state := "is not speaking"
	if turn.Speaking {
		state = "is speaking right now"
	}
	participant := strings.TrimSpace(turn.Participant)
	if participant == "" {
		participant = "An unknown speaker"
	}
	voice := ""
	if turn.AnotherVoice {
		voice = ", in a different voice from the one the agent has been talking to"
	}
	fmt.Fprintf(&asked, "\nThe agent %s.\n%s has just said%s: %q\n\nReturn the JSON object.",
		state, participant, voice, turn.Text)
	return asked.String()
}

func (f *flow) Cancel(candidateID string) error {
	f.mu.Lock()
	asked, pending := f.pending[candidateID]
	if pending {
		delete(f.pending, candidateID)
	}
	f.mu.Unlock()

	if !pending || asked.stream == nil {
		return nil
	}
	return asked.stream.Close()
}

func (f *flow) Close() error {
	// Closing the session abandons whatever the controller is still deciding, which is
	// what lets every consumer reach the end of its stream.
	err := f.model.Close()
	f.running.Wait()
	return err
}

// consume waits for one decision and reports it.
func (f *flow) consume(candidateID string, stream *llm.Stream) {
	defer f.running.Done()

	response, err := llm.Collect(stream)
	took, pending := f.forget(candidateID)
	if !pending {
		// The transcript moved on, so nobody is waiting for this any more.
		return
	}
	if err != nil {
		if response.Status == llm.StatusCancelled {
			return
		}
		f.logger.Error("flow controller failed", "error", err)
		f.emitter.Send(Decided{CandidateID: candidateID, TookMs: millis(took), Err: err})
		return
	}

	answer, err := parseFlow(response.OutputText)
	if err != nil {
		// An answer that cannot be read must not cost the caller their turn, so the agent
		// replies to them, after finishing whatever it was saying.
		f.logger.Warn("unusable flow decision, answering the caller anyway",
			"error", err, "answer", response.OutputText)
		answer = flowAnswer{Disposition: Respond, Floor: Continue}
	} else {
		f.logger.Debug("flow controller answered",
			"candidate", candidateID, "answer", response.OutputText,
			"took_ms", response.TimeToFirstTokenMs)
	}
	f.emitter.Send(Decided{
		CandidateID: candidateID,
		Disposition: answer.Disposition,
		Floor:       answer.Floor,
		TookMs:      millis(took),
	})
}

// forget drops a candidate and reports how long it was pending, so the wait it cost the
// caller is measured from the moment it was asked about rather than guessed at from the
// model's own timings.
func (f *flow) forget(candidateID string) (time.Duration, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	asked, ok := f.pending[candidateID]
	if !ok {
		return 0, false
	}
	delete(f.pending, candidateID)
	return time.Since(asked.askedAt), true
}

func millis(took time.Duration) float64 {
	return float64(took.Microseconds()) / 1000
}

func parseFlow(text string) (flowAnswer, error) {
	var answer flowAnswer
	if err := json.Unmarshal([]byte(llm.Unfence(text)), &answer); err != nil {
		return flowAnswer{}, fmt.Errorf("harness: decode flow decision: %w", err)
	}
	if !answer.Disposition.Valid() {
		return flowAnswer{}, fmt.Errorf("harness: invalid flow disposition %q", answer.Disposition)
	}
	if !answer.Floor.Valid() {
		return flowAnswer{}, fmt.Errorf("harness: invalid floor decision %q", answer.Floor)
	}
	return answer, nil
}
