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
	// Reply is what the agent is currently saying, so the controller can tell a cough
	// overlapping a read-back from a caller who is correcting it.
	Reply string
	// Unfinished says the words are what has been heard so far of an utterance still in
	// progress, so the only question is who keeps the floor rather than what to answer.
	Unfinished bool
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
	// waiting is the newest turn that has not started Create yet. A later Decide
	// replaces it, so Cerebras only ever sees the latest words plus at most one round
	// trip already in flight.
	waiting *candidate
	// inFlight is the turn whose Create is running, empty when the controller is idle.
	// Cancel uses it to start the waiter immediately: Cerebras often keeps a cancelled
	// request open until headers or a 429 retry, and waiting for that used to leave the
	// follow-up never asked.
	inFlight string
	running  sync.WaitGroup
}

// flowDeadline bounds one floor decision. The conversation model may sit for minutes;
// the controller must not, or a 429 leaves the caller unanswered.
const flowDeadline = 3 * time.Second

// candidate is one transcript revision the controller is deciding about.
type candidate struct {
	turn    FlowTurn
	askedAt time.Time
	stream  *llm.Stream
	// ctx and cancel abandon Create itself, not only a stream that has already started. A
	// revision that moved on while the HTTP request was still waiting for headers used to
	// leave that round trip occupying the model, which is how a barge-in sat for seconds
	// behind a decision nobody needed.
	ctx    context.Context
	cancel context.CancelFunc
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

Choose wait when the words are probably incomplete, especially when they end on a PIN,
member ID, phone number, or clock time that may still be growing.
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
addition that makes the current answer too long, and continue for a brief acknowledgement,
a cough or other non-speech noise, or clearly unrelated background speech. If the agent is
not speaking, choose continue.
Words that only repeat what the agent is saying are the caller's line echoing it back, so
choose continue.
When the caller has not finished and is talking over the agent, decide only the floor and
always choose wait for the disposition: stop as soon as what has been said so far is a
correction, a new request, a question, or a direct interruption such as "wait", "no", or
"hang on"; shorten for a related addition; and continue while it is only an acknowledgement,
a noise, an echo of the agent's own words, or too short to tell.`

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

	ctx, cancel := context.WithTimeout(context.Background(), flowDeadline)
	asked := &candidate{turn: turn, askedAt: time.Now(), ctx: ctx, cancel: cancel}

	f.mu.Lock()
	f.pending[turn.ID] = asked
	f.dropWaitingLocked()
	if f.inFlight != "" {
		f.waiting = asked
		f.mu.Unlock()
		return nil
	}
	f.inFlight = turn.ID
	f.mu.Unlock()

	f.running.Add(1)
	go f.run(asked)
	return nil
}

// run asks the model and waits for the decision. It is a goroutine of its own because
// Create waits for response headers, and the transcript loop that called Decide cannot
// sit in that: a revision that arrives while it does is the one that should cancel it.
func (f *flow) run(asked *candidate) {
	turn := asked.turn
	defer f.running.Done()
	defer f.advance(turn.ID)

	stream, err := f.model.Create(asked.ctx, llm.ResponseParams{
		ID:           turn.ID,
		Instructions: flowInstructions + "\n\nThe agent has been told:\n" + turn.Instructions,
		Input:        []llm.Message{{Role: llm.User, Content: flowQuestion(turn)}},
		// A decision is one small JSON object. Gemini 3 still thinks first, and 32 tokens
		// was only enough for `{"dis` — which we treated as continue, so barge-in never
		// stopped the agent.
		MaxOutputTokens: 512,
		Text:            llm.TextParams{Format: llm.FormatJSONObject},
	})
	if err != nil {
		if f.forget(turn.ID) == nil {
			return
		}
		f.logger.Error("flow controller failed", "error", err)
		f.emitter.Send(Decided{
			CandidateID: turn.ID,
			TookMs:      millis(time.Since(asked.askedAt)),
			Err:         fmt.Errorf("harness: decide flow: %w", err),
		})
		return
	}

	f.mu.Lock()
	asked.stream = stream
	abandoned := f.pending[turn.ID] != asked
	f.mu.Unlock()
	if abandoned {
		stream.Close()
		return
	}

	f.consume(asked, stream)
}

// advance starts the newest waiting turn, or marks the controller idle.
//
// Cancel of the in-flight turn may already have started the waiter. That leftover Create
// must not start it again, or two decisions about the same words race.
func (f *flow) advance(turnID string) {
	f.mu.Lock()
	if f.inFlight != turnID {
		f.mu.Unlock()
		return
	}
	next := f.waiting
	f.waiting = nil
	if next == nil {
		f.inFlight = ""
		f.mu.Unlock()
		return
	}
	f.inFlight = next.turn.ID
	f.mu.Unlock()
	f.running.Add(1)
	go f.run(next)
}

// dropWaitingLocked forgets a turn that never reached Create, because newer words
// replaced it. The caller holds f.mu.
func (f *flow) dropWaitingLocked() {
	next := f.waiting
	f.waiting = nil
	if next == nil {
		return
	}
	delete(f.pending, next.turn.ID)
	next.cancel()
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
		if reply := strings.TrimSpace(turn.Reply); reply != "" {
			state = fmt.Sprintf("is speaking right now and has so far said %q", reply)
		}
	}
	participant := strings.TrimSpace(turn.Participant)
	if participant == "" {
		participant = "An unknown speaker"
	}
	voice := ""
	if turn.AnotherVoice {
		voice = ", in a different voice from the one the agent has been talking to"
	}
	if turn.Unfinished {
		fmt.Fprintf(&asked, "\nThe agent %s.\n%s is still talking%s and has so far said: %q. "+
			"They have not finished.\n\nDecide only the floor. Return the JSON object.",
			state, participant, voice, turn.Text)
		return asked.String()
	}
	fmt.Fprintf(&asked, "\nThe agent %s.\n%s has just said%s: %q\n\nReturn the JSON object.",
		state, participant, voice, turn.Text)
	return asked.String()
}

func (f *flow) Cancel(candidateID string) error {
	f.mu.Lock()
	if f.waiting != nil && f.waiting.turn.ID == candidateID {
		f.dropWaitingLocked()
		f.mu.Unlock()
		return nil
	}
	asked, pending := f.pending[candidateID]
	if pending {
		delete(f.pending, candidateID)
	}
	f.mu.Unlock()

	if !pending {
		return nil
	}
	asked.cancel()
	// A cancelled Create can take a while to come back, and the turn waiting behind it
	// must not wait for that. advance starts it now, and the leftover call from run does
	// nothing because the turn in flight has moved on.
	f.advance(candidateID)
	if asked.stream == nil {
		return nil
	}
	return asked.stream.Close()
}

func (f *flow) Close() error {
	f.mu.Lock()
	f.dropWaitingLocked()
	for _, asked := range f.pending {
		asked.cancel()
	}
	clear(f.pending)
	f.mu.Unlock()

	// Closing the session abandons whatever the controller is still deciding, which is
	// what lets every consumer reach the end of its stream.
	err := f.model.Close()
	f.running.Wait()
	return err
}

// consume waits for one decision and reports it.
func (f *flow) consume(asked *candidate, stream *llm.Stream) {
	candidateID := asked.turn.ID
	response, err := llm.Collect(stream)
	if f.forget(candidateID) == nil {
		// The transcript moved on, so nobody is waiting for this any more.
		return
	}
	took := millis(time.Since(asked.askedAt))
	if err != nil {
		if response.Status == llm.StatusCancelled {
			return
		}
		f.logger.Error("flow controller failed", "error", err)
		f.emitter.Send(Decided{CandidateID: candidateID, TookMs: took, Err: err})
		return
	}

	answer, err := parseFlow(response.OutputText)
	if err != nil {
		// Truncated JSON used to mean continue, which is how Gemini thinking ate the
		// token budget and the caller talking over the agent never took the floor.
		if asked.turn.Unfinished {
			f.logger.Warn("unusable flow decision while the caller is talking, stopping",
				"error", err, "answer", response.OutputText)
			answer = flowAnswer{Disposition: Wait, Floor: Stop}
		} else {
			f.logger.Warn("unusable flow decision, answering the caller anyway",
				"error", err, "answer", response.OutputText)
			answer = flowAnswer{Disposition: Respond, Floor: Continue}
		}
	} else {
		f.logger.Debug("flow controller answered",
			"candidate", candidateID, "answer", response.OutputText,
			"took_ms", response.TimeToFirstTokenMs)
	}
	f.emitter.Send(Decided{
		CandidateID: candidateID,
		Disposition: answer.Disposition,
		Floor:       answer.Floor,
		TookMs:      took,
	})
}

// forget drops a candidate and hands it back, so the wait it cost the caller is measured
// from the moment it was asked about rather than guessed at from the model's own timings.
// Nil means the transcript moved on and nobody is waiting for the answer.
func (f *flow) forget(candidateID string) *candidate {
	f.mu.Lock()
	defer f.mu.Unlock()
	asked, ok := f.pending[candidateID]
	if !ok {
		return nil
	}
	delete(f.pending, candidateID)
	return asked
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
