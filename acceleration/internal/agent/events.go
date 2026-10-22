package agent

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/emit"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// Event is emitted on the channel returned by Agent.Events.
//
// These describe the conversation rather than the providers. The three modality event
// streams stay available through the sessions for anyone who needs that detail.
type Event interface {
	isAgentEvent()
}

// Joined means the agent is in the call and listening. Everyone already in the call when it
// joined has been reported before it (ParticipantJoined). SFU is zero when the edge cannot
// say which one it is on.
type Joined struct {
	At  time.Time
	SFU SFU
}

func (Joined) isAgentEvent() {}

// ParticipantJoined means somebody else is now in the call.
//
// Reported because an agent in a call it did not start has no other way of knowing anybody
// is there: an agent answering a phone has to know the caller arrived before it says hello,
// and until they speak there is no audio to infer it from.
type ParticipantJoined struct {
	Participant stt.Participant
	At          time.Time
}

func (ParticipantJoined) isAgentEvent() {}

// ParticipantLeft means somebody else has gone. On a phone call there is only one of them,
// so this is the caller hanging up.
type ParticipantLeft struct {
	Participant stt.Participant
	At          time.Time
}

func (ParticipantLeft) isAgentEvent() {}

// Hearing is a transcript revision as it arrives, before anything has been decided about
// it. It is what the agent is hearing rather than what it has heard: the words are still
// changing, and most revisions are replaced by the next one.
//
// Nothing in the conversation acts on this. It is here so a person watching a call can
// see the words appear as the agent does, which is the difference between watching an
// agent think and waiting to find out what it concluded.
type Hearing struct {
	Participant stt.Participant
	Text        string
	Language    string
	// LineID names the line these words are, from the first of them until they are heard
	// or let go. It is what the transcript stores the line under, so a watcher can tell
	// the words still arriving and the stored line apart from a second one.
	LineID string
}

func (Hearing) isAgentEvent() {}

// Heard is a settled turn from a participant, once the conversation has decided it was
// meant for the agent. A revision of a turn that has not finished is a Hearing instead:
// acting on one would mean answering half a sentence.
type Heard struct {
	Participant stt.Participant
	// Text is the line as the participant said it: without words a transcriber restated
	// from the line before, and with any held from earlier in the same line in front. The
	// words the turn is answered on can differ, and are Responding's Prompt.
	Text     string
	Language string
	// LineID is the line the words were heard as while they arrived, which the transcript
	// stores them under.
	LineID string
}

func (Heard) isAgentEvent() {}

// Said is words the agent speaks as they are, without asking the model: a greeting, or
// any other line it was told to say. A voice reports it in place of a Responded, which is
// the model's answer to somebody, so nothing waiting on an answer takes the greeting for
// one. A conversation in writing has no voice and reports the line as Responded instead.
// It is reported so a transcript has the line, and a watcher sees what the caller hears.
type Said struct {
	TurnID string
	Text   string
	// Purpose is what the line is for, PurposeGreeting for the call's opening and empty
	// for any other line the agent was told to say.
	Purpose string
}

// What a line the agent says unasked is for, as Said and Responded name it.
const (
	PurposeGreeting = "greeting"
	PurposeCheckIn  = "checkin"
	PurposeUpdate   = "update"
)

func (Said) isAgentEvent() {}

// Decided is one judgement the conversation made about how to handle the call: whether
// the caller had finished, whether what they said was meant for the agent, who keeps the
// floor when both are talking, and when a silence needs filling.
//
// It carries the reason as well as the choice. A latency figure explains a slow call on
// its own, but only the reasoning explains a call that went somewhere nobody expected.
type Decided struct {
	At   time.Time
	Kind string
	// Reason is why, in words.
	Reason string
	// TurnID is the turn the judgement was about.
	TurnID string
	// Participant is who it concerned.
	Participant stt.Participant
	// Text is what was heard, or what the agent decided to say.
	Text string
	// LatencyMs is what the flow controller took to rule. Zero where nothing was asked.
	LatencyMs float64
	// LineID is the caller's line the judgement was about, when it was about one: the line
	// an ignore lets go of, so a watcher drops the words it was showing, or the line a turn
	// queued, held or answered was heard as.
	LineID string
}

func (Decided) isAgentEvent() {}

// Responding means the agent has asked the model for a reply.
type Responding struct {
	TurnID string
	// Participant is who the agent is replying to.
	Participant stt.Participant
	// Prompt is what the agent is replying to.
	Prompt string
	// Continues is the turn whose tools or delegated work this reply delivers, empty on a
	// reply to something somebody said.
	Continues string
}

func (Responding) isAgentEvent() {}

// ResponseDelta is a piece of the reply as it streams back.
type ResponseDelta struct {
	TurnID string
	Text   string
}

func (ResponseDelta) isAgentEvent() {}

// ReasoningDelta is a piece of the model's thinking while it works on a written reply.
// It is never the reply: it is only shown live to people watching the conversation.
type ReasoningDelta struct {
	TurnID string
	Text   string
}

func (ReasoningDelta) isAgentEvent() {}

// Responded means the model finished. The reply may still be being spoken.
type Responded struct {
	// PendingWork says this model reply is followed by tools or delegated work.
	PendingWork bool
	TurnID      string
	Text        string
	// Purpose is why the agent said it unasked, empty for an answer: PurposeCheckIn or
	// PurposeUpdate. A watcher tells the caller why the agent spoke up.
	Purpose string
	// TimeToFirstTokenMs is how long the participant waited for the model to start.
	TimeToFirstTokenMs float64
}

func (Responded) isAgentEvent() {}

// ModelCall reports one provider attempt, including flow and delegated work.
type ModelCall struct{ llm.CallTiming }

func (ModelCall) isAgentEvent() {}

// Blocked means the guardrail refused a turn, so what the caller hears is the policy's
// refusal rather than the model's reply.
//
// It is reported beside the ordinary turn events rather than instead of them: a refused
// turn still emits Responding and Responded, so a client that has never heard of a
// guardrail shows it as a normal exchange. This is what a client that has heard of one
// needs in order to say why.
type Blocked struct {
	TurnID string
	// Reason is what the check said, for an operator reading the log. It is not what the
	// caller hears: a reason quoting the policy back is a map of how to get around it.
	Reason string
	// Probability is what a classifier or judge put on the turn violating the policy.
	// Zero from a webhook, which answers yes or no itself.
	Probability float64
	// HeldMs is how long a finished reply waited on the verdict, which is the whole of
	// what screening in parallel costs a caller. Zero in blocking mode, where the model
	// was not asked until the verdict was in.
	HeldMs float64
}

func (Blocked) isAgentEvent() {}

// Speaking means an utterance of the agent's started being heard: the call's track took
// its first audible frame or, from an edge that cannot say, publishing its first chunk
// returned. It comes before the utterance's Spoke, and is sent for one the caller then
// talks over too, whose cut is Interrupted.
type Speaking struct {
	TurnID string
	// SynthesisID is the utterance, as on its Spoke.
	SynthesisID string
	// Text is what the voice had been sent for the utterance by then: all of a line said
	// as written or of a sentence a voice took on its own, and the start of a reply a
	// streaming voice is still being sent. A speech-to-speech model's transcript so far.
	Text string
	// Message is where that text is in the reply's Chat message, which is not the text: a
	// streaming voice's grows as it is sent more. Nil from an agent with no record of it.
	Message *MessageSpan
	// At is when the track took its first audible frame, or for an utterance reported
	// before the track got to it (one that finished being made behind speech still queued),
	// when it was reported.
	At time.Time
	// DelayMs is how long after At the utterance is expected to be heard: the speech queued
	// ahead of it at the edge, for one reported before the track got to it. Zero otherwise,
	// and for an edge that does not count its queue.
	DelayMs float64
}

func (Speaking) isAgentEvent() {}

// Spoke means a piece of the reply finished being synthesised and published.
type Spoke struct {
	TurnID string
	// SynthesisID is the utterance, as on its Speaking.
	SynthesisID string
	// Characters is how much text was synthesised, or for a speech-to-speech model how long
	// its transcript is, in Unicode code points.
	Characters int64
	// Message is where all of it is in the reply's Chat message, as on its Speaking. Nil
	// for a synthesis the agent has no record of.
	Message *MessageSpan
	// AudioDurationMs is how much speech was published.
	AudioDurationMs float64
	// TimeToFirstByteMs is how long the participant waited to hear anything.
	TimeToFirstByteMs float64
}

func (Spoke) isAgentEvent() {}

// Turn is a finished exchange, measured the way the participant experienced it. The
// legs are reported alongside the whole so a slow conversation can be attributed to
// transcription, the model or the voice rather than only observed.
//
// A leg that did not happen is zero: an interrupted turn may never have reached the
// voice, and a pipeline where one model hears and speaks for itself has only the
// roundtrip to report.
type Turn struct {
	TurnID      string
	Participant stt.Participant
	// StartedAt is when the last transcript revision arrived, before cadence settling.
	StartedAt time.Time
	// These consecutive legs run from transcript arrival to the first audio published.
	CadenceMs          float64
	DecisionMs         float64
	ModelToFirstTextMs float64
	TextToTTSMs        float64
	TTSToAudioMs       float64
	// STTLatencyMs is what the transcriber spent settling the turn.
	STTLatencyMs float64
	// LLMTTFTMs is the wait between asking the model and its first token.
	LLMTTFTMs float64
	// TTSTTFBMs is the wait between sending the first sentence and the first audio.
	TTSTTFBMs float64
	// RoundtripMs is the whole delay: last transcript revision to first audio published.
	RoundtripMs float64
	// SpeechEndToAudioMs is voice in to voice out: the roundtrip plus the time the
	// transcriber spent deciding the participant had stopped.
	SpeechEndToAudioMs float64
	// FirstFrameQueuedMs and FirstAudibleFrameMs run from the last transcript revision, like
	// RoundtripMs, to the two moments publishing returning stands in for: the edge queueing the
	// first frame of the reply for its outgoing track, and the track taking the first frame
	// that was not silence, which is when the reply could first be heard. Publishing returns
	// once no more than the queue is left of the chunk, so for a chunk longer than the queue it
	// returns later than both. Zero where the edge does not report them.
	FirstFrameQueuedMs  float64
	FirstAudibleFrameMs float64
	// SpeechEndToAudibleMs is SpeechEndToAudioMs measured to FirstAudibleFrameMs instead.
	SpeechEndToAudibleMs float64
	// AudioOutMs is how much speech the agent published for the turn.
	AudioOutMs float64
	// AudioDroppedMs is speech that was synthesised for the turn but never published,
	// because it arrived after the turn had been abandoned.
	AudioDroppedMs float64
	Interrupted    bool
}

func (Turn) isAgentEvent() {}

// Delegated means the model handed a piece of work to the subagent and carried on
// talking. The caller hears the filler around it, never the request itself.
type Delegated struct {
	StartedAt time.Time
	TaskID    string
	Skill     string
	// Prompt is what was handed over.
	Prompt string
	// TurnID is the reply the request was made in.
	TurnID string
}

func (Delegated) isAgentEvent() {}

// TaskSettled means delegated work finished and, if it produced anything the caller is
// owed, the agent has started a turn to say so.
type TaskSettled struct {
	Evidence []string
	TaskID   string
	Skill    string
	// Text is the answer, when there is one.
	Text string
	// Question is what the subagent needs asked before it can go further.
	Question string
	// ElapsedMs is how long the caller was kept company for.
	ElapsedMs float64
	Err       error
	// Files are what the work's code handed back, published where the caller can see them.
	Files []sandbox.Attachment
}

func (TaskSettled) isAgentEvent() {}

// TaskCancelled means delegated work was abandoned because its premise was gone: the
// caller moved on, the model dropped it, it ran out of time, or the call ended.
type TaskCancelled struct {
	TaskID string
	Skill  string
	Reason string
}

func (TaskCancelled) isAgentEvent() {}

// ToolRan means the model asked for a tool and it has been carried out, or tried. It is
// reported whether or not it worked, because a tool that failed still changed what the
// agent goes on to say.
type ToolStarted struct {
	Product string
	SDK     string
	ID      string
	TurnID  string
	Tool    string
	// Arguments are the model's, as JSON. Only a tool a person's device runs shows them.
	Arguments string
	StartedAt time.Time
	// PreSpeech is what the tool's connector binding asks to be said while it runs
	// (ToolPolicy.PreSpeech), so a client can show it. Empty for none.
	PreSpeech string
}

func (ToolStarted) isAgentEvent() {}

// ToolApprovalDecided is a person's answer to a call that waited for their approval. It
// changes how the call is shown; the call itself is still answered by its result.
type ToolApprovalDecided struct {
	ID      string
	TurnID  string
	Allowed bool
	// Summary is shown on a declined call, such as "Location not shared".
	Summary string
}

func (ToolApprovalDecided) isAgentEvent() {}

type ToolRan struct {
	ID     string
	TurnID string
	Tool   string
	// Arguments is the JSON the model filled in, as it wrote it.
	Arguments string
	// Result is what the model was told about the outcome.
	Result string
	// Files are what code the tool ran handed back, published where the person sees them.
	Files []sandbox.Attachment
	Err   error
}

func (ToolRan) isAgentEvent() {}

// Transferred means a human has been dialled onto the call. The agent leaves once the
// handover is done, which is immediately for a cold transfer and after the summary has been
// spoken for a warm one.
type Transferred struct {
	TurnID string
	To     string
	// Summary is what the human is told about the caller. Empty means a cold transfer.
	Summary string
}

func (Transferred) isAgentEvent() {}

// Pressed means the agent pressed digits at a menu on a call it had placed.
type Pressed struct {
	TurnID string
	Digits string
}

func (Pressed) isAgentEvent() {}

// LookedUp means the agent read the knowledge base to answer something. It carries how
// many passages came back rather than the passages themselves: a watcher wants to know the
// agent went looking and whether it found anything, not to re-read the handbook.
type LookedUp struct {
	TurnID string
	Query  string
	// Documents is how many passages bore on the question. Zero means the knowledge base
	// had nothing, which is why the agent said it did not know.
	Documents int
}

func (LookedUp) isAgentEvent() {}

// Searched means the agent found out something that is true now to answer a question. Like
// a lookup it carries how many sources came back rather than the sources themselves.
type Searched struct {
	TurnID string
	Query  string
	// Results is how many sources bore on the question. Zero means the search found
	// nothing, which is why the agent said it could not find out.
	Results int
}

func (Searched) isAgentEvent() {}

// Backchannel means the agent made a listening noise while a participant was still
// talking, which is not a turn and does not go near the model.
type Backchannel struct {
	Participant stt.Participant
	Text        string
}

func (Backchannel) isAgentEvent() {}

// Interrupted means a participant started talking over the agent, so the reply being
// spoken was abandoned.
type Interrupted struct {
	TurnID      string
	Participant stt.Participant
	// SynthesisID is the utterance the track was playing when it was cut, and HeardMs how
	// much of it had been heard: from the track taking its first audible frame, never more
	// than was published. Empty when none of the turn's open utterances had been heard.
	SynthesisID string
	HeardMs     float64
	// Message is where all that utterance had been sent is in the reply's Chat message, for
	// a reader to take HeardMs of it. Nil with SynthesisID empty.
	Message *MessageSpan
}

func (Interrupted) isAgentEvent() {}

// OverlapDecided says how the agent handled speech arriving while it was talking.
type OverlapDecided struct {
	TurnID      string
	Participant stt.Participant
	Action      string
}

func (OverlapDecided) isAgentEvent() {}

// ConversationCompacted means an old history prefix was replaced with a summary.
type ConversationCompacted struct {
	Before  int
	After   int
	Summary string
}

func (ConversationCompacted) isAgentEvent() {}

// Error reports a failure from one of the three modalities or from the edge. The agent
// keeps going: a failed turn is one lost reply, not a lost conversation.
type Error struct {
	Err error
	// Context says which part failed, e.g. "stt", "llm", "tts" or "edge".
	Context string
}

func (e Error) Error() string { return e.Err.Error() }

func (e Error) Unwrap() error { return e.Err }

func (Error) isAgentEvent() {}

// Left means the agent is out of the call.
type Left struct {
	At time.Time
}

func (Left) isAgentEvent() {}

// ModelsChanged means the session moved onto other models mid-call. Each model is the
// provider/model serving it, empty for the ones the pipeline does not run. Speech
// recognition is the target, since a transcriber opens per participant on first hearing.
type ModelsChanged struct {
	At       time.Time
	Native   bool
	LLM      string
	STT      string
	TTS      string
	STS      string
	Subagent string
	Voice    string
}

func (ModelsChanged) isAgentEvent() {}

// Emitter fans agent events out to a single consumer channel.
type Emitter = emit.Emitter[Event]

// NewEmitter returns an Emitter with the given channel buffer.
func NewEmitter(buffer int) *Emitter { return emit.New[Event](buffer) }
