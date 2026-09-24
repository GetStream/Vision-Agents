package session

import (
	"errors"
	"fmt"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// Recall names a conversation to read history out of. Both halves are needed because a
// channel is only readable as the agent it belongs to, and a fork onto a different agent
// still has to read the old one's words.
//
// Messages is the history itself, for a fork read out of what its parent recorded rather
// than out of a channel. When it is set the channel is not read.
type Recall struct {
	AgentID        string
	ConversationID string
	Messages       []llm.Message
}

// Spec is a conversation somebody outside this process asked for.
//
// It is deliberately the same set of decisions cmd/agent takes on the command line. The
// difference is only who is deciding: a flag becomes a field, and the process that used to
// be started per call becomes a session in a process that is already running.
type Spec struct {
	PersistConversation bool
	ConversationID      string
	ContextTruncated    bool
	// CallID is the call to join. It is the one thing with no sensible default, and the
	// one thing a text session does not have.
	CallID string
	// Text holds the conversation in writing: no call is joined, nothing is transcribed
	// and nothing is spoken. Everything between hearing and answering is unchanged, so a
	// text session has the same skills, knowledge and tools a call would have had.
	Text bool
	// Edge is a call the caller has already opened, used instead of the manager's own.
	// It is how a conversation is held against something other than a real transport: the
	// manager's factory is handed a spec and cannot be given a particular one back.
	Edge agent.Edge
	// NoReview leaves the conversation unreviewed when it ends. A call is summarised on a
	// model afterwards, which is worth paying for once per caller and not worth paying for
	// once per conversation in a batch that is already being judged.
	NoReview bool
	// CallType defaults to "agent".
	CallType string
	// UserID is who the agent joins the call as.
	UserID string
	// UserName is the display name that goes with it.
	UserName string

	// CustomerID owns the session and is what its usage is billed to. It comes from the
	// trusted header rather than the body, so it is filled in by the API.
	CustomerID string
	// Caller is the end user who asked for the session, which is not the same thing as
	// UserID above: that is who the agent joins the call as, this is who wanted it. It
	// comes from the credential rather than the body, so the API fills it in, and it is
	// what the session's daily limits are counted against. Empty for a session a
	// customer's own backend started, which is not limited.
	Caller routing.Caller
	// CallerKind is what sort of caller that was, and it qualifies Caller.UserID: an
	// anonymous caller may go by any name, so the name alone cannot be what one person's
	// conversation is kept from another by. The API fills it in from the credential.
	CallerKind auth.Kind
	// ConfigID names the agent config this session was created from, so a call can later
	// say what the agent was configured as. Empty for a session that spelled itself out.
	ConfigID string
	// AgentName is the name that config was found by, which is what a caller addressed the
	// agent as. Kept alongside the id because it is what a caller filters their old
	// conversations on, and because renaming a config must not rewrite what older sessions
	// were opened against.
	AgentName string
	// AgentID keys transcripts and statistics. Empty means the call id.
	AgentID string
	// Tags are the caller's own cost labels, carried onto every request the session
	// makes.
	Tags routing.Tags

	// Incognito holds the conversation and records nothing about it: no session row, no
	// turns, no transcript. It is the one field that makes a session unfindable afterwards,
	// which is the whole point of it -- a person asking a question they would rather not
	// have kept should not have to trust that a "hidden" flag is honoured everywhere.
	//
	// It forces PersistConversation off, because a channel in Stream Chat is a record.
	Incognito bool
	// Title and Description are the caller's own names for the conversation, for a list a
	// person reads. Never shown to the model: what a conversation is called is a label on
	// it rather than part of it.
	Title       string
	Description string
	// Project is what the conversation belongs to. Also merged into Tags, so spend breaks
	// down by project without the caller labelling it twice.
	Project string
	// Custom is whatever the caller wants to remember about the session, handed back
	// untouched and never read here. A field this package interpreted would be a field it
	// could break.
	Custom map[string]any
	// ModelOverwrites is what the caller asked to change about the models for this session,
	// over whatever the config decided. The route fields are folded into the targets above
	// during Normalize; the rest reach the conversation model as options.
	ModelOverwrites store.ModelOverwrites
	// ForkedFrom is the session this one continued from, empty for one opened fresh.
	ForkedFrom string
	// Recall is the conversation a fork starts from, which is not the conversation it
	// writes into. The parent's words are read out of its channel and given to the model;
	// the fork's own transcript goes into its own channel, so continuing a conversation
	// twice gives two transcripts rather than one with both halves interleaved.
	Recall *Recall

	Instructions string
	// Greeting is said on joining without going through the model. Empty means the agent
	// waits to be spoken to.
	Greeting string
	// Guardrail is a guardrail.md, whole: frontmatter saying how a turn is screened, then
	// the policy in prose. Empty means every turn is answered.
	Guardrail string
	// Navigating tells an agent that placed this call how to get past whatever answers.
	Navigating bool

	LLMTarget string
	STTTarget string
	TTSTarget string
	// STSTarget routes one native audio model that hears and speaks for itself. Naming
	// one makes this a native session: the agent opens that model and no transcriber,
	// conversation model or voice, so the three targets above are left as they are.
	STSTarget      string
	SubagentTarget string
	// ControllerTarget routes the flow controller. Internal rather than customer-facing:
	// a caller configures the conversation's model, not the classifier that decides who
	// holds the floor, so it is defaulted here rather than read from a config.
	ControllerTarget string
	// SearchTarget routes what the agent finds out about today. Unlike the three above it
	// is not needed for a conversation to happen, so a deployment that routes no search
	// simply leaves the tool unoffered.
	SearchTarget  string
	Voice         string
	LanguageHints []string
	// Keyterms are the business-specific words a transcriber would otherwise get wrong.
	// A provider that cannot be told about vocabulary ignores them.
	Keyterms  []string
	MaxTokens int
	Tasks     int

	// Skills are what the voice model may hand to the subagent, spelled out. Nil means
	// SkillNames decides, and both being empty means the built-in set, which is only
	// loaded when there is a subagent to run them.
	Skills *harness.Skills
	// SkillNames are skills to look up rather than spell out: the customer's own, or one
	// of the built-in think, recall and explain.
	SkillNames []string
	// Plugins are hosted MCP servers this session may reach, named from the catalog.
	Plugins []string
	// KnowledgeNamespace is what the agent may look things up in. Empty means it knows
	// only what it was told.
	KnowledgeNamespace string
	// Sandbox names where the subagent may run code it writes, "daytona" being the one
	// provider there is. Empty means it runs none, and works everything out in its head.
	Sandbox string
	// Tools are what the voice model may do rather than say. These are the caller's own
	// functions: the session carries the request out to whoever asked for the session
	// and waits for them to answer it.
	Tools []harness.Tool
	// ToolTimeoutMs bounds how long the model waits on one of them. Zero is the default.
	ToolTimeoutMs int

	// Backchannel murmurs while a participant is still talking, the way a person does.
	Backchannel bool
	// MinConfidence is how sure the transcriber must be before the agent answers rather
	// than checks what was meant.
	MinConfidence float64

	VideoSource    string
	VideoMaxFrames int

	// Memory scopes what the session recalls and remembers. Without it the agent starts
	// the call knowing nothing but its instructions.
	Memory MemorySpec
	// Phone attaches the session to a number, which is what turns transferring on.
	Phone *PhoneSpec

	// CampaignID and ContactID say which piece of outbound work this call is, so a
	// campaign can be told how its contacts went. Both are empty for a call nobody
	// scheduled.
	CampaignID string
	ContactID  string
}

// MemorySpec is the caller's memory filter: who the memories are about, and what narrows
// them further.
type MemorySpec struct {
	// UserID is who the memories belong to. Empty means the customer.
	UserID string
	// AppID separates two deployments sharing one memory account.
	AppID string
	// Filter narrows recall with the caller's own labels.
	Filter map[string]string
}

// PhoneSpec is the number the session acts from.
type PhoneSpec struct {
	// Number is one of the customer's own, which is what a transferred human sees.
	Number string
	// Vendor carries an outbound leg.
	Vendor string
	// VendorCallID is that leg, set for a call the agent placed. Without one the agent
	// has no keypad to press at.
	VendorCallID string
	// To is who was rung, set for a call the agent placed.
	To string
}

// FromConfig is what a stored agent config means as a spec: everything a caller would
// otherwise have had to spell out. What is left empty is what a config does not decide,
// starting with which call to join.
func FromConfig(config store.AgentConfig) Spec {
	return Spec{
		CustomerID: config.CustomerID,
		ConfigID:   config.ID,
		AgentName:  config.Name,
		// A text agent holds its conversation in writing, so a session created from one
		// joins no call unless the request asks for a voice session explicitly.
		Text:           config.Mode == store.AgentModeText,
		STTTarget:      config.STT,
		TTSTarget:      config.TTS,
		STSTarget:      config.STS,
		Voice:          config.Voice,
		LLMTarget:      config.LLM,
		SubagentTarget: config.Subagent,
		VideoSource:    config.VideoSource, VideoMaxFrames: config.VideoMaxFrames,
		SearchTarget:       config.Search,
		Instructions:       config.Instructions,
		Greeting:           config.Greeting,
		Guardrail:          config.Guardrail,
		SkillNames:         config.Skills,
		Plugins:            config.Plugins,
		Keyterms:           config.Keyterms,
		KnowledgeNamespace: config.KnowledgeNamespace,
		Sandbox:            config.Sandbox,
		Tags:               routing.Tags(config.Tags),
	}
}

// Normalize fills in the defaults a caller left out and reports what cannot be defaulted.
func (s *Spec) Normalize() error {
	// The overwrites are applied before anything is defaulted, so a target the caller asked
	// for is what gets defaulted around rather than one the config happened to name. A
	// caller who asks for a native model in a text session, say, should be refused for that
	// reason rather than have the refusal depend on which of the two was read first.
	s.applyOverwrites()

	// Incognito is honoured here rather than at each of the places that records something,
	// because one place that forgot would be a conversation kept against its caller's
	// wishes. Everything downstream reads the spec, so turning persistence off here turns
	// it off everywhere.
	if s.Incognito {
		s.PersistConversation = false
		s.ConversationID = ""
		s.NoReview = true
	}

	// A project is a cost label as much as it is a grouping, so it is merged into the tags
	// rather than the caller having to say it twice. An explicit tag wins: somebody who
	// spelled out project in tags meant that.
	if s.Project != "" {
		if s.Tags == nil {
			s.Tags = routing.Tags{}
		}
		if _, named := s.Tags[projectTag]; !named {
			s.Tags[projectTag] = s.Project
		}
	}

	s.CallID = strings.TrimSpace(s.CallID)
	switch {
	case s.Text && s.CallID != "":
		return errors.New("session: a text session holds no call, so it cannot join one")
	case s.Text && s.Native():
		return errors.New("session: a text session has no voice, so it cannot run a speech-to-speech model")
	case !s.Text && s.CallID == "":
		return errors.New("session: a call id is required")
	}
	if s.CustomerID == "" {
		return errors.New("session: a customer id is required")
	}

	if s.CallType == "" {
		s.CallType = defaultCallType
	}
	if s.UserID == "" {
		s.UserID = defaultUserID
	}
	if s.UserName == "" {
		s.UserName = defaultUserName
	}
	// The agent id keys the transcript and the timings, so a text session is given one of
	// its own rather than the call id it does not have. A session resuming a conversation
	// is given none: the transcript it rejoins was keyed under whichever session opened
	// it, and a second id minted here would not match, so the conversation is asked for
	// the one it was written under instead.
	if s.AgentID == "" && !(s.Text && s.PersistConversation && s.ConversationID != "") {
		s.AgentID = s.CallID
		if s.Text {
			s.AgentID = newID()
		}
	}
	// A native session opens none of the cascade's three models, so none of their targets
	// is defaulted: the call row would otherwise name a model that never ran.
	if s.LLMTarget == "" && !s.Native() {
		s.LLMTarget = defaultLLMTarget
	}
	if s.ControllerTarget == "" && !s.Native() {
		s.ControllerTarget = defaultControllerTarget
	}
	if s.SearchTarget == "" {
		s.SearchTarget = defaultSearchTarget
	}
	// Neither speech target means anything without a voice, and defaulting them would
	// have a text session refused by a deployment that routes only a model.
	if !s.Text && !s.Native() {
		if s.STTTarget == "" {
			s.STTTarget = defaultSTTTarget
		}
		if s.TTSTarget == "" {
			s.TTSTarget = defaultTTSTarget
		}
	}

	s.Keyterms = stt.CleanKeyterms(s.Keyterms)
	if len(s.Keyterms) > stt.MaxKeyterms {
		return fmt.Errorf("session: at most %d keyterms may be named, and this asks for %d",
			stt.MaxKeyterms, len(s.Keyterms))
	}

	if s.VideoMaxFrames == 0 {
		s.VideoMaxFrames = 1
	}
	if s.VideoMaxFrames < 1 || s.VideoMaxFrames > 8 {
		return fmt.Errorf("session: video.max_frames must be between 1 and 8")
	}

	if err := s.Tags.Validate(); err != nil {
		return err
	}
	return harness.Tools{Tools: s.Tools}.Validate()
}

// Native reports whether this session is held by one speech-to-speech model rather than
// the cascade of a transcriber, a conversation model and a voice.
func (s Spec) Native() bool { return s.STSTarget != "" }

// projectTag is the cost label a project is carried as, which is the one the stats rollups
// already break spend down by.
const projectTag = "project"

// applyOverwrites folds what the caller asked to change about the models into the targets.
//
// Route names go onto the spec because that is where the rest of the package looks for
// them; the numbers do not, because they are per-request options rather than routing
// decisions, and they reach the model through LLMOptions instead.
func (s *Spec) applyOverwrites() {
	over := s.ModelOverwrites
	if over.LLM != "" {
		s.LLMTarget = over.LLM
	}
	if over.STT != "" {
		s.STTTarget = over.STT
	}
	if over.TTS != "" {
		s.TTSTarget = over.TTS
	}
	if over.STS != "" {
		s.STSTarget = over.STS
	}
	if over.Subagent != "" {
		s.SubagentTarget = over.Subagent
	}
	if over.Search != "" {
		s.SearchTarget = over.Search
	}
}

// LLMOverwrites is what the caller asked to change about the conversation model itself, as
// options to merge over whatever the route resolved.
//
// Separate from the routing half because they travel differently: a target is chosen once
// when the session opens, while these ride along on every request the session makes. The
// field names are the provider's own, which is why thinking arrives here as the reasoning
// effort the providers that support one already speak.
func (s Spec) LLMOverwrites() options.LLM {
	over := s.ModelOverwrites
	return options.LLM{
		ReasoningEffort: over.Thinking,
		Temperature:     over.Temperature,
		MaxOutputTokens: over.MaxOutputTokens,
		Verbosity:       over.Verbosity,
	}
}

// prompt is what the agent is told to be. An agent that placed the call is told how to get
// through whatever answers, ahead of whatever it was told to do once it has.
func (s Spec) prompt() string {
	if !s.Navigating {
		return s.Instructions
	}
	if s.Instructions == "" {
		return agent.NavigatingInstructions
	}
	return agent.NavigatingInstructions + "\n\n" + s.Instructions
}

// duplex is how the agent listens and talks at the same time.
func (s Spec) duplex() agent.DuplexOptions {
	return agent.DuplexOptions{
		Backchannel:   s.Backchannel,
		MinConfidence: s.MinConfidence,
	}
}
