package session

import (
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
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
	// ID is the id the session is held by. Empty is generated as a UUIDv7; one the caller
	// chose has to be a UUID.
	ID                  string
	PersistConversation bool
	ConversationID      string
	ContextTruncated    bool
	// CallID is the call the agent is on while voice is started: agent:<ID>, which joining
	// creates. Empty while the conversation is held in writing.
	CallID string
	// Text holds the conversation in writing: no call is joined, nothing is transcribed
	// and nothing is spoken. Everything between hearing and answering is unchanged, so a
	// text session has the same skills, knowledge and tools a call would have had. Starting
	// voice clears it (Voiced) and stopping it sets it again (Written).
	Text bool
	// heldSTS and heldSubagent are the speech-to-speech model and the subagent the
	// conversation runs on while voice is started, kept while it is held in writing.
	heldSTS      string
	heldSubagent string
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
	// StreamApp is the Stream app the session acts in, its pin: zero for the deployment's
	// own. The manager resolves it once, before anything is done in Stream, and the whole
	// session, its call, its transcript and its records, keeps it.
	StreamApp int64
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
	//
	// One exception: each connector tool call still leaves its row in the connection's call
	// log (store.ConnectorInvocation), with no session id, no arguments and no results, and
	// the grant changes it causes are audited with no request or session id. The row is the
	// use of a credential, which its owner is owed; nothing in it names the conversation.
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
	// Reopened is when a persistent text session that ended first opened, set to carry it
	// on under the same id rather than refuse an id already used. A chat is never over for
	// the person writing in it, so its session ending is not the conversation ending.
	Reopened time.Time
	// Recall is the conversation a fork starts from, which is not the conversation it
	// writes into. The parent's words are read out of its channel and given to the model;
	// the fork's own transcript goes into its own channel, so continuing a conversation
	// twice gives two transcripts rather than one with both halves interleaved.
	Recall *Recall
	// History is the conversation so far as the caller kept it, for a caller that holds its
	// own thread and opens a session to answer in it. The model is handed it before the
	// first response, where a resumed conversation's history goes, and it is recorded
	// nowhere: not as turns, not in a transcript, not in Chat.
	History []persistent.HistoryLine

	Instructions string
	// Greeting is said on joining without going through the model. Empty means the agent
	// waits to be spoken to.
	Greeting string
	// VaryGreeting has the model say its own variation of Greeting rather than the words.
	VaryGreeting bool
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
	Keyterms []string
	// VisibleTools are the tools whose steps end users see on a persistent conversation's
	// replies, as names or path.Match patterns. Empty shows search and web_search.
	VisibleTools []string
	MaxTokens    int
	Tasks        int

	// Harness is which harness the session runs, from the agent's config. Empty is the
	// default, and so is every session today; a caller cannot choose it.
	Harness string
	// DispatchText hands what an end user writes to the customer's dispatch worker rather
	// than the model, from the agent's config. The model answers only the server.
	DispatchText bool
	// EpisodeCards has a phone call write its episode card into the caller's omni-channel,
	// and a session on a thread channel or a phone call start with the person's other cards,
	// from the agent's config. Off, the session does what it did before the cards existed.
	EpisodeCards bool
	// ProgressiveTools offers plugin, MCP server and connector tools by a summary, and
	// answers the first call to each with its full description instead of running it.
	ProgressiveTools bool

	// SkillNames are the skills the voice model may hand to the subagent: the agent
	// config's own, or one of the built-in think, recall and explain. Empty means the
	// built-in set, which is only loaded when there is a subagent to run them.
	SkillNames []string
	// Plugins are hosted MCP servers this session may reach, named from the catalog with how
	// each is reached. Those marked User the caller reaches with their own account, and a
	// session with no caller is offered none of them.
	Plugins []store.PluginEntry
	// MCPServers are MCP servers outside the catalog, opened by their URL with no login, the
	// app's, or each caller's own.
	MCPServers []store.MCPServer
	// ConnectorBindings are the agent config's connector bindings: which connector's tools
	// the session may call, through which connection, and exactly which tools. A binding
	// wins over a plugin entry for the same provider (withoutBoundPlugins).
	ConnectorBindings []store.ConnectorBinding
	// ConnectorSelections are the connections the caller picked for the config's session
	// bindings, one per alias. Only references: the session checks each against the
	// binding and the verified Caller when it opens, and again on every call.
	ConnectorSelections []ConnectorSelection
	// ServerInstructions are what those servers said at initialize about using their
	// tools, added after Instructions. The session fills it in once they are open.
	ServerInstructions string
	// KnowledgeNamespace is what the agent may look things up in. Empty means it knows
	// only what it was told.
	KnowledgeNamespace string
	// Sandbox names where the subagent may run code it writes, "daytona" being the one
	// provider there is. Empty means it runs none, and works everything out in its head.
	Sandbox string
	// SandboxOptions is how the sandbox is built and how long code may run in it.
	SandboxOptions sandbox.Config
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

// ConnectorSelection is the connection a caller picked for one session binding, by the
// binding's alias. It never carries a credential.
type ConnectorSelection struct {
	Name         string
	ConnectionID string
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
		VaryGreeting:       config.GreetingMode == store.GreetingVariation,
		Guardrail:          config.Guardrail,
		SkillNames:         config.Skills,
		Plugins:            config.Plugins,
		MCPServers:         config.MCPServers,
		ConnectorBindings:  config.Connectors,
		Keyterms:           config.Keyterms,
		VisibleTools:       config.VisibleTools,
		KnowledgeNamespace: config.KnowledgeNamespace,
		Sandbox:            config.Sandbox,
		SandboxOptions:     config.SandboxOptions,
		Harness:            config.Harness,
		DispatchText:       config.DispatchText,
		EpisodeCards:       config.EpisodeCards,
		ProgressiveTools:   config.ProgressiveTools,
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

	if s.Harness == "" {
		s.Harness = harness.Default
	}
	if s.Harness != harness.Default {
		return stack.Wrap(fmt.Errorf("session: there is no harness called %q", s.Harness))
	}

	if s.ID == "" {
		id, err := uuid.NewV7()
		if err != nil {
			return stack.Wrap(fmt.Errorf("session: generating an id: %w", err))
		}
		s.ID = id.String()
	} else if held, ok := persistent.SessionID(s.ID); ok {
		s.ID = held
	} else {
		return stack.Wrap(fmt.Errorf("session: the id %q is not one a session can have: up to 64 letters, digits, - and _, not starting support- or thread-", s.ID))
	}

	// Checked before incognito clears the conversation id, so naming both is refused
	// whatever else the request says.
	if err := checkHistory(s.History, s.ConversationID); err != nil {
		return err
	}

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

	s.CallID = joinedCallID(s.CallID)
	// A voice session given no call to join holds its own, named after the session, which
	// joining creates.
	if !s.Text && s.CallID == "" {
		s.CallID, s.CallType = s.ID, defaultCallType
	}
	// A speech-to-speech model is what the conversation speaks with once voice is started,
	// so a session held in writing keeps it rather than running it.
	if s.Text && s.STSTarget != "" {
		s.heldSTS, s.STSTarget = s.STSTarget, ""
	}
	switch {
	case !s.Reopened.IsZero() && !(s.Text && s.PersistConversation && s.ConversationID != ""):
		return stack.Wrap(errors.New("session: only a persistent text conversation is reopened"))
	case s.Text && s.CallID != "":
		return stack.Wrap(errors.New("session: a text session holds no call, so it cannot join one"))
	case s.Text && s.Native():
		return stack.Wrap(errors.New("session: a text session has no voice, so it cannot run a speech-to-speech model"))
	}
	if s.CustomerID == "" {
		return stack.Wrap(errors.New("session: a customer id is required"))
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
		s.AgentID = s.KeyedAgentID()
		if s.Text {
			s.AgentID = newID()
		}
	}
	// A native session opens none of the cascade's three models, so none of their targets
	// is defaulted: the call row would otherwise name a model that never ran.
	if s.LLMTarget == "" && !s.Native() {
		s.LLMTarget = defaultLLMTarget
	}
	// A text session runs on one model. Nobody is waiting on a voice while it thinks, so
	// the skills it hands over run on the model holding the conversation.
	if s.Text {
		s.heldSubagent, s.SubagentTarget = s.SubagentTarget, s.LLMTarget
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

	// A connector binding wins over a plugin entry for the same provider, so the session
	// does not reach one account by two paths, the second with the plugin's own login.
	s.Plugins = s.withoutBoundPlugins(s.Plugins)

	s.Keyterms = stt.CleanKeyterms(s.Keyterms)
	if len(s.Keyterms) > stt.MaxKeyterms {
		return stack.Wrap(fmt.Errorf("session: at most %d keyterms may be named, and this asks for %d",
			stt.MaxKeyterms, len(s.Keyterms)))
	}

	if s.VideoMaxFrames == 0 {
		s.VideoMaxFrames = 1
	}
	if s.VideoMaxFrames < 1 || s.VideoMaxFrames > 8 {
		return stack.Wrap(fmt.Errorf("session: video.max_frames must be between 1 and 8"))
	}

	if err := s.Tags.Validate(); err != nil {
		return err
	}
	return harness.Tools{Tools: s.Tools}.Validate()
}

// checkHistory refuses history from the caller that the model would not be handed whole.
// The limits are the ones history read back from Chat is cut to, so a caller's thread is
// held to what the router's own would be; a caller is told rather than cut, since only
// it knows which messages matter.
func checkHistory(lines []persistent.HistoryLine, conversationID string) error {
	if len(lines) == 0 {
		return nil
	}
	if conversationID != "" {
		return stack.Wrap(errors.New("session: history and conversation_id both say what was said " +
			"before; a resumed conversation reads its own, so name one"))
	}
	if len(lines) > persistent.MaxHistoryMessages {
		return stack.Wrap(fmt.Errorf("session: history holds %d messages, more than the %d a session opens with",
			len(lines), persistent.MaxHistoryMessages))
	}
	size := 0
	for i, line := range lines {
		switch {
		case line.Role != "user" && line.Role != "assistant":
			return stack.Wrap(fmt.Errorf("session: history[%d].role is %q; it must be user or assistant", i, line.Role))
		case line.Text == "":
			return stack.Wrap(fmt.Errorf("session: history[%d].text is empty", i))
		case utf8.RuneCountInString(line.Name) > persistent.MaxAuthorName:
			return stack.Wrap(fmt.Errorf("session: history[%d].name is longer than %d characters", i, persistent.MaxAuthorName))
		}
		size += utf8.RuneCountInString(line.Text)
	}
	if size > persistent.MaxHistoryRunes {
		return stack.Wrap(fmt.Errorf("session: history holds %d characters of text, more than the %d a session opens with",
			size, persistent.MaxHistoryRunes))
	}
	return nil
}

// KeyedAgentID is the agent id a caller's spec names for the session, before Normalize: its
// own, else a voice session's call id, which Normalize gives it. A text session without one
// is given a new id, which no caller names, so it names none.
func (s Spec) KeyedAgentID() string {
	if s.AgentID != "" || s.Text {
		return s.AgentID
	}
	return joinedCallID(s.CallID)
}

// Voiced is the spec once voice is started: on the call agent:<ID>, speaking with the
// models the conversation was configured with, or the defaults.
func (s Spec) Voiced() Spec {
	if !s.Text {
		return s
	}
	s.Text = false
	s.CallID, s.CallType = s.ID, defaultCallType
	s.STSTarget, s.heldSTS = s.heldSTS, ""
	s.SubagentTarget, s.heldSubagent = s.heldSubagent, ""
	if !s.Native() {
		if s.STTTarget == "" {
			s.STTTarget = defaultSTTTarget
		}
		if s.TTSTarget == "" {
			s.TTSTarget = defaultTTSTarget
		}
	}
	return s
}

// Written is the spec once voice is stopped: on no call, and on one model, as Normalize
// leaves a session held in writing.
func (s Spec) Written() Spec {
	if s.Text {
		return s
	}
	s.Text = true
	s.CallID = ""
	s.heldSTS, s.STSTarget = s.STSTarget, ""
	if s.LLMTarget == "" {
		s.LLMTarget = defaultLLMTarget
	}
	s.heldSubagent, s.SubagentTarget = s.SubagentTarget, s.LLMTarget
	return s
}

// joinedCallID is a call id as the session joins it and is keyed under: without the spaces
// around it. Normalize and KeyedAgentID both read a call id through it.
func joinedCallID(id string) string {
	return strings.TrimSpace(id)
}

// ConversationChannel is the id, without its type, of the agent channel ConversationID names:
// the channel a call's transcript is written into (chatlog.Options.Channel). Empty for a
// ConversationID that names no agent channel, whose transcript goes into the agent id's.
func (s Spec) ConversationChannel() string {
	channel := strings.TrimPrefix(s.ConversationID, streamapp.AgentChannelType+":")
	if channel == s.ConversationID {
		return ""
	}
	return channel
}

// TranscriptChannel is the cid of the channel a call's transcript is written into: the
// conversation's agent channel, else the agent id's, as chatlog.New picks it from
// ConversationChannel. Example: conversation_id "messaging:X" under agent id "front-desk" is
// written into agent:front-desk.
func (s Spec) TranscriptChannel() string {
	channel := s.ConversationChannel()
	if channel == "" {
		channel = s.AgentID
	}
	return streamapp.AgentChannelType + ":" + channel
}

// Shared reports whether more than one verified person writes in the conversation: a thread
// channel (persistent.ThreadChannelPrefix), where everyone in the external thread does, such
// as a thread in a Slack channel. Such a session uses the app's connections only, never one
// person's: the multi-person rule (architecture doc on connectors/planning, «One-way doors»
// row 7).
func (s Spec) Shared() bool {
	return strings.HasPrefix(s.ConversationID, streamapp.AgentChannelType+":"+persistent.ThreadChannelPrefix)
}

// boundProvider reports whether a connector binding names the provider id. A connector id
// is the provider's id: the built-in connectors share theirs with the plugin catalog (slack
// is in both internal/plugins/plugins.yaml and internal/connectors/providers/slack.yaml).
func (s Spec) boundProvider(id string) bool {
	for _, binding := range s.ConnectorBindings {
		if binding.ConnectorID == id {
			return true
		}
	}
	return false
}

// withoutBoundPlugins is entries less those for a provider a connector binding names.
func (s Spec) withoutBoundPlugins(entries []store.PluginEntry) []store.PluginEntry {
	// Without a binding the entries are left exactly as they were, the same slice.
	if len(s.ConnectorBindings) == 0 {
		return entries
	}
	var kept []store.PluginEntry
	for _, entry := range entries {
		if !s.boundProvider(entry.Name) {
			kept = append(kept, entry)
		}
	}
	return kept
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
	var parts []string
	if s.Navigating {
		parts = append(parts, agent.NavigatingInstructions)
	}
	if s.Instructions != "" {
		parts = append(parts, s.Instructions)
	}
	if s.ServerInstructions != "" {
		parts = append(parts, s.ServerInstructions)
	}
	return strings.Join(parts, "\n\n")
}

// duplex is how the agent listens and talks at the same time.
func (s Spec) duplex() agent.DuplexOptions {
	return agent.DuplexOptions{
		Backchannel:   s.Backchannel,
		MinConfidence: s.MinConfidence,
	}
}
