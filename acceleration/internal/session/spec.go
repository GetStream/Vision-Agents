package session

import (
	"errors"
	"fmt"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// Spec is a conversation somebody outside this process asked for.
//
// It is deliberately the same set of decisions cmd/agent takes on the command line. The
// difference is only who is deciding: a flag becomes a field, and the process that used to
// be started per call becomes a session in a process that is already running.
type Spec struct {
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
	// CallType defaults to "default".
	CallType string
	// UserID is who the agent joins the call as.
	UserID string
	// UserName is the display name that goes with it.
	UserName string

	// CustomerID owns the session and is what its usage is billed to. It comes from the
	// trusted header rather than the body, so it is filled in by the API.
	CustomerID string
	// ConfigID names the agent config this session was created from, so a call can later
	// say what the agent was configured as. Empty for a session that spelled itself out.
	ConfigID string
	// AgentID keys transcripts and statistics. Empty means the call id.
	AgentID string
	// Tags are the caller's own cost labels, carried onto every request the session
	// makes.
	Tags routing.Tags

	Instructions string
	// Greeting is said on joining without going through the model. Empty means the agent
	// waits to be spoken to.
	Greeting string
	// Navigating tells an agent that placed this call how to get past whatever answers.
	Navigating bool

	LLMTarget      string
	STTTarget      string
	TTSTarget      string
	SubagentTarget string
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
		// A text agent holds its conversation in writing, so a session created from one
		// joins no call unless the request asks for a voice session explicitly.
		Text:               config.Mode == store.AgentModeText,
		STTTarget:          config.STT,
		TTSTarget:          config.TTS,
		Voice:              config.Voice,
		LLMTarget:          config.LLM,
		SubagentTarget:     config.Subagent,
		SearchTarget:       config.Search,
		Instructions:       config.Instructions,
		Greeting:           config.Greeting,
		SkillNames:         config.Skills,
		Plugins:            config.Plugins,
		Keyterms:           config.Keyterms,
		KnowledgeNamespace: config.KnowledgeNamespace,
		Tags:               routing.Tags(config.Tags),
	}
}

// Normalize fills in the defaults a caller left out and reports what cannot be defaulted.
func (s *Spec) Normalize() error {
	s.CallID = strings.TrimSpace(s.CallID)
	switch {
	case s.Text && s.CallID != "":
		return errors.New("session: a text session holds no call, so it cannot join one")
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
	// its own rather than the call id it does not have.
	if s.AgentID == "" {
		s.AgentID = s.CallID
		if s.Text {
			s.AgentID = newID()
		}
	}
	if s.LLMTarget == "" {
		s.LLMTarget = defaultLLMTarget
	}
	if s.SearchTarget == "" {
		s.SearchTarget = defaultSearchTarget
	}
	// Neither speech target means anything without a voice, and defaulting them would
	// have a text session refused by a deployment that routes only a model.
	if !s.Text {
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

	if err := s.Tags.Validate(); err != nil {
		return err
	}
	return harness.Tools{Tools: s.Tools}.Validate()
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
