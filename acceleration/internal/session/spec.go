package session

import (
	"errors"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Spec is a conversation somebody outside this process asked for.
//
// It is deliberately the same set of decisions cmd/agent takes on the command line. The
// difference is only who is deciding: a flag becomes a field, and the process that used to
// be started per call becomes a session in a process that is already running.
type Spec struct {
	// CallID is the call to join. It is the one thing with no sensible default.
	CallID string
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
	Voice          string
	LanguageHints  []string
	MaxTokens      int
	Tasks          int

	// Skills are what the voice model may hand to the subagent, spelled out. Nil means
	// SkillNames decides, and both being empty means the built-in set, which is only
	// loaded when there is a subagent to run them.
	Skills *harness.Skills
	// SkillNames are skills to look up rather than spell out: the customer's own, or one
	// of the built-in think, recall and explain.
	SkillNames []string
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
		CustomerID:         config.CustomerID,
		ConfigID:           config.ID,
		STTTarget:          config.STT,
		TTSTarget:          config.TTS,
		Voice:              config.Voice,
		LLMTarget:          config.LLM,
		SubagentTarget:     config.Subagent,
		Instructions:       config.Instructions,
		Greeting:           config.Greeting,
		SkillNames:         config.Skills,
		KnowledgeNamespace: config.KnowledgeNamespace,
		Tags:               routing.Tags(config.Tags),
	}
}

// Normalize fills in the defaults a caller left out and reports what cannot be defaulted.
func (s *Spec) Normalize() error {
	s.CallID = strings.TrimSpace(s.CallID)
	if s.CallID == "" {
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
	if s.AgentID == "" {
		s.AgentID = s.CallID
	}
	if s.LLMTarget == "" {
		s.LLMTarget = defaultLLMTarget
	}
	if s.STTTarget == "" {
		s.STTTarget = defaultSTTTarget
	}
	if s.TTSTarget == "" {
		s.TTSTarget = defaultTTSTarget
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
