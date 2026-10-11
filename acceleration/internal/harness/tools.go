package harness

import (
	"embed"
	"errors"
	"fmt"
	"os"

	"gopkg.in/yaml.v3"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// defaultToolsFS carries the built-in tool set so an agent with telephony works without an
// external file.
//
//go:embed tools.yaml
var defaultToolsFS embed.FS

// Tool is one thing the fast model may do rather than say.
//
// It is deliberately not a skill: a skill asks a better model a question and folds the
// answer into the conversation, while a tool reaches outside the conversation and changes
// something that cannot be changed back. That difference is why a tool is run by the agent,
// which knows what call it is on, rather than here.
type Tool struct {
	// Name is how the model asks for it.
	Name string `yaml:"name"`
	// Description is what the model is told the tool does, and is the whole of how it
	// decides when to reach for one.
	Description string `yaml:"description"`
	// Parameters is a JSON Schema object describing the arguments. It is untyped because
	// a schema is untyped: the shape is whatever the tool accepts.
	Parameters map[string]any `yaml:"parameters"`
	// Client says a person's device runs the tool rather than the caller. The model never
	// sees it; a persistent conversation shows the call as awaiting that device.
	Client bool `yaml:"-"`
	// DisplayTitle is what a call is doing, in words for the people in the conversation.
	DisplayTitle string `yaml:"-"`
	// Approval, when set, is what a person is asked before each call runs. A persistent
	// conversation shows the call as awaiting their answer.
	Approval *ToolApproval `yaml:"-"`
}

// ToolApproval is the question a person answers before a call runs.
type ToolApproval struct {
	Title   string
	Message string
	// ReasonArgument names the string argument in which the model says why it wants the
	// call, which is shown as the question's reason.
	ReasonArgument string
	AllowTitle     string
	DeclineTitle   string
}

// Tools is the set a harness was configured with.
type Tools struct {
	Tools []Tool `yaml:"tools"`
}

// Lookup returns a tool by name.
func (t Tools) Lookup(name string) (Tool, bool) {
	for _, tool := range t.Tools {
		if tool.Name == name {
			return tool, true
		}
	}
	return Tool{}, false
}

// Requests renders the set for a completion request. It returns nil when there are none,
// so a request carrying no tools is not merely one carrying an empty list: a model offered
// an empty toolbox still answers as though it had one.
func (t Tools) Requests() []llm.Tool {
	if len(t.Tools) == 0 {
		return nil
	}

	rendered := make([]llm.Tool, 0, len(t.Tools))
	for _, tool := range t.Tools {
		rendered = append(rendered, llm.Tool{
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  tool.Parameters,
		})
	}
	return rendered
}

// usePolicy is what a reply model is told about using the tools it is offered. A model that
// answers as fast as a voice needs to will otherwise gather more than a tool asks for, ask
// leave to do what it was just asked, or stop after one call, and a caller hangs up on a
// conversation that never acts. Acting at once is not the same as acting in silence: whatever
// the operator's instructions have the agent say before acting, such as reading the caller's
// details back, is still said, in the same turn as the call, and a bare filler does not stand
// in for it.
//
// The caller also hears nothing but that sentence while a slow tool runs, and a read-back
// takes seconds to say, so a hold phrase tacked on at its end, or after the result, comes
// after the wait it was meant to cover. The sentence therefore opens with the hold phrase and
// runs on into the read-back. It is one sentence, not a one-word sentence and then the
// read-back, because the pause between two utterances is where a caller's interruption lands.
// After a result the answer is given without another hold phrase, so a wait has one. A chain
// of calls is one wait too: a sentence before each link queues behind the last, so the
// caller hears a run of hold phrases while the answer is already waiting behind them. Tools
// the request still needs are called straight away, together when they are independent.
//
// It gives no example of a hold phrase: the models open every reply with whatever example
// they are shown, and a caller hears the same words before every lookup.
//
// It says nothing about any one tool, so it holds for any set, and it leaves confirmation to
// the operator's own instructions and to a tool's approval.
const usePolicy = "Before calling a tool, say one short sentence that opens with a brief hold " +
	"phrase in your own words, never the same twice, and goes straight on, with no full stop " +
	"between, into what your instructions ask you to say before acting (such as reading the " +
	"caller's details back) or else what you are doing. Say it there and only there: never " +
	"as a sentence of its own, never in place of a required read-back, never after the call " +
	"or after a result. Then call the tool in the same turn once every argument it requires " +
	"is known. " + useArguments

// useArguments is what usePolicy and textUsePolicy both say about filling a call in and
// going on from its result, held once so the two cannot drift apart.
const useArguments = "Do not collect optional arguments or ask permission for what was " +
	"asked. Take a name or value as given (a surname is a name). After a result, answer from " +
	"it; if the request needs tools, call them straight away without a word, together when " +
	"independent. Pass bare values, not phrases. Where your instructions or a tool's " +
	"approval require confirmation first, follow them."

// textUsePolicy is usePolicy for a conversation held in writing. The reasons to act at
// once, to take values as given and to say first what the operator's instructions have the
// agent say before acting all hold in writing too. The hold phrase does not: in writing
// there is no silence for it to fill, and whatever the model writes on the way to an answer
// is part of the reply. A sentence before each call therefore stays on the page, and a
// request that takes a chain of calls opens its answer with a run of near-identical lines,
// one per link, which reads as the agent repeating itself.
//
// It is about tool calls only. Handing work to a skill is another matter: Skills.TextPrompt
// still has the model say it is on it, since that answer comes in a later reply.
const textUsePolicy = "Call a tool as soon as every argument it requires is known, without " +
	"announcing it: what you write before a call does not fill the wait, it stays in your " +
	"reply. So write nothing before a call or between calls, except what your instructions " +
	"ask you to write before acting (such as reading the caller's details back). " +
	useArguments

// sayDo holds what the agent says to what its tools did. A voice model told to say a hold
// phrase before a call can say the outcome instead: in Voicebench Gemma told a caller
// "You're all set" for a booking it never made. It sits beside the use policy rather than in
// it, so that stays short.
const sayDo = "Never say something is done (booked, changed, cancelled, sent) unless a " +
	"tool result in this conversation shows it; if you have not called that tool, call it now."

// Prompt is what the model is told about using its tools: when to call one and how to fill
// it in. It is empty when there are none, so a harness without tools adds nothing to the
// system prompt.
func (t Tools) Prompt() string {
	if len(t.Tools) == 0 {
		return ""
	}
	return usePolicy + " " + sayDo
}

// TextPrompt is Prompt for a conversation held in writing, which fills no pause before a
// call (textUsePolicy).
func (t Tools) TextPrompt() string {
	if len(t.Tools) == 0 {
		return ""
	}
	return textUsePolicy + " " + sayDo
}

// Validate reports the first tool the harness could not use.
func (t Tools) Validate() error {
	seen := map[string]struct{}{}
	for _, tool := range t.Tools {
		if tool.Name == "" {
			return stack.Wrap(errors.New("harness: every tool needs a name"))
		}
		if tool.Description == "" {
			return stack.Wrap(fmt.Errorf("harness: tool %s has no description, so the model would "+
				"never know when to use it", tool.Name))
		}
		if _, duplicate := seen[tool.Name]; duplicate {
			return stack.Wrap(fmt.Errorf("harness: tool %s is declared twice", tool.Name))
		}
		seen[tool.Name] = struct{}{}
	}
	return nil
}

// DefaultTools returns the built-in tool set.
func DefaultTools() (Tools, error) {
	raw, err := defaultToolsFS.ReadFile("tools.yaml")
	if err != nil {
		return Tools{}, stack.Wrap(fmt.Errorf("harness: read default tools: %w", err))
	}
	return parseTools(raw)
}

// LoadTools reads a tool set, or the built-in default when path is empty.
func LoadTools(path string) (Tools, error) {
	if path == "" {
		return DefaultTools()
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		return Tools{}, fmt.Errorf("harness: read tools %s: %w", path, err)
	}
	return parseTools(raw)
}

func parseTools(raw []byte) (Tools, error) {
	var tools Tools
	if err := yaml.Unmarshal(raw, &tools); err != nil {
		return Tools{}, stack.Wrap(fmt.Errorf("harness: parse tools: %w", err))
	}
	if err := tools.Validate(); err != nil {
		return Tools{}, err
	}
	return tools, nil
}
