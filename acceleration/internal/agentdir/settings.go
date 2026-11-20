package agentdir

import (
	"slices"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Settings is what agent.yaml declares: what the agent is run with. A setting the file
// leaves out is empty or nil here, and leaves whatever the stored config has.
type Settings struct {
	Name string
	// Description is for whoever reads the file. Nothing is stored of it.
	Description string
	Mode        string
	Region      string
	STT         string
	TTS         string
	// STS and Image are nil when the file says nothing, and empty when it turns them off.
	STS            *string
	Voice          string
	LLM            string
	Harness        string
	Role           string
	Subagent       string
	Search         string
	Image          *string
	Greeting       *Greeting
	Sandbox        string
	SandboxOptions *SandboxOptions
	// Connectors, Tools, ChatChannels and Agents are nil when the file says nothing, which
	// leaves the stored ones, and empty when it lists none, which removes them.
	Connectors   *[]Connector
	Tools        *[]ToolGroup
	Channels     *Channels
	ChatChannels *[]ChatChannel
	Scheduling   *Scheduling
	Calls        *Calls
	FloorControl *FloorControl
	Agents       *[]ChildAgent
	Keyterms     []string
	Tags         map[string]string
	Video        *Video
	Dispatch     *Dispatch
	UI           *UISettings
}

type Greeting struct {
	Text string
	Mode string
}

type SandboxOptions struct {
	Image            string
	Setup            []string
	Timeout          time.Duration
	CPU              int
	MemoryGB         int
	DiskGB           int
	Network          bool
	AllowedDomains   []string
	NetworkAllowList []string
}

// Connector is one connector the agent binds: the alias alone, or a mapping naming it.
type Connector struct {
	Name string
	// Connector is the connector id. Empty means the name.
	Connector string
	// Tools are names or patterns. Nil is every tool.
	Tools *[]string
	// Connection is an id or a label of one of the app's connections. Nil is a session binding.
	Connection      *string
	Required        *bool
	Sandbox         bool
	RequireApproval *[]string
	Loading         string
	OutputRetention string
}

type ToolGroup struct {
	Name            string
	Loading         string
	Tools           []string
	OutputRetention string
}

type Channels struct {
	WhatsApp string
	SMS      string
	IMessage string
	Identity string
}

type ChatChannel struct {
	Type   string
	Answer string
}

type Scheduling struct {
	Enabled         *bool
	MaxTasksPerUser *int
	DefaultTimezone string
}

type Calls struct {
	EndCall  EndCall
	Press    CallTool
	Transfer Transfer
}

type CallTool struct {
	Enabled *bool
}

type EndCall struct {
	Enabled *bool
	When    string
}

type Transfer struct {
	Enabled      *bool
	Destinations []TransferDestination
}

type TransferDestination struct {
	Name   string
	Number string
	When   string
	Warm   bool
}

type FloorControl struct {
	Model     *string
	MaxWaitMs *int
}

type ChildAgent struct {
	Config        string
	Tool          string
	Description   string
	TimeoutMs     int
	MaxToolRounds int
}

type Video struct {
	Source    string
	MaxFrames int
}

type Dispatch struct {
	IncomingCall string
	Text         string
}

// retired are the keys agent.yaml no longer reads, each with what replaced it. One that
// declares nothing, a null or an empty list, is let through: it never did anything.
var retired = map[string]string{
	"plugins":       "plugins were replaced by connectors, so name the provider under connectors",
	"plugin_events": "plugins were replaced by connectors, so name the provider under connectors",
	"mcp_servers":   "a server named by URL is now a custom connector, so create one for its URL and name it under connectors",
}

var toolLoadings = []string{"full", "summary", "deferred"}

// settings reads agent.yaml.
func (r *reader) settings(content string) Settings {
	var s Settings
	root := r.document(content)
	if root == nil {
		return s
	}
	refused := func(key string) field {
		return field{key, func(value *yaml.Node) {
			if list := resolved(value); list.Kind != yaml.SequenceNode || len(list.Content) > 0 {
				r.fail(value, "no longer read: %s", retired[key])
			}
		}}
	}
	r.mapping(root, AgentFile,
		field{"name", r.text(&s.Name)},
		field{"description", r.text(&s.Description)},
		field{"mode", r.text(&s.Mode)},
		field{"region", r.text(&s.Region)},
		field{"stt", r.text(&s.STT)},
		field{"tts", r.text(&s.TTS)},
		field{"sts", r.optionalText(&s.STS)},
		field{"voice", r.text(&s.Voice)},
		field{"llm", r.text(&s.LLM)},
		field{"harness", r.text(&s.Harness)},
		field{"role", r.text(&s.Role)},
		field{"subagent", r.text(&s.Subagent)},
		field{"search", r.text(&s.Search)},
		field{"image", r.optionalText(&s.Image)},
		field{"greeting", func(value *yaml.Node) {
			s.Greeting = &Greeting{}
			// The text alone, as the getstream CLI has always read it, or a mapping with its mode.
			if resolved(value).Kind == yaml.ScalarNode {
				s.Greeting.Text = resolved(value).Value
				return
			}
			r.mapping(value, "greeting", field{"text", r.text(&s.Greeting.Text)}, field{"mode", r.text(&s.Greeting.Mode)})
		}},
		field{"sandbox", r.text(&s.Sandbox)},
		field{"sandbox_options", func(value *yaml.Node) { s.SandboxOptions = r.sandboxOptions(value) }},
		field{"connectors", func(value *yaml.Node) {
			connectors := []Connector{}
			if r.each(value, func(item *yaml.Node) { connectors = append(connectors, r.connector(item)) }) {
				s.Connectors = &connectors
			}
		}},
		field{"tools", func(value *yaml.Node) { s.Tools = r.toolGroups(value) }},
		field{"channels", func(value *yaml.Node) {
			s.Channels = &Channels{}
			line := func(into *string) func(*yaml.Node) {
				return func(value *yaml.Node) { r.mapping(value, "a channel", field{"number", r.text(into)}) }
			}
			r.mapping(value, "channels",
				field{"whatsapp", line(&s.Channels.WhatsApp)},
				field{"sms", line(&s.Channels.SMS)},
				field{"imessage", line(&s.Channels.IMessage)},
				field{"identity", r.text(&s.Channels.Identity)})
		}},
		field{"chat_channels", func(value *yaml.Node) {
			channels := []ChatChannel{}
			if r.each(value, func(item *yaml.Node) {
				var channel ChatChannel
				r.mapping(item, "a chat channel", field{"type", r.text(&channel.Type)}, field{"answer", r.text(&channel.Answer)})
				channels = append(channels, channel)
			}) {
				s.ChatChannels = &channels
			}
		}},
		field{"scheduling", func(value *yaml.Node) {
			s.Scheduling = &Scheduling{}
			r.mapping(value, "scheduling",
				field{"enabled", r.optionalBoolean(&s.Scheduling.Enabled)},
				field{"max_tasks_per_user", r.optionalInteger(&s.Scheduling.MaxTasksPerUser)},
				field{"default_timezone", r.text(&s.Scheduling.DefaultTimezone)})
		}},
		field{"calls", func(value *yaml.Node) { s.Calls = r.calls(value) }},
		field{"floor_control", func(value *yaml.Node) {
			s.FloorControl = &FloorControl{}
			r.mapping(value, "floor_control",
				field{"model", func(value *yaml.Node) {
					if value.Kind != yaml.ScalarNode || value.ShortTag() != "!!str" {
						r.fail(value, "must be a string, such as stream/chorus-asr")
						return
					}
					s.FloorControl.Model = &value.Value
				}},
				field{"max_wait_ms", r.optionalInteger(&s.FloorControl.MaxWaitMs)})
		}},
		field{"agents", func(value *yaml.Node) {
			agents := []ChildAgent{}
			if r.each(value, func(item *yaml.Node) {
				var agent ChildAgent
				r.mapping(item, "an agent",
					field{"config", r.text(&agent.Config)},
					field{"tool", r.text(&agent.Tool)},
					field{"description", r.text(&agent.Description)},
					field{"timeout_ms", r.integer(&agent.TimeoutMs)},
					field{"max_tool_rounds", r.integer(&agent.MaxToolRounds)})
				agents = append(agents, agent)
			}) {
				s.Agents = &agents
			}
		}},
		field{"keyterms", r.list(&s.Keyterms)},
		field{"tags", r.labels(&s.Tags)},
		field{"video", func(value *yaml.Node) {
			s.Video = &Video{MaxFrames: 1}
			r.mapping(value, "video",
				field{"source", r.text(&s.Video.Source)},
				field{"max_frames", func(value *yaml.Node) {
					r.integer(&s.Video.MaxFrames)(value)
					// Zero reads as one.
					s.Video.MaxFrames = max(s.Video.MaxFrames, 1)
					if s.Video.MaxFrames < 1 || s.Video.MaxFrames > 8 {
						r.fail(value, "must be an integer from 1 to 8")
					}
				}})
		}},
		field{"dispatch", func(value *yaml.Node) {
			s.Dispatch = &Dispatch{}
			r.mapping(value, "dispatch",
				field{"incoming_call", r.text(&s.Dispatch.IncomingCall)},
				field{"text", r.text(&s.Dispatch.Text)})
		}},
		field{"ui", func(value *yaml.Node) { s.UI = r.uiSettings(value) }},
		// The application's own section, which nothing here reads.
		field{"app", func(*yaml.Node) {}},
		refused("plugins"), refused("plugin_events"), refused("mcp_servers"),
	)
	return s
}

func (r *reader) sandboxOptions(node *yaml.Node) *SandboxOptions {
	options := &SandboxOptions{}
	r.mapping(node, "sandbox_options",
		field{"image", r.text(&options.Image)},
		field{"setup", r.list(&options.Setup)},
		field{"timeout", func(value *yaml.Node) {
			timeout, err := time.ParseDuration(value.Value)
			if err != nil || timeout <= 0 || timeout > 30*time.Minute {
				r.fail(value, "must be a duration up to 30m, such as 90s, not %q", value.Value)
				return
			}
			options.Timeout = timeout
		}},
		field{"cpu", r.integer(&options.CPU)},
		field{"memory_gb", r.integer(&options.MemoryGB)},
		field{"disk_gb", r.integer(&options.DiskGB)},
		field{"network", r.boolean(&options.Network)},
		field{"allowed_domains", r.list(&options.AllowedDomains)},
		field{"network_allow_list", r.list(&options.NetworkAllowList)})
	return options
}

func (r *reader) connector(node *yaml.Node) Connector {
	var connector Connector
	switch node = resolved(node); {
	case null(node):
	case node.Kind == yaml.ScalarNode:
		connector.Name = node.Value
	default:
		r.mapping(node, "a connector",
			field{"name", r.text(&connector.Name)},
			field{"connector", r.text(&connector.Connector)},
			field{"tools", func(value *yaml.Node) {
				r.optionalList(&connector.Tools)(value)
				if connector.Tools != nil && len(*connector.Tools) == 0 {
					r.fail(value, "[] grants nothing; leave tools out for every tool")
				}
			}},
			field{"connection", func(value *yaml.Node) {
				r.optionalText(&connector.Connection)(value)
				if connector.Connection != nil && *connector.Connection == "" {
					r.fail(value, "must be a connection id or label")
				}
			}},
			field{"required", r.optionalBoolean(&connector.Required)},
			field{"sandbox", r.boolean(&connector.Sandbox)},
			field{"require_approval", r.optionalList(&connector.RequireApproval)},
			field{"loading", r.loading(&connector.Loading)},
			field{"output_retention", r.text(&connector.OutputRetention)})
	}
	if connector.Name == "" {
		r.fail(node, "a connector needs a name")
	}
	return connector
}

func (r *reader) loading(into *string) func(*yaml.Node) {
	return func(value *yaml.Node) {
		r.text(into)(value)
		if *into != "" && !slices.Contains(toolLoadings, *into) {
			r.fail(value, "is %s, not %q", strings.Join(toolLoadings, ", "), *into)
		}
	}
}

// toolGroups reads the tools list. The mapping tools was before groups is refused with where
// each of its settings went.
func (r *reader) toolGroups(node *yaml.Node) *[]ToolGroup {
	if resolved(node).Kind == yaml.MappingNode {
		r.fail(node, "is a list of groups, each with a name and its tools, such as "+
			"- {name: booking, loading: full, tools: [lookup_booking]}; progressive: true is now "+
			"loading: summary, and output_retention moves onto each group and connector")
		return nil
	}
	groups := []ToolGroup{}
	if !r.each(node, func(item *yaml.Node) {
		var group ToolGroup
		if !r.mapping(item, "a tool group",
			field{"name", r.text(&group.Name)},
			field{"loading", r.loading(&group.Loading)},
			field{"tools", r.list(&group.Tools)},
			field{"output_retention", r.text(&group.OutputRetention)}) {
			return
		}
		switch {
		case group.Name == "":
			r.fail(item, "a tool group needs a name")
		case len(group.Tools) == 0:
			r.fail(item, "tool group %q names no tools", group.Name)
		}
		groups = append(groups, group)
	}) {
		return nil
	}
	return &groups
}

func (r *reader) calls(node *yaml.Node) *Calls {
	calls := &Calls{}
	r.mapping(node, "calls",
		field{"end_call", func(value *yaml.Node) {
			r.mapping(value, "end_call",
				field{"enabled", r.optionalBoolean(&calls.EndCall.Enabled)},
				field{"when", r.text(&calls.EndCall.When)})
		}},
		field{"press", func(value *yaml.Node) {
			r.mapping(value, "press", field{"enabled", r.optionalBoolean(&calls.Press.Enabled)})
		}},
		field{"transfer", func(value *yaml.Node) {
			r.mapping(value, "transfer",
				field{"enabled", r.optionalBoolean(&calls.Transfer.Enabled)},
				field{"destinations", func(value *yaml.Node) {
					r.each(value, func(item *yaml.Node) {
						var destination TransferDestination
						r.mapping(item, "a destination",
							field{"name", r.text(&destination.Name)},
							field{"number", r.text(&destination.Number)},
							field{"when", r.text(&destination.When)},
							field{"warm", r.boolean(&destination.Warm)})
						calls.Transfer.Destinations = append(calls.Transfer.Destinations, destination)
					})
				}})
		}})
	return calls
}
