package agentdir

import (
	"bytes"
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Render writes a directory as its files, in path order. The output is canonical: agent.yaml's
// keys in the order Settings has them, what is empty left out, a connector that says only its
// name written as the name alone, and UI files at their default paths. Parse reads back what
// was rendered, apart from Description and Ignored, which are not written, and from what
// unwritten says, each a sentence about something the format cannot hold as it is.
func Render(directory *Directory) (files []File, unwritten []string) {
	settings := directory.Settings
	if directory.UI != nil {
		ui := directory.UI.Settings
		ui.Components, ui.Blueprints = "", ""
		settings.UI = &ui
	}
	files = []File{{AgentFile, encode(settingsNode(settings))}}
	add := func(file, content string) {
		if content != "" {
			files = append(files, File{file, content})
		}
	}
	add(InstructionsFile, ended(directory.Instructions))
	add(GuardrailFile, ended(directory.Guardrail))
	if directory.UI != nil {
		add(UIComponentsFile, directory.UI.Components)
		add(UIBlueprintsFile, directory.UI.Blueprints)
	}
	folders := map[string]bool{}
	for _, skill := range directory.Skills {
		// A folder of its own on every file system, ones that fold case included.
		folder := skill.Name
		for number := 1; !skillFolderName.MatchString(folder) || folders[strings.ToLower(folder)]; number++ {
			folder = fmt.Sprintf("skill-%d", number)
		}
		folders[strings.ToLower(folder)] = true
		at := SkillsDir + "/" + folder + "/"
		add(at+SkillFile, skillFile(skill, folder != skill.Name))
		// Frontmatter is read a line at a time, so a value is one line.
		if strings.ContainsAny(skill.Name+skill.Description, "\r\n") {
			unwritten = append(unwritten, fmt.Sprintf("The skill %s has a name or a description over several lines, which a "+
				"SKILL.md's frontmatter cannot hold, so it is written on one line.", oneLine(skill.Name)))
		}
		for _, file := range slices.Sorted(maps.Keys(skill.Files)) {
			if file == SkillFile || !ValidPath(file) || hidden(file) {
				unwritten = append(unwritten, fmt.Sprintf("The skill %s has a file called %s, which a skill folder cannot "+
					"hold, so it is left out and a sync of these files removes it.", oneLine(skill.Name), file))
				continue
			}
			add(at+file, skill.Files[file])
		}
	}
	for _, document := range directory.Knowledge {
		add(KnowledgeDir+"/"+document.Source, document.Text)
	}
	if len(directory.KnowledgeURLs) > 0 {
		pages := sequence()
		for _, page := range directory.KnowledgeURLs {
			if page.Title == "" && page.Description == "" && page.RefreshHours == 0 {
				pages.Content = append(pages.Content, scalar(page.URL))
				continue
			}
			pages.Content = append(pages.Content, mapping(
				pair{"url", scalar(page.URL)}, pair{"title", text(page.Title)},
				pair{"description", text(page.Description)}, pair{"refresh_hours", number(page.RefreshHours)}))
		}
		add(KnowledgeURLsFile, encode(pages))
	}
	if len(directory.Simulations) > 0 {
		simulations := sequence()
		for _, simulation := range directory.Simulations {
			simulations.Content = append(simulations.Content, mapping(
				pair{"name", scalar(simulation.Name)}, pair{"scenario", scalar(simulation.Scenario)},
				pair{"assertion", scalar(simulation.Assertion)}, pair{"mode", text(simulation.Mode)},
				pair{"variations", number(simulation.Variations)}, pair{"max_turns", number(simulation.MaxTurns)},
				pair{"caller_target", text(simulation.CallerTarget)}, pair{"judge_target", text(simulation.JudgeTarget)},
				pair{"caller_stt", text(simulation.CallerSTT)}, pair{"caller_tts", text(simulation.CallerTTS)},
				pair{"caller_voice", text(simulation.CallerVoice)}, pair{"tags", labels(simulation.Tags)}))
		}
		add(SimulationsDir+"/simulations.yaml", encode(simulations))
	}
	slices.SortFunc(files, func(a, b File) int { return strings.Compare(a.Path, b.Path) })
	return files, unwritten
}

// skillFolderName is a skill name that is also a folder name on every file system.
var skillFolderName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$`)

// skillFile writes a SKILL.md. Its frontmatter is read a line at a time, as a key and its
// value, so a value is written on one line, in quotes when it would otherwise read back as
// something else. The instructions follow the closing --- as they are.
func skillFile(skill Skill, named bool) string {
	var file strings.Builder
	file.WriteString("---\n")
	if named {
		file.WriteString("name: " + frontmatterValue(skill.Name) + "\n")
	}
	file.WriteString("description: " + frontmatterValue(skill.Description) + "\n")
	if skill.Deadline > 0 {
		file.WriteString("deadline: " + Duration(skill.Deadline) + "\n")
	}
	if skill.CaptureVideo {
		file.WriteString("capture_video: true\n")
	}
	file.WriteString("---\n" + ended(skill.Instructions))
	return file.String()
}

// frontmatterValue is a value as unquoted reads it back: quoted when it has space at an end,
// or a pair of quotes around it that would be taken off.
func frontmatterValue(value string) string {
	value = strings.NewReplacer("\r\n", " ", "\n", " ", "\r", " ").Replace(value)
	if value != strings.TrimSpace(value) || unquoted(value) != value {
		return `"` + value + `"`
	}
	return value
}

func oneLine(text string) string {
	return strings.Join(strings.Fields(text), " ")
}

// Duration writes a duration the way a person would: 30m, not 30m0s.
func Duration(duration time.Duration) string {
	written := duration.String()
	if strings.HasSuffix(written, "m0s") {
		written = strings.TrimSuffix(written, "0s")
	}
	if strings.HasSuffix(written, "h0m") {
		written = strings.TrimSuffix(written, "0m")
	}
	return written
}

func settingsNode(s Settings) *yaml.Node {
	root := mapping(
		pair{"name", text(s.Name)}, pair{"mode", text(s.Mode)}, pair{"region", text(s.Region)},
		pair{"stt", text(s.STT)}, pair{"tts", text(s.TTS)}, pair{"sts", optional(s.STS)},
		pair{"voice", text(s.Voice)}, pair{"llm", text(s.LLM)}, pair{"harness", text(s.Harness)},
		pair{"role", text(s.Role)}, pair{"subagent", text(s.Subagent)}, pair{"search", text(s.Search)},
		pair{"image", optional(s.Image)},
	)
	set := func(key string, value *yaml.Node) {
		if value != nil {
			root.Content = append(root.Content, scalar(key), value)
		}
	}
	if s.Greeting != nil {
		set("greeting", mapping(pair{"text", scalar(s.Greeting.Text)}, pair{"mode", text(s.Greeting.Mode)}))
	}
	set("sandbox", text(s.Sandbox))
	if options := s.SandboxOptions; options != nil {
		var timeout *yaml.Node
		if options.Timeout > 0 {
			timeout = scalar(Duration(options.Timeout))
		}
		set("sandbox_options", filled(mapping(
			pair{"image", text(options.Image)}, pair{"setup", list(options.Setup)}, pair{"timeout", timeout},
			pair{"cpu", number(options.CPU)}, pair{"memory_gb", number(options.MemoryGB)},
			pair{"disk_gb", number(options.DiskGB)}, pair{"network", flag(options.Network)},
			pair{"allowed_domains", list(options.AllowedDomains)},
			pair{"network_allow_list", list(options.NetworkAllowList)})))
	}
	if s.Connectors != nil {
		connectors := sequence()
		for _, connector := range *s.Connectors {
			entry := mapping(
				pair{"name", scalar(connector.Name)}, pair{"connector", text(connector.Connector)},
				pair{"tools", optionalList(connector.Tools)}, pair{"connection", optional(connector.Connection)},
				pair{"required", optionalFlag(connector.Required)}, pair{"sandbox", flag(connector.Sandbox)},
				pair{"require_approval", optionalList(connector.RequireApproval)},
				pair{"loading", text(connector.Loading)}, pair{"output_retention", text(connector.OutputRetention)})
			if len(entry.Content) == 2 {
				entry = scalar(connector.Name)
			}
			connectors.Content = append(connectors.Content, entry)
		}
		set("connectors", connectors)
	}
	if s.Tools != nil {
		groups := sequence()
		for _, group := range *s.Tools {
			groups.Content = append(groups.Content, mapping(
				pair{"name", scalar(group.Name)}, pair{"loading", text(group.Loading)},
				pair{"tools", list(group.Tools)}, pair{"output_retention", text(group.OutputRetention)}))
		}
		set("tools", groups)
	}
	if channels := s.Channels; channels != nil {
		line := func(number string) *yaml.Node { return filled(mapping(pair{"number", text(number)})) }
		set("channels", filled(mapping(
			pair{"whatsapp", line(channels.WhatsApp)}, pair{"sms", line(channels.SMS)},
			pair{"imessage", line(channels.IMessage)}, pair{"identity", text(channels.Identity)})))
	}
	if s.ChatChannels != nil {
		channels := sequence()
		for _, channel := range *s.ChatChannels {
			channels.Content = append(channels.Content, mapping(pair{"type", scalar(channel.Type)}, pair{"answer", text(channel.Answer)}))
		}
		set("chat_channels", channels)
	}
	if scheduling := s.Scheduling; scheduling != nil {
		var maximum *yaml.Node
		if scheduling.MaxTasksPerUser != nil {
			maximum = scalar(*scheduling.MaxTasksPerUser)
		}
		set("scheduling", filled(mapping(
			pair{"enabled", optionalFlag(scheduling.Enabled)}, pair{"max_tasks_per_user", maximum},
			pair{"default_timezone", text(scheduling.DefaultTimezone)})))
	}
	if calls := s.Calls; calls != nil {
		var destinations *yaml.Node
		if len(calls.Transfer.Destinations) > 0 {
			destinations = sequence()
			for _, destination := range calls.Transfer.Destinations {
				destinations.Content = append(destinations.Content, mapping(
					pair{"name", scalar(destination.Name)}, pair{"number", scalar(destination.Number)},
					pair{"when", text(destination.When)}, pair{"warm", flag(destination.Warm)}))
			}
		}
		set("calls", filled(mapping(
			pair{"end_call", filled(mapping(pair{"enabled", optionalFlag(calls.EndCall.Enabled)}, pair{"when", text(calls.EndCall.When)}))},
			pair{"press", filled(mapping(pair{"enabled", optionalFlag(calls.Press.Enabled)}))},
			pair{"transfer", filled(mapping(pair{"enabled", optionalFlag(calls.Transfer.Enabled)}, pair{"destinations", destinations}))})))
	}
	if floor := s.FloorControl; floor != nil {
		var wait *yaml.Node
		if floor.MaxWaitMs != nil {
			wait = scalar(*floor.MaxWaitMs)
		}
		set("floor_control", mapping(pair{"model", optional(floor.Model)}, pair{"max_wait_ms", wait}))
	}
	if s.Agents != nil {
		agents := sequence()
		for _, agent := range *s.Agents {
			agents.Content = append(agents.Content, mapping(
				pair{"config", scalar(agent.Config)}, pair{"tool", scalar(agent.Tool)},
				pair{"description", text(agent.Description)}, pair{"timeout_ms", number(agent.TimeoutMs)},
				pair{"max_tool_rounds", number(agent.MaxToolRounds)}))
		}
		set("agents", agents)
	}
	set("keyterms", list(s.Keyterms))
	set("tags", labels(s.Tags))
	if video := s.Video; video != nil {
		frames := video.MaxFrames
		if frames == 1 {
			frames = 0
		}
		set("video", filled(mapping(pair{"source", text(video.Source)}, pair{"max_frames", number(frames)})))
	}
	if dispatch := s.Dispatch; dispatch != nil {
		set("dispatch", filled(mapping(pair{"incoming_call", text(dispatch.IncomingCall)}, pair{"text", text(dispatch.Text)})))
	}
	if ui := s.UI; ui != nil {
		set("ui", mapping(
			pair{"format", text(ui.Format)}, pair{"keep", list(ui.Keep)}, pair{"patch_mode", flag(ui.PatchMode)},
			pair{"voice_ui", flag(ui.VoiceUI)}, pair{"rules", list(ui.Rules)}))
	}
	return root
}

// pair is one key of a mapping being written. A nil value leaves the key out.
type pair struct {
	key   string
	value *yaml.Node
}

func mapping(pairs ...pair) *yaml.Node {
	node := &yaml.Node{Kind: yaml.MappingNode}
	for _, pair := range pairs {
		if pair.value != nil {
			node.Content = append(node.Content, scalar(pair.key), pair.value)
		}
	}
	return node
}

// filled is a mapping that says something, or nil for one with no keys.
func filled(node *yaml.Node) *yaml.Node {
	if len(node.Content) == 0 {
		return nil
	}
	return node
}

func sequence() *yaml.Node {
	return &yaml.Node{Kind: yaml.SequenceNode}
}

// scalar is a value as YAML writes it, quoted where it would otherwise read as something else.
func scalar(value any) *yaml.Node {
	node := &yaml.Node{}
	if err := node.Encode(value); err != nil {
		panic(fmt.Sprintf("agentdir: encoding %v: %v", value, err))
	}
	return node
}

func text(value string) *yaml.Node {
	if value == "" {
		return nil
	}
	return scalar(value)
}

func optional(value *string) *yaml.Node {
	if value == nil {
		return nil
	}
	return scalar(*value)
}

func number(value int) *yaml.Node {
	if value == 0 {
		return nil
	}
	return scalar(value)
}

func flag(value bool) *yaml.Node {
	if !value {
		return nil
	}
	return scalar(value)
}

func optionalFlag(value *bool) *yaml.Node {
	if value == nil {
		return nil
	}
	return scalar(*value)
}

func list(values []string) *yaml.Node {
	if len(values) == 0 {
		return nil
	}
	return optionalList(&values)
}

func optionalList(values *[]string) *yaml.Node {
	if values == nil {
		return nil
	}
	node := sequence()
	for _, value := range *values {
		node.Content = append(node.Content, scalar(value))
	}
	return node
}

func labels(values map[string]string) *yaml.Node {
	if len(values) == 0 {
		return nil
	}
	node := &yaml.Node{Kind: yaml.MappingNode}
	for _, key := range slices.Sorted(maps.Keys(values)) {
		node.Content = append(node.Content, scalar(key), scalar(values[key]))
	}
	return node
}

func encode(node *yaml.Node) string {
	var written bytes.Buffer
	encoder := yaml.NewEncoder(&written)
	encoder.SetIndent(2)
	if err := encoder.Encode(node); err != nil {
		panic(fmt.Sprintf("agentdir: writing yaml: %v", err))
	}
	return written.String()
}

// ended is text with the one newline a file ends with, or empty for none.
func ended(text string) string {
	if text == "" {
		return ""
	}
	return text + "\n"
}
