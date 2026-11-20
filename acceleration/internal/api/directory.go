package api

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agentdir"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// maxDirectoryBody is the body of a sync of a directory's files: twice what the files may
// hold, since JSON escapes their line breaks and quotes.
const maxDirectoryBody = 2 * agentdir.MaxBytes

// DirectoryFile is one file of an agent directory.
type DirectoryFile struct {
	Path    string `json:"path" minLength:"1" maxLength:"512" example:"skills/refunds/SKILL.md" doc:"Where the file is in the directory, with forward slashes: agent.yaml, instructions.md, guardrail.md, skills/<name>.md or skills/<name>/SKILL.md with the files beside it, knowledge/<file>, knowledge/urls.yaml, simulations/<file>.yaml, ui/components.json, ui/blueprints.yaml."`
	Content string `json:"content" doc:"The file, whole, as text."`
}

func (*DirectoryFile) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One file of an agent directory, as a sync takes it and an export writes it."
	return schema
}

// DirectoryProblem is one thing wrong with an agent directory, at its place in a file.
type DirectoryProblem struct {
	File    string `json:"file" example:"agent.yaml" doc:"The file's path in the directory."`
	Line    int    `json:"line,omitempty" example:"12" doc:"The line, counted from 1. Left out when the problem is the file as a whole."`
	Column  int    `json:"column,omitempty" example:"3" doc:"The column, counted from 1. Left out with line."`
	Message string `json:"message"`
}

func (*DirectoryProblem) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One thing wrong with an agent directory, at its place in a file, so an editor can point at it."
	return schema
}

// A refusal names at most maxProblems problems, and its message the first maxProblemsSaid.
const (
	maxProblems     = 100
	maxProblemsSaid = 5
)

// directoryRefusal is a directory refused for what its files say, with every problem at its
// place.
type directoryRefusal struct {
	APIError
	problems []DirectoryProblem
	// omitted is how many more problems there are than the ones named.
	omitted int
}

func (d directoryRefusal) MarshalJSON() ([]byte, error) {
	detail := d.detail()
	detail.Problems, detail.ProblemsOmitted = d.problems, d.omitted
	return marshalFailure(detail)
}

// invalidDirectory is the refusal of a directory that does not read. Any other error is
// returned as it is.
func invalidDirectory(err error) error {
	var problems agentdir.Problems
	if !errors.As(err, &problems) {
		return err
	}
	// Bounded, whatever was sent: the first problems are what somebody fixes first.
	said := problems[:min(len(problems), maxProblemsSaid)]
	message := said.Error()
	if more := len(problems) - len(said); more > 0 {
		message += fmt.Sprintf("; and %d more", more)
	}
	refusal := directoryRefusal{APIError: invalidRequest(message)}
	refusal.Code = codeInvalidAgentDirectory
	for _, problem := range problems[:min(len(problems), maxProblems)] {
		refusal.problems = append(refusal.problems, DirectoryProblem(problem))
	}
	refusal.omitted = max(len(problems)-maxProblems, 0)
	return refusal
}

// SyncAgentFilesRequest is an agent directory as its files, with what a program sets beside it.
type SyncAgentFilesRequest struct {
	Name         string          `json:"name,omitempty" doc:"What the config is called. It wins over agent.yaml's name, and may be left out when agent.yaml names the agent."`
	Files        []DirectoryFile `json:"files" nullable:"false" doc:"The files of the agent's folder, as they are on disk. A client sends what it finds and the router decides what is the directory: agent.yaml, instructions.md, guardrail.md, what is under skills/, knowledge/ and simulations/, and the UI files agent.yaml names. A file or folder whose name starts with a dot, and anything under __pycache__ or node_modules, is skipped. Any other file is left out with a warning, and is no part of the limits or the hash."`
	CheckChanges *bool           `json:"check_changes,omitempty" doc:"As check_changes of POST /v1/agents/sync: refuse the sync, with unsynced_changes, when it would write over a setting somebody changed since the last one."`
	BaseChange   *string         `json:"base_change,omitempty" doc:"As base_change of POST /v1/agents/sync: the newest change the caller has already seen."`
	DryRun       *bool           `json:"dry_run,omitempty" doc:"Work out what the sync would store and write nothing. The answer is the config as it would be stored, one a dry run would create having no id, with the warnings a sync would give, and the request is refused as the sync would be: for what the files say, for what the router cannot run, and with check_changes for what it would write over, the money for writing its knowledge included. What only the writes can tell is not told: whether the knowledge provider takes the passages, and whether a page of knowledge/urls.yaml can be read. unchanged is true when the hash matches and a sync would do nothing."`
	// What follows is what a program sets in code beside the directory, which wins over the files.
	Instructions *string            `json:"instructions,omitempty" doc:"The system prompt a program gives in code, in place of instructions.md."`
	Guardrail    *string            `json:"guardrail,omitempty" doc:"The guardrail a program gives in code, in place of guardrail.md."`
	Skills       *[]SkillRequest    `json:"skills,omitempty" doc:"The skills a program writes in code, in place of every skill under skills/. An empty list names none."`
	Harness      *Harness           `json:"harness,omitempty" doc:"In place of agent.yaml's harness."`
	Subagent     *string            `json:"subagent,omitempty" doc:"In place of agent.yaml's subagent."`
	Sandbox      *Sandbox           `json:"sandbox,omitempty" doc:"In place of agent.yaml's sandbox."`
	Tags         *map[string]string `json:"tags,omitempty" doc:"Labels added to agent.yaml's tags, winning where both name a key: what an SDK's cost tracking writes."`
	Llm          *string            `json:"llm,omitempty" doc:"In place of agent.yaml's llm."`
	Stt          *string            `json:"stt,omitempty" doc:"In place of agent.yaml's stt."`
	Tts          *string            `json:"tts,omitempty" doc:"In place of agent.yaml's tts."`
	Sts          *string            `json:"sts,omitempty" doc:"In place of agent.yaml's sts."`
	Voice        *string            `json:"voice,omitempty" doc:"In place of agent.yaml's voice."`
	Greeting     *Greeting          `json:"greeting,omitempty" doc:"In place of agent.yaml's greeting."`
	Video        *SessionVideo      `json:"video,omitempty" doc:"In place of agent.yaml's video."`
}

func (*SyncAgentFilesRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "An agent directory as its files, which the router reads, so every client's " +
		"directory means the same thing. The " +
		"fields from instructions down are what a program sets in code beside the directory: each wins " +
		"over what the files say, and is part of the hash.\n\n" +
		fmt.Sprintf("A request sends at most %d files, and the directory among them holds at most %d bytes "+
			"a file and %d bytes in all.\n\n", agentdir.MaxFiles, agentdir.MaxFileBytes, agentdir.MaxBytes) +
		"The hash of a sync is the router's own, answered as the config's sync_hash: the SHA-256, in hex, " +
		"of a run of parts, each written as its length in bytes in decimal, a colon and the part. The " +
		"parts are, for every file of the directory in path order by byte, its path and its content; " +
		"then, for each field from instructions down that is sent, its name and its value, in this " +
		"order: instructions, guardrail, harness, subagent, sandbox, llm, stt, tts, sts and voice as " +
		"sent; greeting.text and greeting.mode; video.source and video.max_frames in decimal; tags.<key> " +
		"for each tag in key order; skills with the number of skills in decimal, then for each skill in " +
		"name order skills.<name>.description, .instructions, .deadline_ms in decimal, .capture_video as " +
		"true or false, and .files.<path> for each file in path order; then, for a directory whose " +
		"agent.yaml has a ui key, ui and the version of the router's UI compiler, so a router that " +
		"compiles ui differently syncs the directory again. A file sent that is no part of the " +
		"directory is no part of the hash, so every client's sync of one folder has one hash. While a " +
		"connector is left out sync_hash is another hash, so the sync that finds its connection is " +
		"not taken for the one that left it out.\n\n" +
		"What a sync does with a setting agent.yaml leaves out depends on the setting. A model, the " +
		"voice, the mode, the region, a greeting, the sandbox, connectors, tools, channels, " +
		"chat_channels, scheduling, floor_control, agents, video and dispatch are left as stored. " +
		"calls is always written whole: a call tool the file does not name goes back to its default. " +
		"keyterms and tags are left as stored when the file has none, an empty list or mapping " +
		"included, so a file cannot clear them. instructions, guardrail, the skills the config names, " +
		"its UI instructions and its knowledge base are the directory's: a file taken out is taken " +
		"out of the agent. simulations are left alone without a simulations file, and are exactly " +
		"what the files declare with one."
	schema.Properties["files"].MaxItems = new(agentdir.MaxFiles)
	return schema
}

// keptUISource is what a config's UI instructions were compiled from, for as long as they are
// those instructions: once somebody writes others, the files no longer say what the agent is
// told, and an export says so instead of writing them back.
func keptUISource(stored store.AgentConfig, instructions string) *store.UISource {
	if instructions != stored.UIInstructions {
		return nil
	}
	return stored.UISource
}

// directorySync is a sync and what no field of its request carries. A sync of a directory's
// files has all of it; one a client sent as fields has the request alone.
type directorySync struct {
	body SyncAgentRequest
	// directory is what the files said, for a sync of files: where a refusal points.
	directory *agentdir.Directory
	// ui is what the UI instructions were compiled from, which an export writes back.
	ui       *store.UISource
	warnings []string
	dryRun   bool
}

// readDirectory reads the files a sync sent into the request they amount to. The router
// reads them, rather than each client, so every client's directory means the same thing.
func (s *Server) readDirectory(ctx context.Context, customerID string, sent SyncAgentFilesRequest) (directorySync, error) {
	files := make([]agentdir.File, 0, len(sent.Files))
	for _, file := range sent.Files {
		files = append(files, agentdir.File(file))
	}
	// Everything wrong is said at once: what the files say that does not read, and then what
	// the settings that did read may not be.
	directory, err := agentdir.Parse(files)
	var problems agentdir.Problems
	if !errors.As(err, &problems) && err != nil {
		return directorySync{}, err
	}
	if directory == nil {
		return directorySync{}, invalidDirectory(problems)
	}

	read := directorySync{directory: directory, dryRun: value(sent.DryRun), body: SyncAgentRequest{
		Name:         cmp.Or(strings.TrimSpace(sent.Name), directory.Settings.Name),
		CheckChanges: sent.CheckChanges, BaseChange: sent.BaseChange,
	}}
	if read.body.Name == "" {
		problems = append(problems, agentdir.Problem{File: agentdir.AgentFile, Line: 1, Column: 1,
			Message: "the agent has no name: add name to agent.yaml, or send name with the files"})
	}
	declareDirectory(&read.body, directory)
	if ui := directory.UI; ui != nil {
		read.body.UiInstructions = optional(ui.Instructions)
		read.ui = &store.UISource{
			Format: ui.Settings.Format, Keep: ui.Settings.Keep, PatchMode: ui.Settings.PatchMode,
			VoiceUI: ui.Settings.VoiceUI, Rules: ui.Settings.Rules,
			Components: ui.Components, Blueprints: ui.Blueprints,
		}
	}
	written := writtenInCode(&read.body, sent)
	unbound, err := s.bindDirectoryConnectors(ctx, customerID, directory, &read)
	if err != nil {
		return directorySync{}, err
	}
	problems = append(append(problems, unbound...), s.schemaProblems(directory, read.body, problems)...)
	if len(problems) > 0 {
		return directorySync{}, invalidDirectory(problems)
	}

	beside := written
	if directory.UI != nil {
		beside = append(beside, "ui", agentdir.UIVersion)
	}
	// Another hash while a connector is left out, so the sync that finds its connection is not
	// taken for this one and written.
	for _, connector := range value(read.body.LeftOutConnectors) {
		beside = append(beside, "left out", connector.Name, connector.ConnectorId, connector.Connection)
	}
	read.body.Hash = agentdir.Hash(directory.Files, beside...)
	if ignored := directory.Ignored; len(ignored) > 0 {
		named := strings.Join(ignored[:min(len(ignored), 3)], ", ")
		if len(ignored) > 3 {
			named += fmt.Sprintf(" and %d more", len(ignored)-3)
		}
		read.warnings = append(read.warnings, fmt.Sprintf(
			"%d files sent are not part of an agent directory, so they were left out: %s", len(ignored), named))
	}
	return read, nil
}

// writtenInCode puts what a program set in code beside the directory over what its files
// say, and returns each as the hash takes it: its name and then its value, in the order
// SyncAgentFilesRequest documents.
func writtenInCode(body *SyncAgentRequest, sent SyncAgentFilesRequest) []string {
	var written []string
	said := func(name, text string) { written = append(written, name, text) }
	text := func(name string, into **string, sent *string) {
		if sent != nil {
			*into = sent
			said(name, *sent)
		}
	}
	text("instructions", &body.Instructions, sent.Instructions)
	text("guardrail", &body.Guardrail, sent.Guardrail)
	if sent.Harness != nil {
		body.Harness = sent.Harness
		said("harness", string(*sent.Harness))
	}
	text("subagent", &body.Subagent, sent.Subagent)
	if sent.Sandbox != nil {
		body.Sandbox = sent.Sandbox
		said("sandbox", string(*sent.Sandbox))
	}
	text("llm", &body.Llm, sent.Llm)
	text("stt", &body.Stt, sent.Stt)
	text("tts", &body.Tts, sent.Tts)
	text("sts", &body.Sts, sent.Sts)
	text("voice", &body.Voice, sent.Voice)
	if greeting := sent.Greeting; greeting != nil {
		body.Greeting = greeting
		said("greeting.text", greeting.Text)
		if greeting.Mode != nil {
			said("greeting.mode", string(*greeting.Mode))
		}
	}
	if video := sent.Video; video != nil {
		body.Video = video
		if video.Source != nil {
			said("video.source", *video.Source)
		}
		if video.MaxFrames != nil {
			said("video.max_frames", strconv.Itoa(*video.MaxFrames))
		}
	}
	if sent.Tags != nil {
		tags := maps.Clone(value(body.Tags))
		if tags == nil {
			tags = map[string]string{}
		}
		for _, key := range slices.Sorted(maps.Keys(*sent.Tags)) {
			tags[key] = (*sent.Tags)[key]
			said("tags."+key, tags[key])
		}
		body.Tags = &tags
	}
	if sent.Skills != nil {
		skills := slices.SortedFunc(slices.Values(*sent.Skills), func(a, b SkillRequest) int { return strings.Compare(a.Name, b.Name) })
		body.Skills = nil
		if len(skills) > 0 {
			body.Skills = sent.Skills
		}
		said("skills", strconv.Itoa(len(skills)))
		for _, skill := range skills {
			at := "skills." + skill.Name + "."
			said(at+"description", skill.Description)
			said(at+"instructions", skill.Instructions)
			if skill.DeadlineMs != nil {
				said(at+"deadline_ms", strconv.FormatInt(*skill.DeadlineMs, 10))
			}
			if skill.CaptureVideo != nil {
				said(at+"capture_video", strconv.FormatBool(*skill.CaptureVideo))
			}
			for _, file := range slices.Sorted(maps.Keys(value(skill.Files))) {
				said(at+"files."+file, (*skill.Files)[file])
			}
		}
	}
	return written
}

// declareDirectory carries what a directory says onto a sync request. Only what its files
// name is set, so the sync leaves whatever is stored for the rest.
func declareDirectory(body *SyncAgentRequest, directory *agentdir.Directory) {
	settings := directory.Settings
	body.Instructions = optional(directory.Instructions)
	body.Guardrail = optional(directory.Guardrail)
	if len(directory.Skills) > 0 {
		skills := make([]SkillRequest, 0, len(directory.Skills))
		for _, skill := range directory.Skills {
			declared := SkillRequest{
				Name: skill.Name, Description: skill.Description, Instructions: skill.Instructions,
				CaptureVideo: &skill.CaptureVideo,
			}
			if skill.Deadline > 0 {
				declared.DeadlineMs = new(skill.Deadline.Milliseconds())
			}
			if len(skill.Files) > 0 {
				declared.Files = &skill.Files
			}
			skills = append(skills, declared)
		}
		body.Skills = &skills
	}
	if len(directory.Knowledge) > 0 {
		documents := make([]KnowledgeDocument, 0, len(directory.Knowledge))
		for _, document := range directory.Knowledge {
			documents = append(documents, KnowledgeDocument(document))
		}
		body.Knowledge = &documents
	}
	if len(directory.KnowledgeURLs) > 0 {
		pages := make([]KnowledgeUrlDeclaration, 0, len(directory.KnowledgeURLs))
		for _, page := range directory.KnowledgeURLs {
			declared := KnowledgeUrlDeclaration{Url: page.URL, Title: optional(page.Title), Description: optional(page.Description)}
			if page.RefreshHours > 0 {
				declared.RefreshHours = &page.RefreshHours
			}
			pages = append(pages, declared)
		}
		body.KnowledgeUrls = &pages
	}
	if directory.Simulations != nil {
		simulations := make([]SimulationDeclaration, 0, len(directory.Simulations))
		for _, simulation := range directory.Simulations {
			declared := SimulationDeclaration{
				Name: simulation.Name, Scenario: simulation.Scenario, Assertion: simulation.Assertion,
				Mode:         optional(simulation.Mode),
				Variations:   positive(simulation.Variations),
				MaxTurns:     positive(simulation.MaxTurns),
				CallerTarget: optional(simulation.CallerTarget), JudgeTarget: optional(simulation.JudgeTarget),
				CallerStt: optional(simulation.CallerSTT), CallerTts: optional(simulation.CallerTTS),
				CallerVoice: optional(simulation.CallerVoice),
			}
			if len(simulation.Tags) > 0 {
				declared.Tags = &simulation.Tags
			}
			simulations = append(simulations, declared)
		}
		body.Simulations = &simulations
	}

	body.Role = named[AgentRole](settings.Role)
	body.Mode = named[AgentMode](settings.Mode)
	body.Region = optional(settings.Region)
	body.Stt = optional(settings.STT)
	body.Tts = optional(settings.TTS)
	body.Sts = settings.STS
	body.Voice = optional(settings.Voice)
	body.Llm = optional(settings.LLM)
	body.Harness = named[Harness](settings.Harness)
	body.Subagent = optional(settings.Subagent)
	body.Search = optional(settings.Search)
	// Sent even empty, since that is how a file turns drawing off.
	body.Image = settings.Image
	if settings.Greeting != nil {
		body.Greeting = &Greeting{Text: settings.Greeting.Text, Mode: named[GreetingMode](settings.Greeting.Mode)}
	}
	body.Sandbox = named[Sandbox](settings.Sandbox)
	if options := settings.SandboxOptions; options != nil {
		body.SandboxOptions = &SandboxOptions{
			Image: &options.Image, Setup: &options.Setup, TimeoutMs: new(int(options.Timeout.Milliseconds())),
			Cpu: &options.CPU, MemoryGb: &options.MemoryGB, DiskGb: &options.DiskGB, Network: &options.Network,
			AllowedDomains: &options.AllowedDomains, NetworkAllowList: &options.NetworkAllowList,
		}
	}
	if settings.Tools != nil {
		groups := make([]ToolGroup, 0, len(*settings.Tools))
		for _, group := range *settings.Tools {
			groups = append(groups, ToolGroup{
				Name: group.Name, Tools: group.Tools,
				Loading: named[ToolLoading](group.Loading), OutputRetention: optional(group.OutputRetention),
			})
		}
		body.Tools = &groups
	}
	if channels := settings.Channels; channels != nil {
		line := func(number string) *ChannelLineRequest {
			if number == "" {
				return nil
			}
			return &ChannelLineRequest{Number: number}
		}
		body.Channels = &AgentChannels{
			Whatsapp: line(channels.WhatsApp), Sms: line(channels.SMS), Imessage: line(channels.IMessage),
			Identity: named[ChannelIdentity](channels.Identity),
		}
	}
	if settings.ChatChannels != nil {
		channels := make([]AgentChatChannel, 0, len(*settings.ChatChannels))
		for _, channel := range *settings.ChatChannels {
			channels = append(channels, AgentChatChannel{Type: channel.Type, Answer: named[ChatChannelAnswer](channel.Answer)})
		}
		body.ChatChannels = &channels
	}
	if scheduling := settings.Scheduling; scheduling != nil {
		body.Scheduling = &AgentScheduling{
			Enabled: scheduling.Enabled, MaxTasksPerUser: scheduling.MaxTasksPerUser,
			DefaultTimezone: optional(scheduling.DefaultTimezone),
		}
	}
	body.Calls = declaredCalls(settings.Calls)
	if floor := settings.FloorControl; floor != nil {
		body.FloorControl = &FloorControl{Model: floor.Model, MaxWaitMs: floor.MaxWaitMs}
	}
	if len(settings.Keyterms) > 0 {
		body.Keyterms = &settings.Keyterms
	}
	if len(settings.Tags) > 0 {
		body.Tags = &settings.Tags
	}
	if settings.Agents != nil {
		agents := make([]ChildAgent, 0, len(*settings.Agents))
		for _, agent := range *settings.Agents {
			declared := ChildAgent{
				Config: agent.Config, Tool: agent.Tool, Description: optional(agent.Description),
				TimeoutMs: positive(agent.TimeoutMs),
			}
			if agent.MaxToolRounds != 0 {
				declared.MaxToolRounds = &agent.MaxToolRounds
			}
			agents = append(agents, declared)
		}
		body.Agents = &agents
	}
	if video := settings.Video; video != nil {
		body.Video = &SessionVideo{MaxFrames: &video.MaxFrames, Source: optional(video.Source)}
	}
	if dispatch := settings.Dispatch; dispatch != nil {
		body.Dispatch = &AgentDispatch{
			IncomingCall: named[DispatchSetting](dispatch.IncomingCall), Text: named[DispatchSetting](dispatch.Text),
		}
	}
}

// declaredCalls always fills the whole block, with the default for what the file leaves
// out, so a sync replaces what the dashboard set.
func declaredCalls(calls *agentdir.Calls) *AgentCalls {
	if calls == nil {
		calls = &agentdir.Calls{}
	}
	destinations := make([]TransferDestination, 0, len(calls.Transfer.Destinations))
	for _, destination := range calls.Transfer.Destinations {
		destinations = append(destinations, TransferDestination(destination))
	}
	return &AgentCalls{
		EndCall:  &EndCallSettings{Enabled: new(value(calls.EndCall.Enabled)), When: calls.EndCall.When},
		Press:    &PressSettings{Enabled: new(calls.Press.Enabled == nil || *calls.Press.Enabled)},
		Transfer: &TransferSettings{Enabled: new(calls.Transfer.Enabled == nil || *calls.Transfer.Enabled), Destinations: destinations},
	}
}

// bindDirectoryConnectors turns the connectors agent.yaml names into the bindings a sync
// stores, finding each connection it names among the app's own by id or by label.
//
// An entry whose connection the app does not have is left out and recorded, so the rest of the
// agent still syncs, unless the entry is required: a session needs that one.
func (s *Server) bindDirectoryConnectors(
	ctx context.Context, customerID string, directory *agentdir.Directory, read *directorySync,
) (agentdir.Problems, error) {
	if directory.Settings.Connectors == nil {
		return nil, nil
	}
	var problems agentdir.Problems
	bindings := []AgentConnectorBinding{}
	var leftOut []LeftOutConnector
	for index, entry := range *directory.Settings.Connectors {
		fail := func(format string, args ...any) {
			line, column := directory.Locate(fmt.Sprintf("connectors[%d]", index))
			problems = append(problems, agentdir.Problem{
				File: agentdir.AgentFile, Line: line, Column: column,
				Message: fmt.Sprintf("connector %q: ", entry.Name) + fmt.Sprintf(format, args...),
			})
		}
		binding := AgentConnectorBinding{
			Name:            entry.Name,
			ConnectorId:     cmp.Or(entry.Connector, entry.Name),
			Connection:      AgentConnectorSelection{Type: AgentConnectorSelectionTypeSession},
			Tools:           []ConnectorToolGrant{},
			ToolPatterns:    &[]string{"*"},
			Required:        entry.Required,
			Sandbox:         entry.Sandbox,
			RequireApproval: value(entry.RequireApproval),
			Loading:         named[ToolLoading](entry.Loading),
			OutputRetention: optional(entry.OutputRetention),
		}
		if entry.Tools != nil {
			binding.ToolPatterns = entry.Tools
		}
		if entry.Connection != nil {
			named, err := s.store.AppConnectionsNamed(ctx, customerID, binding.ConnectorId, *entry.Connection)
			if err != nil {
				return nil, err
			}
			// An id wins over a label, and a label names a connection only when one has it.
			if byID := slices.IndexFunc(named, func(c store.ConnectorConnection) bool { return c.ID == *entry.Connection }); byID >= 0 {
				named = named[byID : byID+1]
			}
			switch {
			case len(named) > 1:
				fail("%q matches %d app connections: name one by its id", *entry.Connection, len(named))
				continue
			case len(named) == 0 && value(entry.Required):
				fail("no app connection with id or label %q: add the connection, or drop required", *entry.Connection)
				continue
			case len(named) == 0:
				leftOut = append(leftOut, LeftOutConnector{Name: entry.Name, ConnectorId: binding.ConnectorId, Connection: *entry.Connection})
				read.warnings = append(read.warnings, fmt.Sprintf(
					"connector %q was left out: the app has no connection with id or label %q", entry.Name, *entry.Connection))
				continue
			}
			binding.Connection = AgentConnectorSelection{Type: AgentConnectorSelectionTypeFixed, ConnectionId: &named[0].ID}
		}
		bindings = append(bindings, binding)
	}
	read.body.Connectors = &bindings
	if len(leftOut) > 0 {
		read.body.LeftOutConnectors = &leftOut
	}
	return problems, nil
}

// schemaProblems holds the request a directory amounts to against the schema a request sent
// as fields is held to, so both are refused for the same things. Each problem is put where
// agent.yaml says the setting, or at the files a setting was read from, and one about files
// that read already found wrong is left to what read said of them.
func (s *Server) schemaProblems(directory *agentdir.Directory, body SyncAgentRequest, read agentdir.Problems) agentdir.Problems {
	encoded, err := json.Marshal(body)
	if err != nil {
		return agentdir.Problems{{File: agentdir.AgentFile, Message: err.Error()}}
	}
	var decoded any
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		return agentdir.Problems{{File: agentdir.AgentFile, Message: err.Error()}}
	}
	result := &huma.ValidateResult{}
	huma.Validate(s.schemas, s.schemas.Schema(reflect.TypeFor[SyncAgentRequest](), true, ""),
		huma.NewPathBuffer(nil, 0), huma.ModeWriteToServer, decoded, result)

	elsewhere := map[string]string{
		"skills": agentdir.SkillsDir, "knowledge": agentdir.KnowledgeDir,
		"knowledge_urls": agentdir.KnowledgeURLsFile, "simulations": agentdir.SimulationsDir,
		"instructions": agentdir.InstructionsFile, "guardrail": agentdir.GuardrailFile,
	}
	var problems agentdir.Problems
	for _, failure := range result.Errors {
		var detail *huma.ErrorDetail
		if !errors.As(failure, &detail) {
			problems = append(problems, agentdir.Problem{File: agentdir.AgentFile, Message: failure.Error()})
			continue
		}
		problem := agentdir.Problem{File: agentdir.AgentFile, Message: detail.Location + ": " + detail.Message}
		setting, _, _ := strings.Cut(strings.NewReplacer("[", ".").Replace(detail.Location), ".")
		if file, found := elsewhere[setting]; found {
			// What a file there already had wrong with it is what the schema is refusing.
			if slices.ContainsFunc(read, func(said agentdir.Problem) bool { return strings.HasPrefix(said.File, file) }) {
				continue
			}
			problem.File = file
		} else {
			problem.Line, problem.Column = directory.Locate(detail.Location)
		}
		problems = append(problems, problem)
	}
	return problems
}

// named is text as the enum a setting is one of, or nil for none.
func named[T ~string](text string) *T {
	if text == "" {
		return nil
	}
	return new(T(text))
}

// positive is a count, or nil for none.
func positive(count int) *int {
	if count <= 0 {
		return nil
	}
	return &count
}

// AgentDirectory is an agent as the directory it is synced from.
type AgentDirectory struct {
	Name        string          `json:"name" doc:"What the agent is called, as agent.yaml names it."`
	Files       []DirectoryFile `json:"files" nullable:"false" doc:"The directory's files, in path order. Sent back to POST /v1/agents/sync/files, they store the agent as it is."`
	NotExported []string        `json:"not_exported" nullable:"false" doc:"What this agent holds that its directory cannot say and a sync of these files would change, each in a sentence: a connector binding's events, policy or timeout, a connector that grants no tools, a knowledge document that is not a text file, UI instructions no directory's files were compiled into. Empty when a sync of these files leaves the agent as it is."`
}

func (*AgentDirectory) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "An agent config as the directory it is synced from: agent.yaml, instructions.md, " +
		"guardrail.md, its skills, its knowledge, its simulations and its UI files."
	return schema
}

// agentDirectory writes a stored agent as its directory.
func (s *Server) agentDirectory(ctx context.Context, id string) (AgentDirectory, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return AgentDirectory{}, errMissingCustomer
	}
	if s.store == nil {
		return AgentDirectory{}, errNoConfigs
	}
	config, err := s.configs.AgentConfig(ctx, customerID, id)
	if err != nil {
		return AgentDirectory{}, errUnknownConfig
	}

	directory := &agentdir.Directory{Instructions: config.Instructions, Guardrail: config.Guardrail}
	exported := AgentDirectory{Name: config.Name, NotExported: []string{}}
	note := func(format string, args ...any) {
		exported.NotExported = append(exported.NotExported, fmt.Sprintf(format, args...))
	}
	directory.Settings = directorySettings(config, note)
	switch ui := config.UISource; {
	case ui != nil:
		directory.UI = &agentdir.UI{
			Settings: agentdir.UISettings{
				Format: ui.Format, Keep: ui.Keep, PatchMode: ui.PatchMode, VoiceUI: ui.VoiceUI, Rules: ui.Rules,
			},
			Components: ui.Components, Blueprints: ui.Blueprints,
		}
	case config.UIInstructions != "":
		note("The agent's UI instructions were not compiled from a directory's files, so there is no ui key to write and a sync of these files removes them.")
	}

	skills, err := s.store.CustomerSkills(ctx, customerID, config.ID)
	if err != nil {
		return AgentDirectory{}, err
	}
	slices.SortFunc(skills, func(a, b store.Skill) int { return strings.Compare(a.Name, b.Name) })
	for _, skill := range skills {
		if slices.Contains(config.Skills, skill.Name) {
			directory.Skills = append(directory.Skills, agentdir.Skill{
				Name: skill.Name, Description: skill.Description, Instructions: skill.Instructions,
				Deadline: time.Duration(skill.DeadlineMs) * time.Millisecond, CaptureVideo: skill.CaptureVideo,
				Files: skill.Files,
			})
		}
	}

	var documents []store.KnowledgeDocument
	if namespace := config.KnowledgeNamespace; namespace != "" {
		if namespace != config.Name {
			note("The agent reads the knowledge base %q. A directory's knowledge is the base named after the agent, so a sync of these files writes it there.", namespace)
		}
		if documents, err = s.store.CustomerKnowledgeDocumentsWithText(ctx, customerID, namespace); err != nil {
			return AgentDirectory{}, err
		}
		slices.SortFunc(documents, func(a, b store.KnowledgeDocument) int { return strings.Compare(a.Source, b.Source) })
		if s.pages != nil {
			pages, err := s.pages.List(ctx, customerID, namespace)
			if err != nil {
				return AgentDirectory{}, err
			}
			for _, page := range pages {
				directory.KnowledgeURLs = append(directory.KnowledgeURLs, agentdir.KnowledgeURL{
					URL: page.URL, Title: page.DeclaredTitle, Description: page.Description, RefreshHours: page.RefreshHours,
				})
			}
		}
	}

	simulations, err := s.store.CustomerSimulations(ctx, customerID)
	if err != nil {
		return AgentDirectory{}, err
	}
	slices.SortFunc(simulations, func(a, b store.Simulation) int { return strings.Compare(a.Name, b.Name) })
	for _, simulation := range simulations {
		if simulation.ConfigID != config.ID {
			continue
		}
		declared := agentdir.Simulation{
			Name: simulation.Name, Scenario: simulation.Scenario, Assertion: simulation.Assertion,
			CallerTarget: simulation.CallerTarget, JudgeTarget: simulation.JudgeTarget,
			CallerSTT: simulation.CallerSTT, CallerTTS: simulation.CallerTTS, CallerVoice: simulation.CallerVoice,
			Tags: simulation.Tags,
		}
		// What a simulation that says nothing is stored with is left out again.
		if simulation.Mode != store.SimulationText {
			declared.Mode = simulation.Mode
		}
		if simulation.Variations > 1 {
			declared.Variations = simulation.Variations
		}
		if simulation.MaxTurns != store.DefaultSimulationTurns {
			declared.MaxTurns = simulation.MaxTurns
		}
		directory.Simulations = append(directory.Simulations, declared)
	}

	// The knowledge last, held to what a sync takes back once everything else is written: a
	// document that is too large for one, or that would take the directory past what one may
	// hold, is left out and said to be.
	rendered, _ := agentdir.Render(directory)
	held := 0
	for _, file := range rendered {
		held += len(file.Content)
	}
	for _, document := range documents {
		switch {
		case !agentdir.KnowledgeFile(document.Source, document.Text):
			note("The knowledge document %q is not a text file a directory holds, so it is left out and a sync of these files removes it.", document.Source)
		case len(document.Text) > agentdir.MaxFileBytes, held+len(document.Text) > agentdir.MaxBytes,
			len(rendered)+len(directory.Knowledge) >= agentdir.MaxFiles:
			note("The knowledge document %q is more than a directory may hold, so it is left out and a sync of these files removes it.", document.Source)
		default:
			held += len(document.Text)
			directory.Knowledge = append(directory.Knowledge, agentdir.Document{Source: document.Source, Text: document.Text})
		}
	}
	if config.KnowledgeNamespace != "" && len(directory.Knowledge)+len(directory.KnowledgeURLs) == 0 {
		note("None of the agent's knowledge base %q is in these files, so a sync of them leaves the agent with no knowledge base.", config.KnowledgeNamespace)
	}
	for _, kept := range []struct{ file, text string }{
		{agentdir.InstructionsFile, config.Instructions}, {agentdir.GuardrailFile, config.Guardrail},
	} {
		if kept.text != strings.TrimSpace(kept.text) {
			note("The agent's %s starts or ends with blank space, which a directory's does not keep, so a sync of these files trims it.", kept.file)
		}
	}
	files, unwritten := agentdir.Render(directory)
	exported.NotExported = append(exported.NotExported, unwritten...)
	for _, file := range files {
		exported.Files = append(exported.Files, DirectoryFile(file))
	}
	return exported, nil
}

// directorySettings is a stored config as agent.yaml says it: what a sync fills in on its own is
// left out, and what agent.yaml cannot say and a sync would change is noted.
func directorySettings(config store.AgentConfig, note func(format string, args ...any)) agentdir.Settings {
	settings := agentdir.Settings{
		Name: config.Name, Mode: config.Mode, STT: config.STT, TTS: config.TTS, Voice: config.Voice,
		LLM: config.LLM, Role: config.Role, Subagent: config.Subagent, Search: config.Search,
		Sandbox: config.Sandbox, Keyterms: config.Keyterms, Tags: config.Tags,
	}
	if config.Region != pricing.DefaultRegion {
		settings.Region = config.Region
	}
	if config.Harness != string(Default) {
		settings.Harness = config.Harness
	}
	if config.STS != "" {
		settings.STS = &config.STS
	}
	if config.Image != "" {
		settings.Image = &config.Image
	}
	if config.Greeting != "" {
		settings.Greeting = &agentdir.Greeting{Text: config.Greeting}
		if config.GreetingMode != store.GreetingExact {
			settings.Greeting.Mode = config.GreetingMode
		}
	}
	if options := config.SandboxOptions; sandboxOptionsOf(options) != nil {
		settings.SandboxOptions = &agentdir.SandboxOptions{
			Image: options.Image, Setup: options.Setup, Timeout: time.Duration(options.TimeoutMs) * time.Millisecond,
			CPU: options.CPU, MemoryGB: options.MemoryGB, DiskGB: options.DiskGB, Network: options.Network,
			AllowedDomains: options.AllowedDomains, NetworkAllowList: options.NetworkAllowList,
		}
	}
	if len(config.Connectors)+len(config.LeftOutConnectors) > 0 {
		connectors := []agentdir.Connector{}
		for _, binding := range config.Connectors {
			if connector, ok := directoryConnector(binding, note); ok {
				connectors = append(connectors, connector)
			}
		}
		for _, leftOut := range config.LeftOutConnectors {
			connector := agentdir.Connector{Name: leftOut.Name, Connection: &leftOut.Connection}
			if leftOut.ConnectorID != leftOut.Name {
				connector.Connector = leftOut.ConnectorID
			}
			connectors = append(connectors, connector)
		}
		settings.Connectors = &connectors
	}
	if len(config.ToolGroups) > 0 {
		groups := make([]agentdir.ToolGroup, 0, len(config.ToolGroups))
		for _, group := range config.ToolGroups {
			groups = append(groups, agentdir.ToolGroup{
				Name: group.Name, Loading: group.Loading, Tools: group.Tools, OutputRetention: string(group.OutputRetention),
			})
		}
		settings.Tools = &groups
	}
	if channels := config.Channels; channels.WhatsApp != nil || channels.SMS != nil || channels.IMessage != nil || channels.Identity != "" {
		number := func(line *store.ChannelLine) string {
			if line == nil {
				return ""
			}
			return line.Number
		}
		settings.Channels = &agentdir.Channels{
			WhatsApp: number(channels.WhatsApp), SMS: number(channels.SMS), IMessage: number(channels.IMessage),
			Identity: channels.Identity,
		}
	}
	if len(config.ChatChannels) > 0 {
		channels := make([]agentdir.ChatChannel, 0, len(config.ChatChannels))
		for _, channel := range config.ChatChannels {
			channels = append(channels, agentdir.ChatChannel(channel))
		}
		settings.ChatChannels = &channels
	}
	if scheduling := config.Scheduling; scheduling != (store.AgentScheduling{}) {
		settings.Scheduling = &agentdir.Scheduling{DefaultTimezone: scheduling.DefaultTimezone}
		if scheduling.Enabled {
			settings.Scheduling.Enabled = &scheduling.Enabled
		}
		settings.Scheduling.MaxTasksPerUser = positive(scheduling.MaxTasksPerUser)
	}
	calls := agentdir.Calls{}
	if config.Calls.EndCallOn() {
		calls.EndCall.Enabled = new(true)
	}
	if config.Calls.EndCall != nil {
		calls.EndCall.When = config.Calls.EndCall.When
	}
	if !config.Calls.PressOn() {
		calls.Press.Enabled = new(false)
	}
	if !config.Calls.TransferOn() {
		calls.Transfer.Enabled = new(false)
	}
	if config.Calls.Transfer != nil {
		for _, destination := range config.Calls.Transfer.Destinations {
			calls.Transfer.Destinations = append(calls.Transfer.Destinations, agentdir.TransferDestination(destination))
		}
	}
	settings.Calls = &calls
	if floor := config.FloorControl; !floor.IsZero() {
		settings.FloorControl = &agentdir.FloorControl{Model: &floor.Model, MaxWaitMs: positive(floor.MaxWaitMs)}
	}
	if len(config.Agents) > 0 {
		agents := make([]agentdir.ChildAgent, 0, len(config.Agents))
		for _, agent := range config.Agents {
			agents = append(agents, agentdir.ChildAgent(agent))
		}
		settings.Agents = &agents
	}
	if config.VideoSource != "" || config.VideoMaxFrames > 1 {
		settings.Video = &agentdir.Video{Source: config.VideoSource, MaxFrames: max(config.VideoMaxFrames, 1)}
	}
	if config.DispatchIncomingCall || config.DispatchText {
		settings.Dispatch = &agentdir.Dispatch{}
		if config.DispatchIncomingCall {
			settings.Dispatch.IncomingCall = string(Enabled)
		}
		if config.DispatchText {
			settings.Dispatch.Text = string(Enabled)
		}
	}
	return settings
}

// directoryConnector is a stored binding as agent.yaml names it, or false for one it cannot name.
func directoryConnector(binding store.ConnectorBinding, note func(format string, args ...any)) (agentdir.Connector, bool) {
	connector := agentdir.Connector{
		Name: binding.Name, Sandbox: binding.Sandbox, Loading: binding.Loading,
		OutputRetention: string(binding.OutputRetention),
	}
	if binding.ConnectorID != binding.Name {
		connector.Connector = binding.ConnectorID
	}
	switch patterns := value(binding.ToolPatterns); {
	case len(patterns) == 1 && patterns[0] == "*":
	case len(patterns) > 0:
		connector.Tools = &patterns
	case len(binding.Tools) > 0:
		granted := make([]string, 0, len(binding.Tools))
		for _, tool := range binding.Tools {
			granted = append(granted, tool.Name)
		}
		connector.Tools = &granted
		note("The connector %s grants its tools by name and schema digest. agent.yaml names them alone, so a sync of these files grants them whatever their schema.", binding.Name)
	default:
		// Written with no tools key, a sync would grant every tool instead.
		note("The connector %s grants no tools, which agent.yaml cannot say, so it is left out and a sync of these files unbinds it.", binding.Name)
		return agentdir.Connector{}, false
	}
	if binding.Connection.Type == string(AgentConnectorSelectionTypeFixed) {
		connector.Connection = &binding.Connection.ConnectionID
	}
	if binding.Required {
		connector.Required = &binding.Required
	}
	if len(binding.RequireApproval) > 0 {
		connector.RequireApproval = &binding.RequireApproval
	}
	if binding.TimeoutMs > 0 || len(binding.Events) > 0 || binding.Policy != nil {
		note("The connector %s has a timeout, events or a policy, which agent.yaml cannot say, so a sync of these files removes them.", binding.Name)
	}
	return connector, true
}

type getAgentDirectoryRequest struct {
	Id string `path:"id"`
}

type getAgentDirectoryResponse struct {
	Body AgentDirectory
}

const directoryFiles = "The files are agent.yaml, instructions.md, guardrail.md, each skill as " +
	"skills/<name>/SKILL.md with the files beside it, the agent's knowledge base as knowledge/<source> " +
	"with the pages it is kept filled from in knowledge/urls.yaml, its simulations in " +
	"simulations/simulations.yaml, and ui/components.json and ui/blueprints.yaml when a directory's " +
	"files gave the agent its UI. agent.yaml is canonical: its keys in a fixed order, what a sync " +
	"fills in on its own left out (region us, the default harness, a call tool at its default, a " +
	"connector's tools when it grants every one), and a connector that says only its name written as " +
	"the name alone.\n\n" +
	"Sent to POST /v1/agents/sync/files, the files store the agent as it is, apart from what `not_exported` lists. What a " +
	"directory never says is not in the files and a sync leaves it alone: llm_fallbacks, review, " +
	"resume_history, visible_tools and episode_cards. A fixed connector is written with its " +
	"connection's id, whatever label agent.yaml first named it by, and a skill the config no longer " +
	"names is left out.\n\n" +
	"Server-side only: it needs a server-side token, so it cannot be reached from an end user's device."

type syncAgentFilesRequest struct {
	Body SyncAgentFilesRequest
}

// registerDirectory declares the operations served in directory.go.
func (s *Server) registerDirectory(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "getAgentDirectory",
		Method:      http.MethodGet,
		Path:        "/v1/agents/configs/{id}/directory",
		Summary:     "An agent config as the files of its directory",
		Description: "The reverse of a sync: the agent as the directory it would be synced from. " + directoryFiles,
		Responses: map[string]*huma.Response{
			"200": {Description: "The agent's directory"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, func(ctx context.Context, request *getAgentDirectoryRequest) (*getAgentDirectoryResponse, error) {
		directory, err := s.agentDirectory(ctx, request.Id)
		if err != nil {
			return nil, err
		}
		return &getAgentDirectoryResponse{Body: directory}, nil
	})
	huma.Register(api, huma.Operation{
		OperationID:     "syncAgentFiles",
		Method:          http.MethodPost,
		Path:            "/v1/agents/sync/files",
		MaxBodyBytes:    maxDirectoryBody,
		BodyReadTimeout: uploadReadTimeout,
		Summary:         "Store an agent directory sent as its files",
		Description: "POST /v1/agents/sync for a client that sends the directory's files as they are " +
			"and leaves reading them to the router, so a CLI, an SDK and the dashboard all mean the same " +
			"thing by one directory. It stores what that sync stores, with the same answer.\n\n" +
			"A key agent.yaml does not have, a retired one or a value of the wrong type is refused (400 " +
			"`invalid_agent_directory`) with every problem at its file, line and column in the error's " +
			"`problems`. A connector's connection is found among the app's own by id or by label; one the " +
			"app does not have is left out of the agent, stored in `left_out_connectors` and named in a " +
			"warning, unless the connector is required, which is refused. agent.yaml's `ui` is compiled " +
			"into `ui_instructions` here.\n\n" +
			"`dry_run` answers with what the sync would store and writes nothing, which is how to show " +
			"what an upload would change before applying it. `GET /v1/agents/configs/{id}/directory` is " +
			"the reverse: the stored agent as files.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The config as stored, as it already was, or with dry_run as it would be stored"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusConflict, http.StatusRequestEntityTooLarge},
	}, func(ctx context.Context, request *syncAgentFilesRequest) (*syncAgentResponse, error) {
		customerID, ok := CustomerFrom(ctx)
		if !ok {
			return nil, errMissingCustomer
		}
		if s.store == nil {
			return nil, errNoConfigs
		}
		read, err := s.readDirectory(ctx, customerID, request.Body)
		if err != nil {
			return nil, err
		}
		return s.sync(ctx, read)
	})
}
