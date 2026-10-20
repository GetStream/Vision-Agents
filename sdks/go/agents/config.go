package agents

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// Sync stores the agent's configuration in the backend, along with its skills and whatever
// its knowledge directory holds.
//
// A config is what a session can be created from by name, so the things worth deciding once
// are decided once. Both the config and its skills are found by name first, so calling this
// twice edits what is stored rather than storing another copy of it.
//
// An agent read from a directory is synced whole, with what its agent.yaml declares, and
// .agent_sync records the fingerprint of what was stored. A directory unchanged since then
// is only read back, not written again.
func (a *Agent) Sync(ctx context.Context) (*acceleration.AgentConfig, error) {
	client, err := a.Client()
	if err != nil {
		return nil, err
	}
	if a.folder != nil {
		return a.syncFolder(ctx, client)
	}

	skills := a.syncedSkills()
	if len(skills) > 0 {
		if err := DefineSkills(ctx, client, skills); err != nil {
			return nil, err
		}
	}

	wanted := acceleration.AgentConfigRequest{Name: a.options.Name}
	setString(&wanted.Instructions, a.options.Instructions)
	setString(&wanted.Guardrail, a.options.Guardrail)
	harness, subagent, sandbox := a.options.Harness.stored()
	wanted.Harness, wanted.Subagent, wanted.Sandbox = harness, subagent, sandbox
	if len(a.options.CostTracking) > 0 {
		tags := a.options.CostTracking
		wanted.Tags = &tags
	}
	if len(skills) > 0 {
		named := make([]string, 0, len(skills))
		for _, skill := range skills {
			named = append(named, skill.Name)
		}
		wanted.Skills = &named
	}

	return DefineAgent(ctx, client, wanted)
}

// syncFolder stores the agent's directory in one request. What the code set wins over what
// the directory says, and is part of the fingerprint, so changing either syncs again.
func (a *Agent) syncFolder(ctx context.Context, client *acceleration.ClientWithResponses) (*acceleration.AgentConfig, error) {
	folder := a.folder
	// Still sent as before, until the router stops taking them.
	if len(folder.Settings.Plugins) > 0 || len(folder.Settings.PluginEvents) > 0 {
		named := make([]string, 0, len(folder.Settings.Plugins))
		for _, plugin := range folder.Settings.Plugins {
			named = append(named, plugin.Name)
		}
		a.logger.Warn("agent.yaml plugins are deprecated: rename plugins to connectors",
			"agent", a.options.Name, "plugins", named, "plugin_events", len(folder.Settings.PluginEvents))
	}
	skills := a.syncedSkills()
	hash := fingerprint(folder.Declaration, a.options.Instructions, a.options.Guardrail,
		skills, folder.Knowledge, folder.KnowledgeURLs, folder.Simulations)
	harness, subagent, sandbox := a.options.Harness.stored()
	if harness != nil || subagent != nil || sandbox != nil || len(a.options.CostTracking) > 0 {
		hash = fingerprint(hash, fmt.Sprint(deref(harness), deref(subagent), deref(sandbox)),
			fmt.Sprint(a.options.CostTracking), nil, nil, nil, nil)
	}

	if ReadStamp(folder.Path) == hash {
		if stored, err := storedConfig(ctx, client, a.options.Name); err != nil || stored != nil {
			return stored, err
		}
	}

	body := acceleration.SyncAgentRequest{Name: a.options.Name, Hash: hash}
	setString(&body.Instructions, a.options.Instructions)
	setString(&body.Guardrail, a.options.Guardrail)
	if len(skills) > 0 {
		requests := make([]acceleration.SkillRequest, 0, len(skills))
		for _, skill := range skills {
			requests = append(requests, skillRequestOf(skill))
		}
		body.Skills = &requests
	}
	if len(folder.Knowledge) > 0 {
		documents := make([]acceleration.KnowledgeDocument, 0, len(folder.Knowledge))
		for _, document := range folder.Knowledge {
			documents = append(documents, acceleration.KnowledgeDocument{Source: document.Source, Text: document.Text})
		}
		body.Knowledge = &documents
	}
	if len(folder.KnowledgeURLs) > 0 {
		pages := make([]acceleration.KnowledgeUrlDeclaration, 0, len(folder.KnowledgeURLs))
		for _, page := range folder.KnowledgeURLs {
			declared := acceleration.KnowledgeUrlDeclaration{Url: page.URL}
			setString(&declared.Title, page.Title)
			setString(&declared.Description, page.Description)
			if page.RefreshHours > 0 {
				hours := int64(page.RefreshHours)
				declared.RefreshHours = &hours
			}
			pages = append(pages, declared)
		}
		body.KnowledgeUrls = &pages
	}
	if folder.Simulations != nil {
		declared := make([]acceleration.SimulationDeclaration, 0, len(folder.Simulations))
		for _, simulation := range folder.Simulations {
			declared = append(declared, simulationDeclarationOf(simulation))
		}
		body.Simulations = &declared
	}
	declareSettings(&body, folder.Settings)
	bindings, missing, err := connectorBindings(ctx, client, folder.Settings)
	if err != nil {
		return nil, err
	}
	body.Connectors = bindings
	for _, connection := range missing {
		a.logger.Warn("agent.yaml connector left out: the app has no such connection",
			"agent", a.options.Name, "connector", connection.name, "connection", connection.token)
	}
	if len(missing) > 0 {
		// Sent so the router stores them and the dashboard can say what the agent is missing.
		// An entry over the router's caps is left out of the list (the warning above still
		// names it), and so is one past the router's 64, so a sync never fails over the list.
		leftOut := make([]acceleration.LeftOutConnector, 0, len(missing))
		for _, connection := range missing {
			if len(leftOut) == maxLeftOutEntries {
				break
			}
			if utf8.RuneCountInString(connection.name) > maxLeftOutName ||
				utf8.RuneCountInString(connection.connectorID) > maxLeftOutConnectorID ||
				utf8.RuneCountInString(connection.token) > maxLeftOutConnection {
				continue
			}
			leftOut = append(leftOut, acceleration.LeftOutConnector{
				Name: connection.name, ConnectorId: connection.connectorID, Connection: connection.token,
			})
		}
		if len(leftOut) > 0 {
			body.LeftOutConnectors = &leftOut
		}
		// Another hash for a sync that left a connector out, sent and stamped: the router does
		// nothing for a hash it already stored, and the sync that finds the connection must not
		// be taken for this one. The marker keeps it from being the hash a sync that left the
		// same connectors out sent before the list was sent, which the router has stored
		// without the list.
		body.Hash = fingerprint(hash, "missing connections", "left_out_connectors "+fmt.Sprint(missing), nil, nil, nil, nil)
	}
	if harness != nil {
		body.Harness = harness
	}
	if subagent != nil {
		body.Subagent = subagent
	}
	if sandbox != nil {
		body.Sandbox = sandbox
	}
	if len(a.options.CostTracking) > 0 {
		tags := map[string]string{}
		if body.Tags != nil {
			tags = *body.Tags
		}
		for key, value := range a.options.CostTracking {
			tags[key] = value
		}
		body.Tags = &tags
	}

	synced, err := client.SyncAgentWithResponse(ctx, body)
	if err != nil {
		return nil, fmt.Errorf("agents: syncing %s: %w", a.options.Name, err)
	}
	result, err := answer(synced.JSON200, synced.HTTPResponse, synced.Body)
	if err != nil {
		return nil, err
	}
	for _, warning := range deref(result.Warnings) {
		a.logger.Warn("synced with a warning", "agent", a.options.Name, "warning", warning)
	}
	if err := WriteStamp(folder.Path, body.Hash); err != nil {
		return nil, err
	}
	return &result.Config, nil
}

// declareSettings carries what agent.yaml declared onto the sync request. Only what the file
// names is sent, so the router leaves whatever is already stored for the rest.
// pluginEntries are the plugins a declaration names as the router takes them: an id alone
// for one with nothing else said about it, or nothing for none.
func pluginEntries(named []PluginSettings) *[]acceleration.PluginEntry {
	if len(named) == 0 {
		return nil
	}
	entries := make([]acceleration.PluginEntry, 0, len(named))
	for _, plugin := range named {
		var entry acceleration.PluginEntry
		if plugin.User == nil && !plugin.Readonly && len(plugin.Scopes) == 0 && len(plugin.Toolsets) == 0 && len(plugin.Tools) == 0 {
			// A string always encodes.
			_ = entry.FromPluginEntry0(plugin.Name)
			entries = append(entries, entry)
			continue
		}
		declared := acceleration.PluginWithOptions{Name: plugin.Name}
		declared.User = plugin.User
		if plugin.Readonly {
			declared.Readonly = &plugin.Readonly
		}
		if len(plugin.Scopes) > 0 {
			declared.Scopes = &plugin.Scopes
		}
		if len(plugin.Toolsets) > 0 {
			declared.Toolsets = &plugin.Toolsets
		}
		if len(plugin.Tools) > 0 {
			declared.Tools = &plugin.Tools
		}
		// Neither does a struct of strings and a bool.
		_ = entry.FromPluginWithOptions(declared)
		entries = append(entries, entry)
	}
	return &entries
}

func declareSettings(body *acceleration.SyncAgentRequest, settings Settings) {
	if settings.Role != "" {
		role := acceleration.AgentRole(settings.Role)
		body.Role = &role
	}
	if settings.Mode != "" {
		mode := acceleration.AgentMode(settings.Mode)
		body.Mode = &mode
	}
	setString(&body.Stt, settings.STT)
	setString(&body.Tts, settings.TTS)
	body.Sts = settings.STS
	setString(&body.Voice, settings.Voice)
	setString(&body.Llm, settings.LLM)
	if settings.Harness != "" {
		harness := acceleration.Harness(settings.Harness)
		body.Harness = &harness
	}
	if settings.Dispatch != nil {
		body.Dispatch = &acceleration.AgentDispatch{}
		if settings.Dispatch.IncomingCall != "" {
			setting := acceleration.DispatchSetting(settings.Dispatch.IncomingCall)
			body.Dispatch.IncomingCall = &setting
		}
		if settings.Dispatch.Text != "" {
			setting := acceleration.DispatchSetting(settings.Dispatch.Text)
			body.Dispatch.Text = &setting
		}
	}
	setString(&body.Subagent, settings.Subagent)
	setString(&body.Search, settings.Search)
	if settings.Greeting != nil {
		body.Greeting = &acceleration.Greeting{Text: settings.Greeting.Text}
		if settings.Greeting.Mode != "" {
			mode := acceleration.GreetingMode(settings.Greeting.Mode)
			body.Greeting.Mode = &mode
		}
	}
	if settings.Sandbox != "" {
		sandbox := acceleration.Sandbox(settings.Sandbox)
		body.Sandbox = &sandbox
	}
	if options := settings.SandboxOptions; options != nil {
		timeout := options.timeout.Milliseconds()
		cpu, memory, disk := int64(options.CPU), int64(options.MemoryGB), int64(options.DiskGB)
		body.SandboxOptions = &acceleration.SandboxOptions{
			Image: &options.Image, Setup: &options.Setup, TimeoutMs: &timeout,
			Cpu: &cpu, MemoryGb: &memory, DiskGb: &disk, Network: &options.Network,
			AllowedDomains: &options.AllowedDomains, NetworkAllowList: &options.NetworkAllowList,
		}
	}
	body.Plugins = pluginEntries(settings.Plugins)
	if len(settings.PluginEvents) > 0 {
		events := make([]acceleration.PluginEvent, 0, len(settings.PluginEvents))
		for _, event := range settings.PluginEvents {
			declared := acceleration.PluginEvent{Plugin: event.Plugin, Event: event.Event}
			if len(event.Arguments) > 0 {
				arguments := event.Arguments
				declared.Arguments = &arguments
			}
			if event.Instructions != "" {
				declared.Instructions = &event.Instructions
			}
			events = append(events, declared)
		}
		body.PluginEvents = &events
	}
	if len(settings.MCPServers) > 0 {
		servers := make([]acceleration.McpServer, 0, len(settings.MCPServers))
		for _, server := range settings.MCPServers {
			declared := acceleration.McpServer{Name: server.Name, Url: server.URL}
			if len(server.Tools) > 0 {
				declared.Tools = &server.Tools
			}
			if len(server.Scopes) > 0 {
				declared.Scopes = &server.Scopes
			}
			if server.User {
				declared.User = &server.User
			}
			servers = append(servers, declared)
		}
		body.McpServers = &servers
	}
	if settings.Tools != nil {
		body.Tools = &acceleration.AgentTools{Progressive: settings.Tools.Progressive}
	}
	if settings.Channels != nil {
		declared := acceleration.AgentChannels{
			Whatsapp: channelLine(settings.Channels.WhatsApp),
			Sms:      channelLine(settings.Channels.SMS),
			Imessage: channelLine(settings.Channels.IMessage),
		}
		if settings.Channels.Identity != "" {
			identity := acceleration.ChannelIdentity(settings.Channels.Identity)
			declared.Identity = &identity
		}
		body.Channels = &declared
	}
	if settings.Scheduling != nil {
		declared := acceleration.AgentScheduling{
			Enabled:         settings.Scheduling.Enabled,
			DefaultTimezone: &settings.Scheduling.DefaultTimezone,
		}
		if settings.Scheduling.MaxTasksPerUser != nil {
			maximum := int64(*settings.Scheduling.MaxTasksPerUser)
			declared.MaxTasksPerUser = &maximum
		}
		if settings.Scheduling.DefaultTimezone == "" {
			declared.DefaultTimezone = nil
		}
		body.Scheduling = &declared
	}
	if len(settings.Keyterms) > 0 {
		body.Keyterms = &settings.Keyterms
	}
	if len(settings.Tags) > 0 {
		tags := maps.Clone(settings.Tags)
		body.Tags = &tags
	}
	if settings.Video != nil {
		body.Video = &acceleration.SessionVideo{MaxFrames: &settings.Video.MaxFrames}
		setString(&body.Video.Source, settings.Video.Source)
	}
}

// connectorBindings are the connectors agent.yaml names, in the shape the sync takes.
// Nil when the key was left out, so the stored bindings stay; empty when the file
// lists none, which removes them.
//
// An entry whose connection the app does not have is left out and returned, so the rest of
// the agent still syncs, unless the entry is required: a session needs that one.
func connectorBindings(ctx context.Context, client *acceleration.ClientWithResponses, settings Settings) (*[]acceleration.AgentConnectorBinding, []missingConnection, error) {
	if settings.Connectors == nil {
		return nil, nil, nil
	}
	bindings := make([]acceleration.AgentConnectorBinding, 0, len(settings.Connectors))
	var missing []missingConnection
	for _, entry := range settings.Connectors {
		binding, err := connectorBinding(ctx, client, entry)
		var absent missingConnection
		if errors.As(err, &absent) {
			if deref(entry.Required) {
				return nil, nil, fmt.Errorf("%w: add the connection, or drop required", err)
			}
			missing = append(missing, absent)
			continue
		}
		if err != nil {
			return nil, nil, err
		}
		bindings = append(bindings, binding)
	}
	return &bindings, missing, nil
}

// The longest alias, connector id and connection label the router takes in a
// left_out_connectors entry (LeftOutConnector in acceleration/internal/api/sync.go).
const (
	maxLeftOutName        = 63
	maxLeftOutConnectorID = 64
	maxLeftOutConnection  = 120
	maxLeftOutEntries     = 64
)

// missingConnection is a connection an agent.yaml entry names that the app does not have.
//
// The sync's hash prints a list of them, which formats each through Error, so a field that
// Error leaves out does not change the hash. The hash also carries a "left_out_connectors"
// marker, so it is not the one a sync sent before the list was sent.
type missingConnection struct {
	name, token string
	connectorID string
}

func (m missingConnection) Error() string {
	return fmt.Sprintf("agents: connector %q: no app connection with id or label %q", m.name, m.token)
}

func connectorBinding(ctx context.Context, client *acceleration.ClientWithResponses, entry ConnectorSettings) (acceleration.AgentConnectorBinding, error) {
	connectorID := entry.Connector
	if connectorID == "" {
		connectorID = entry.Name
	}
	patterns := []string{"*"}
	if entry.Tools != nil {
		patterns = append([]string(nil), (*entry.Tools)...)
	}
	binding := acceleration.AgentConnectorBinding{
		Name:         entry.Name,
		ConnectorId:  connectorID,
		Tools:        []acceleration.ConnectorToolGrant{},
		ToolPatterns: &patterns,
		Required:     entry.Required,
		Connection:   acceleration.AgentConnectorSelection{Type: acceleration.AgentConnectorSelectionTypeSession},
	}
	if entry.Connection == nil {
		return binding, nil
	}
	id, err := fixedConnection(ctx, client, entry.Name, connectorID, *entry.Connection)
	if err != nil {
		return acceleration.AgentConnectorBinding{}, err
	}
	binding.Connection = acceleration.AgentConnectorSelection{
		Type:         acceleration.AgentConnectorSelectionTypeFixed,
		ConnectionId: &id,
	}
	return binding, nil
}

// fixedConnection is the app connection entry names, by id or by the one label that
// matches. An id wins over a label. Every page is read first: a label on an early page
// must not hide the id on a later one, and a label is only unique once the list ends.
func fixedConnection(ctx context.Context, client *acceleration.ClientWithResponses, name, connectorID, token string) (string, error) {
	connections, err := appConnections(ctx, client, connectorID)
	if err != nil {
		return "", fmt.Errorf("agents: connector %q: %w", name, err)
	}
	for _, connection := range connections {
		if connection.Id != nil && *connection.Id == token {
			return *connection.Id, nil
		}
	}
	var labels []string
	for _, connection := range connections {
		if connection.Label != nil && *connection.Label == token && connection.Id != nil {
			labels = append(labels, *connection.Id)
		}
	}
	switch len(labels) {
	case 1:
		return labels[0], nil
	case 0:
		return "", missingConnection{name: name, token: token, connectorID: connectorID}
	default:
		return "", fmt.Errorf("agents: connector %q: %q matches %d app connections", name, token, len(labels))
	}
}

func appConnections(ctx context.Context, client *acceleration.ClientWithResponses, connectorID string) ([]acceleration.Connection, error) {
	// Named although omitted is app, so a router from before owner_type was optional answers too.
	app := acceleration.ConnectionOwnerTypeApp
	var cursor *string
	asked := map[string]bool{}
	var connections []acceleration.Connection
	for {
		key := ""
		if cursor != nil {
			key = *cursor
		}
		if asked[key] {
			break
		}
		asked[key] = true

		listed, err := client.ListConnectionsWithResponse(ctx, &acceleration.ListConnectionsParams{
			OwnerType:   &app,
			ConnectorId: &connectorID,
			Cursor:      cursor,
		})
		if err != nil {
			return nil, fmt.Errorf("listing connections: %w", err)
		}
		page, err := answer(listed.JSON200, listed.HTTPResponse, listed.Body)
		if err != nil {
			return nil, err
		}
		if page.Items != nil {
			connections = append(connections, (*page.Items)...)
		}
		if page.NextCursor == nil || *page.NextCursor == "" {
			break
		}
		cursor = page.NextCursor
	}
	return connections, nil
}

// storedConfig is the config stored under a name, or nil when there is none.
func storedConfig(ctx context.Context, client *acceleration.ClientWithResponses, name string) (*acceleration.AgentConfig, error) {
	listed, err := client.ListAgentConfigsWithResponse(ctx, &acceleration.ListAgentConfigsParams{Name: &name})
	if err != nil {
		return nil, fmt.Errorf("agents: listing configs: %w", err)
	}
	stored, err := answer(listed.JSON200, listed.HTTPResponse, listed.Body)
	if err != nil {
		return nil, err
	}
	for _, config := range *stored {
		if config.Name == name {
			return &config, nil
		}
	}
	return nil, nil
}

// syncedSkills are the skills the stored config should name. The harness is the whole of
// them: a directory's skills are folded into it when the agent is built.
func (a *Agent) syncedSkills() []Skill {
	if a.options.Harness != nil && len(a.options.Harness.Skills) > 0 {
		return a.options.Harness.Skills
	}
	if a.folder != nil {
		return a.folder.Skills
	}
	return nil
}

// DefineAgent stores an agent configuration, editing whichever is already under that name.
func DefineAgent(
	ctx context.Context,
	client *acceleration.ClientWithResponses,
	wanted acceleration.AgentConfigRequest,
) (*acceleration.AgentConfig, error) {
	// Narrowed to the name rather than read whole and filtered here: names are unique per
	// customer, so this asks the router the question instead of downloading every config an
	// app has to answer it locally.
	listed, err := client.ListAgentConfigsWithResponse(ctx,
		&acceleration.ListAgentConfigsParams{Name: &wanted.Name})
	if err != nil {
		return nil, fmt.Errorf("agents: listing configs: %w", err)
	}
	stored, err := answer(listed.JSON200, listed.HTTPResponse, listed.Body)
	if err != nil {
		return nil, err
	}

	for _, config := range *stored {
		if config.Name != wanted.Name {
			continue
		}
		updated, err := client.UpdateAgentConfigWithResponse(ctx, config.Id, wanted)
		if err != nil {
			return nil, fmt.Errorf("agents: updating config %s: %w", config.Id, err)
		}
		return answer(updated.JSON200, updated.HTTPResponse, updated.Body)
	}

	created, err := client.CreateAgentConfigWithResponse(ctx, wanted)
	if err != nil {
		return nil, fmt.Errorf("agents: creating config %s: %w", wanted.Name, err)
	}
	return answer(created.JSON201, created.HTTPResponse, created.Body)
}

// DefineSkills stores skills, editing whichever is already under each name.
func DefineSkills(ctx context.Context, client *acceleration.ClientWithResponses, skills []Skill) error {
	listed, err := client.ListSkillsWithResponse(ctx, nil)
	if err != nil {
		return fmt.Errorf("agents: listing skills: %w", err)
	}
	stored, err := answer(listed.JSON200, listed.HTTPResponse, listed.Body)
	if err != nil {
		return err
	}

	known := map[string]string{}
	for _, skill := range *stored {
		known[skill.Name] = skill.Id
	}

	for _, skill := range skills {
		body := skillRequestOf(skill)
		if id, ok := known[skill.Name]; ok {
			updated, err := client.UpdateSkillWithResponse(ctx, id, body)
			if err != nil {
				return fmt.Errorf("agents: updating skill %s: %w", skill.Name, err)
			}
			if _, err := answer(updated.JSON200, updated.HTTPResponse, updated.Body); err != nil {
				return err
			}
			continue
		}

		created, err := client.CreateSkillWithResponse(ctx, body)
		if err != nil {
			return fmt.Errorf("agents: creating skill %s: %w", skill.Name, err)
		}
		if _, err := answer(created.JSON201, created.HTTPResponse, created.Body); err != nil {
			return err
		}
	}
	return nil
}

func skillRequestOf(skill Skill) acceleration.SkillRequest {
	body := acceleration.SkillRequest{
		Name:         skill.Name,
		CaptureVideo: &skill.CaptureVideo,
		Description:  skill.Description,
		Instructions: skill.Instructions,
	}
	if skill.Deadline > 0 {
		milliseconds := skill.Deadline.Milliseconds()
		body.DeadlineMs = &milliseconds
	}
	if len(skill.Files) > 0 {
		files := skill.Files
		body.Files = &files
	}
	return body
}

func simulationDeclarationOf(simulation Simulation) acceleration.SimulationDeclaration {
	declared := acceleration.SimulationDeclaration{
		Name:      simulation.Name,
		Scenario:  simulation.Scenario,
		Assertion: simulation.Assertion,
	}
	if simulation.Mode != "" {
		mode := acceleration.SimulationDeclarationMode(simulation.Mode)
		declared.Mode = &mode
	}
	if simulation.Variations > 0 {
		variations := int64(simulation.Variations)
		declared.Variations = &variations
	}
	if simulation.MaxTurns > 0 {
		turns := int64(simulation.MaxTurns)
		declared.MaxTurns = &turns
	}
	setString(&declared.CallerTarget, simulation.CallerTarget)
	setString(&declared.JudgeTarget, simulation.JudgeTarget)
	setString(&declared.CallerStt, simulation.CallerSTT)
	setString(&declared.CallerTts, simulation.CallerTTS)
	setString(&declared.CallerVoice, simulation.CallerVoice)
	if len(simulation.Tags) > 0 {
		tags := maps.Clone(simulation.Tags)
		declared.Tags = &tags
	}
	return declared
}

// IngestKnowledge fills a knowledge base with documents an agent can look things up in.
//
// The documents are cut into passages by the backend, so a directory pushed from here and
// one read off the router's own disk are cut the same way and replace each other.
func IngestKnowledge(
	ctx context.Context,
	client *acceleration.ClientWithResponses,
	namespace string,
	documents []Document,
) error {
	if len(documents) == 0 {
		return nil
	}

	body := acceleration.IngestKnowledgeRequest{
		Namespace: namespace,
		Documents: make([]acceleration.KnowledgeDocument, 0, len(documents)),
	}
	for _, document := range documents {
		body.Documents = append(body.Documents, acceleration.KnowledgeDocument{
			Source: document.Source,
			Text:   document.Text,
		})
	}

	written, err := client.IngestKnowledgeWithResponse(ctx, body)
	if err != nil {
		return fmt.Errorf("agents: filling %s: %w", namespace, err)
	}
	_, err = answer(written.JSON200, written.HTTPResponse, written.Body)
	return err
}

// SubscribeKnowledgeURLs keeps a knowledge base filled from the pages a directory declares.
//
// A page is subscribed by url, so this is safe to call with the same declaration again: the
// backend re-reads a page it already has rather than storing a second copy of it.
func SubscribeKnowledgeURLs(
	ctx context.Context,
	client *acceleration.ClientWithResponses,
	namespace string,
	pages []KnowledgeURL,
) error {
	for _, page := range pages {
		body := acceleration.KnowledgeUrlRequest{Namespace: namespace, Url: page.URL}
		setString(&body.Title, page.Title)
		setString(&body.Description, page.Description)
		if page.RefreshHours > 0 {
			body.RefreshHours = &page.RefreshHours
		}

		added, err := client.AddKnowledgeUrlWithResponse(ctx, body)
		if err != nil {
			return fmt.Errorf("agents: reading %s: %w", page.URL, err)
		}
		if _, err := answer(added.JSON201, added.HTTPResponse, added.Body); err != nil {
			return err
		}
	}
	return nil
}

// answer returns what the router sent, raising what it said went wrong instead.
func answer[T any](ok *T, response *http.Response, body []byte) (*T, error) {
	if ok != nil {
		return ok, nil
	}
	return nil, stream.NewRouterError(response, body, "agents", "", "the router answered "+response.Status)
}

// deref is what a field holds, or its zero value when it holds nothing.
func deref[T any](field *T) T {
	var zero T
	if field == nil {
		return zero
	}
	return *field
}

func setString(field **string, value string) {
	if value != "" {
		*field = &value
	}
}

// channelLine is one declared channel for the wire, or nothing when it names no number.
func channelLine(line *ChannelSettings) *acceleration.ChannelLineRequest {
	if line == nil || line.Number == "" {
		return nil
	}
	return &acceleration.ChannelLineRequest{Number: line.Number}
}
