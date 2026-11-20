package agents

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// Sync stores the agent's configuration in the backend, along with its skills and whatever
// its knowledge directory holds.
//
// A config is what a session can be created from by name, so the things worth deciding once
// are decided once. Both the config and its skills are found by name first, so calling this
// twice edits what is stored rather than storing another copy of it. When the stored config
// has a sandbox its skills are written before the config, so a config the router refuses
// leaves those skill edits applied; a corrected retry converges.
//
// An agent read from a directory is synced as its files, which the router reads, and
// .agent_sync records the fingerprint the router took of them. A directory unchanged since
// then is only read back, not sent again.
func (a *Agent) Sync(ctx context.Context) (*acceleration.AgentConfig, error) {
	client, err := a.Client()
	if err != nil {
		return nil, err
	}
	if a.folder != nil {
		return a.syncFolder(ctx, client)
	}

	var skills []Skill
	if a.options.Harness != nil {
		skills = a.options.Harness.Skills
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

	stored, err := storedConfig(ctx, client, a.options.Name)
	if err != nil {
		return nil, err
	}
	// A config with a sandbox takes any skill, and an edit that drops the sandbox is refused
	// while a skill it names still has scripts. So the skills of a config with a sandbox are
	// written first, and a skill can lose its scripts in the sync that drops the sandbox.
	skillsFirst := stored != nil && deref(stored.Sandbox) != ""
	if skillsFirst && len(skills) > 0 {
		if err := defineSkills(ctx, client, stored.Id, skills); err != nil {
			return nil, err
		}
	}
	config, err := putConfig(ctx, client, stored, wanted)
	if err != nil {
		return nil, err
	}
	// A skill belongs to a config, so it is stored once the config has an id to name. Without
	// a sandbox stored, the config goes first too, so a skill with scripts finds the sandbox
	// that this sync adds.
	if !skillsFirst && len(skills) > 0 {
		if err := defineSkills(ctx, client, config.Id, skills); err != nil {
			return nil, err
		}
	}
	return config, nil
}

// syncFolder stores the agent's directory in one request: its files as they are, which the
// router reads, and what the code set beside them, which wins over them.
func (a *Agent) syncFolder(ctx context.Context, client *acceleration.ClientWithResponses) (*acceleration.AgentConfig, error) {
	folder := a.folder
	files, err := folder.files()
	if err != nil {
		return nil, err
	}
	body := acceleration.SyncAgentFilesRequest{Name: &a.options.Name, Files: files}
	if a.options.Instructions != folder.Instructions {
		body.Instructions = &a.options.Instructions
	}
	if a.options.Guardrail != folder.Guardrail {
		body.Guardrail = &a.options.Guardrail
	}
	body.Harness, body.Subagent, body.Sandbox = a.options.Harness.stored()
	if len(a.options.CostTracking) > 0 {
		body.Tags = &a.options.CostTracking
	}
	if a.skillsInCode {
		requests := make([]acceleration.SkillRequest, 0, len(a.options.Harness.Skills))
		for _, skill := range a.options.Harness.Skills {
			requests = append(requests, skillRequestOf(skill))
		}
		body.Skills = &requests
	}

	// Nothing is sent when the directory and what the code set are what was last sent, and the
	// config still is what the router then stored: one somebody has edited or synced since, or
	// one on another router, is synced again, and so is one that left a connector out, so the
	// sync that finds its connection is made.
	hash, err := sentHash(body)
	if err != nil {
		return nil, fmt.Errorf("agents: syncing %s: %w", a.options.Name, err)
	}
	if stamp := readStamp(folder.Path); stamp.Hash == hash {
		stored, err := storedConfig(ctx, client, a.options.Name)
		if err != nil {
			return nil, err
		}
		if stored != nil && deref(stored.SyncHash) == stamp.SyncHash && len(deref(stored.LeftOutConnectors)) == 0 {
			return stored, nil
		}
	}

	synced, err := client.SyncAgentFilesWithResponse(ctx, body)
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
	if err := writeStamp(folder.Path, syncStamp{Hash: hash, SyncHash: deref(result.Config.SyncHash)}); err != nil {
		return nil, err
	}
	return &result.Config, nil
}

// sentHash is the fingerprint .agent_sync keeps of a sync's request: the SHA-256, in hex, of
// the request as JSON with every object's keys in order and nothing between its tokens. It
// is the one the JavaScript SDK takes (sentHash in sdks/js/src/agent.ts), so a directory
// either of them synced is not sent again by the other.
func sentHash(body acceleration.SyncAgentFilesRequest) (string, error) {
	encoded, err := json.Marshal(body)
	if err != nil {
		return "", err
	}
	var ordered any
	if err := json.Unmarshal(encoded, &ordered); err != nil {
		return "", err
	}
	var canonical bytes.Buffer
	encoder := json.NewEncoder(&canonical)
	// JavaScript writes <, > and & as they are.
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(ordered); err != nil {
		return "", err
	}
	hashed := sha256.Sum256(bytes.TrimSuffix(canonical.Bytes(), []byte("\n")))
	return hex.EncodeToString(hashed[:]), nil
}

// storedConfig is the config stored under a name, or nil when there is none.
func storedConfig(ctx context.Context, client *acceleration.ClientWithResponses, name string) (*acceleration.AgentConfig, error) {
	// Narrowed to the name rather than read whole and filtered here: names are unique per
	// customer, so this asks the router the question instead of downloading every config an
	// app has to answer it locally.
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

// DefineAgent stores an agent configuration on the router backend points at, editing
// whichever is already under that name.
func DefineAgent(
	ctx context.Context,
	backend stream.Backend,
	wanted acceleration.AgentConfigRequest,
) (*acceleration.AgentConfig, error) {
	client, err := backend.Client()
	if err != nil {
		return nil, err
	}
	return defineAgent(ctx, client, wanted)
}

func defineAgent(
	ctx context.Context,
	client *acceleration.ClientWithResponses,
	wanted acceleration.AgentConfigRequest,
) (*acceleration.AgentConfig, error) {
	stored, err := storedConfig(ctx, client, wanted.Name)
	if err != nil {
		return nil, err
	}
	return putConfig(ctx, client, stored, wanted)
}

// putConfig edits stored to be wanted, or creates wanted when nothing is stored.
func putConfig(
	ctx context.Context,
	client *acceleration.ClientWithResponses,
	stored *acceleration.AgentConfig,
	wanted acceleration.AgentConfigRequest,
) (*acceleration.AgentConfig, error) {
	if stored != nil {
		updated, err := client.UpdateAgentConfigWithResponse(ctx, stored.Id, wanted)
		if err != nil {
			return nil, fmt.Errorf("agents: updating config %s: %w", stored.Id, err)
		}
		return answer(updated.JSON200, updated.HTTPResponse, updated.Body)
	}

	created, err := client.CreateAgentConfigWithResponse(ctx, wanted)
	if err != nil {
		return nil, fmt.Errorf("agents: creating config %s: %w", wanted.Name, err)
	}
	return answer(created.JSON201, created.HTTPResponse, created.Body)
}

// DefineSkills stores skills on the router backend points at, as the skills of the agent
// config with this id, editing whichever of its skills is already under each name.
func DefineSkills(ctx context.Context, backend stream.Backend, configID string, skills []Skill) error {
	client, err := backend.Client()
	if err != nil {
		return err
	}
	return defineSkills(ctx, client, configID, skills)
}

func defineSkills(ctx context.Context, client *acceleration.ClientWithResponses, configID string, skills []Skill) error {
	// Names are unique among one config's skills, so only this config's are looked at.
	listed, err := client.ListSkillsWithResponse(ctx, &acceleration.ListSkillsParams{ConfigId: &configID})
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
		body.ConfigId = configID
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

// IngestKnowledge fills a knowledge base with documents an agent can look things up in.
//
// The documents are cut into passages by the backend, so a directory pushed from here and
// one read off the router's own disk are cut the same way and replace each other.
func IngestKnowledge(
	ctx context.Context,
	backend stream.Backend,
	namespace string,
	documents []Document,
) error {
	if len(documents) == 0 {
		return nil
	}
	client, err := backend.Client()
	if err != nil {
		return err
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
	backend stream.Backend,
	namespace string,
	pages []KnowledgeURL,
) error {
	client, err := backend.Client()
	if err != nil {
		return err
	}
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
