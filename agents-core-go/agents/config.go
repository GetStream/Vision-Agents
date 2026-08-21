package agents

import (
	"context"
	"fmt"

	"github.com/GetStream/Vision-Agents/agents-core-go/acceleration"
)

// Sync stores the agent's configuration in the backend, along with its skills and whatever
// its knowledge directory holds.
//
// A config is what a session can be created from by name, so the things worth deciding once
// are decided once. Both the config and its skills are found by name first, so calling this
// twice edits what is stored rather than storing another copy of it.
func (a *Agent) Sync(ctx context.Context) (*acceleration.AgentConfig, error) {
	client, err := a.Client()
	if err != nil {
		return nil, err
	}

	skills := a.syncedSkills()
	if len(skills) > 0 {
		if err := DefineSkills(ctx, client, skills); err != nil {
			return nil, err
		}
	}

	namespace := ""
	if a.folder != nil && len(a.folder.Knowledge) > 0 {
		namespace = a.folder.KnowledgeNamespace()
		if err := IngestKnowledge(ctx, client, namespace, a.folder.Knowledge); err != nil {
			return nil, err
		}
	}

	wanted := acceleration.AgentConfigRequest{Name: a.options.Name}
	setString(&wanted.Instructions, a.options.Instructions)
	setString(&wanted.KnowledgeNamespace, namespace)
	setString(&wanted.Subagent, a.options.Harness.Subagent())
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
	listed, err := client.ListAgentConfigsWithResponse(ctx)
	if err != nil {
		return nil, fmt.Errorf("agents: listing configs: %w", err)
	}
	stored, err := answer(listed.JSON200, listed.JSON400, listed.JSON401, nil, listed.Status())
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
		return answer(updated.JSON200, updated.JSON400, updated.JSON401, updated.JSON404, updated.Status())
	}

	created, err := client.CreateAgentConfigWithResponse(ctx, wanted)
	if err != nil {
		return nil, fmt.Errorf("agents: creating config %s: %w", wanted.Name, err)
	}
	return answer(created.JSON201, created.JSON400, created.JSON401, nil, created.Status())
}

// DefineSkills stores skills, editing whichever is already under each name.
func DefineSkills(ctx context.Context, client *acceleration.ClientWithResponses, skills []Skill) error {
	listed, err := client.ListSkillsWithResponse(ctx)
	if err != nil {
		return fmt.Errorf("agents: listing skills: %w", err)
	}
	stored, err := answer(listed.JSON200, listed.JSON400, listed.JSON401, nil, listed.Status())
	if err != nil {
		return err
	}

	known := map[string]string{}
	for _, skill := range *stored {
		known[skill.Name] = skill.Id
	}

	for _, skill := range skills {
		body := acceleration.SkillRequest{
			Name:         skill.Name,
			Description:  skill.Description,
			Instructions: skill.Instructions,
		}
		if skill.Deadline > 0 {
			milliseconds := skill.Deadline.Milliseconds()
			body.DeadlineMs = &milliseconds
		}

		if id, ok := known[skill.Name]; ok {
			updated, err := client.UpdateSkillWithResponse(ctx, id, body)
			if err != nil {
				return fmt.Errorf("agents: updating skill %s: %w", skill.Name, err)
			}
			if _, err := answer(updated.JSON200, updated.JSON400, updated.JSON401, updated.JSON404, updated.Status()); err != nil {
				return err
			}
			continue
		}

		created, err := client.CreateSkillWithResponse(ctx, body)
		if err != nil {
			return fmt.Errorf("agents: creating skill %s: %w", skill.Name, err)
		}
		if _, err := answer(created.JSON201, created.JSON400, created.JSON401, nil, created.Status()); err != nil {
			return err
		}
	}
	return nil
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
	_, err = answer(written.JSON200, written.JSON400, written.JSON401, nil, written.Status())
	return err
}

// answer returns what the router sent, raising what it said went wrong instead.
func answer[T any](ok *T, bad, unauthorized, missing *acceleration.Error, status string) (*T, error) {
	if ok != nil {
		return ok, nil
	}
	for _, failure := range []*acceleration.Error{bad, unauthorized, missing} {
		if failure != nil {
			return nil, fmt.Errorf("agents: %s", failure.Error)
		}
	}
	return nil, fmt.Errorf("agents: the router answered %s", status)
}

func setString(field **string, value string) {
	if value != "" {
		*field = &value
	}
}
