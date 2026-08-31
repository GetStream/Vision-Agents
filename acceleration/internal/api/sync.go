package api

import (
	"context"
	"errors"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// SyncAgent stores an agent directory's instructions, skills and knowledge.
//
// The hash is a fingerprint of that directory. A second call with the same hash does
// nothing, so a process that syncs on startup is cheap when nothing has changed.
func (s *Server) SyncAgent(ctx context.Context, request SyncAgentRequestObject) (SyncAgentResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return SyncAgent401JSONResponse{missingCustomer()}, nil
	}
	if request.Body == nil {
		return SyncAgent400JSONResponse{badRequest("a request body is required")}, nil
	}

	body := *request.Body
	name := strings.TrimSpace(body.Name)
	hash := strings.TrimSpace(body.Hash)
	if name == "" {
		return SyncAgent400JSONResponse{badRequest("an agent config needs a name")}, nil
	}
	if hash == "" {
		return SyncAgent400JSONResponse{badRequest("a hash is required, so a second sync can do nothing")}, nil
	}

	if s.store == nil {
		return SyncAgent400JSONResponse{badRequest(noConfigs)}, nil
	}

	existing, found, err := s.store.AgentConfigByName(ctx, customerID, name)
	if err != nil {
		return nil, err
	}
	if found && existing.SyncHash == hash {
		return SyncAgent200JSONResponse{Unchanged: true, Config: agentConfigOf(existing)}, nil
	}

	skills := skillsOf(body.Skills)
	if len(skills) > 0 {
		if err := s.upsertSkills(ctx, customerID, skills); err != nil {
			return SyncAgent400JSONResponse{badRequest(err.Error())}, nil
		}
	}

	documents := documentsOf(body.Knowledge)
	namespace := ""
	if len(documents) > 0 {
		if s.knowledge == nil {
			return SyncAgent400JSONResponse{badRequest(noKnowledge)}, nil
		}
		namespace = name
		if _, _, err := s.fillKnowledge(ctx, namespace, documents, nil); err != nil {
			return SyncAgent400JSONResponse{badRequest(err.Error())}, nil
		}
	}

	named := make([]string, 0, len(skills))
	for _, skill := range skills {
		named = append(named, skill.Name)
	}

	config := existing
	if !found {
		config = store.AgentConfig{CustomerID: customerID, Name: name}
	}
	config.Instructions = value(body.Instructions)
	config.Skills = named
	config.KnowledgeNamespace = namespace
	config.SyncHash = hash

	if found {
		if err := s.store.UpdateAgentConfig(ctx, &config); err != nil {
			return SyncAgent400JSONResponse{badRequest(err.Error())}, nil
		}
	} else {
		if err := s.store.CreateAgentConfig(ctx, &config); err != nil {
			return SyncAgent400JSONResponse{badRequest(err.Error())}, nil
		}
	}
	return SyncAgent200JSONResponse{Unchanged: false, Config: agentConfigOf(config)}, nil
}

func skillsOf(list *[]SkillRequest) []SkillRequest {
	if list == nil {
		return nil
	}
	return *list
}

func documentsOf(list *[]KnowledgeDocument) []KnowledgeDocument {
	if list == nil {
		return nil
	}
	return *list
}

func (s *Server) upsertSkills(ctx context.Context, customerID string, skills []SkillRequest) error {
	names := make([]string, 0, len(skills))
	for _, skill := range skills {
		if message, ok := skillComplaint(skill); !ok {
			return errors.New(message)
		}
		names = append(names, strings.TrimSpace(skill.Name))
	}

	stored, err := s.store.SkillsNamed(ctx, customerID, names)
	if err != nil {
		return err
	}
	known := map[string]store.Skill{}
	for _, skill := range stored {
		known[skill.Name] = skill
	}

	for _, skill := range skills {
		row := storedSkill(skill, customerID)
		if existing, ok := known[row.Name]; ok {
			row.ID = existing.ID
			row.CreatedAt = existing.CreatedAt
			if err := s.store.UpdateSkill(ctx, &row); err != nil {
				return err
			}
			continue
		}
		if err := s.store.CreateSkill(ctx, &row); err != nil {
			return err
		}
	}
	return nil
}
