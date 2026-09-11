package api

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// SyncAgent stores an agent directory: its instructions, skills and knowledge, and the
// settings its declaration decided.
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
	if message, ok := syncComplaint(body); !ok {
		return SyncAgent400JSONResponse{badRequest(message)}, nil
	}

	existing, found, err := s.store.AgentConfigByName(ctx, customerID, name)
	if err != nil {
		return nil, err
	}
	if found && existing.SyncHash == hash {
		return SyncAgent200JSONResponse{Unchanged: true, Config: agentConfigOf(existing)}, nil
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

	skills := skillsOf(body.Skills)
	named := make([]string, 0, len(skills))
	for _, skill := range skills {
		named = append(named, strings.TrimSpace(skill.Name))
	}

	config := existing
	if !found {
		config = store.AgentConfig{CustomerID: customerID, Name: name}
	}
	config.Instructions = value(body.Instructions)
	config.Skills = named
	config.KnowledgeNamespace = namespace
	config.SyncHash = hash
	applySettings(&config, body)

	if found {
		if err := s.store.UpdateAgentConfig(ctx, &config); err != nil {
			return SyncAgent400JSONResponse{badRequest(err.Error())}, nil
		}
	} else {
		if err := s.store.CreateAgentConfig(ctx, &config); err != nil {
			return SyncAgent400JSONResponse{badRequest(err.Error())}, nil
		}
	}

	// The skills belong to the config, so they are written after it: a new agent has no
	// id to hang them off until it has been stored.
	if len(skills) > 0 {
		if err := s.upsertSkills(ctx, customerID, config.ID, skills); err != nil {
			return SyncAgent400JSONResponse{badRequest(err.Error())}, nil
		}
	}
	return SyncAgent200JSONResponse{Unchanged: false, Config: agentConfigOf(config)}, nil
}

// syncComplaint reports what is wrong with the settings a directory declared, if
// anything. It is the same reading configComplaint does, since a directory decides the
// same things a config written by hand does.
func syncComplaint(body SyncAgentRequest) (string, bool) {
	if err := workerConflict(body.Subagent, body.Subagents); err != nil {
		return err.Error(), false
	}
	if _, ok := modeOf(body.Mode); !ok {
		return fmt.Sprintf("an agent is either %s or %s", store.AgentModeVoice, store.AgentModeText), false
	}
	if len(keytermsOf(body.Keyterms)) > stt.MaxKeyterms {
		return fmt.Sprintf("a config may name at most %d keyterms", stt.MaxKeyterms), false
	}
	if _, ok := sandboxOf(body.Sandbox); !ok {
		return fmt.Sprintf("there is no sandbox provider called %q", *body.Sandbox), false
	}
	return "", true
}

// applySettings writes onto a config what the directory's declaration decided. Only what
// was sent is applied: a directory that says nothing about a model leaves the one already
// stored, so a target chosen in the dashboard survives a sync.
func applySettings(config *store.AgentConfig, body SyncAgentRequest) {
	if body.Subagents != nil {
		if config.Subagents == nil {
			config.Subagents = map[string]string{}
		}
		for name, target := range *body.Subagents {
			config.Subagents[name] = target
		}
	}
	if body.Video != nil {
		config.VideoSource = override(config.VideoSource, body.Video.Source)
		config.VideoMaxFrames = override(config.VideoMaxFrames, body.Video.MaxFrames)
	}
	if body.Mode != nil && *body.Mode != "" {
		mode, _ := modeOf(body.Mode)
		config.Mode = mode
	}
	if body.Stt != nil {
		config.STT = *body.Stt
	}
	if body.Tts != nil {
		config.TTS = *body.Tts
	}
	if body.Voice != nil {
		config.Voice = *body.Voice
	}
	if body.Llm != nil {
		config.LLM = *body.Llm
	}
	if body.Subagent != nil {
		config.Subagent = *body.Subagent
		if body.Subagents == nil {
			delete(config.Subagents, "default")
		} else if _, exists := (*body.Subagents)["default"]; !exists {
			delete(config.Subagents, "default")
		}
	}
	if body.Search != nil {
		config.Search = *body.Search
	}
	if body.Greeting != nil {
		config.Greeting = *body.Greeting
	}
	if body.Plugins != nil {
		config.Plugins = *body.Plugins
	}
	if body.Keyterms != nil {
		config.Keyterms = keytermsOf(body.Keyterms)
	}
	if body.SandboxProfile != nil {
		config.SandboxProfile = *body.SandboxProfile
	}
	if body.Sandbox != nil {
		box, _ := sandboxOf(body.Sandbox)
		config.Sandbox = box
	}
	if body.Tags != nil {
		config.Tags = *body.Tags
	}
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

func (s *Server) upsertSkills(ctx context.Context, customerID, configID string, skills []SkillRequest) error {
	names := make([]string, 0, len(skills))
	for _, skill := range skills {
		// The directory names the skill and the config owns it, so the caller does not
		// repeat the config id per skill and it is filled in here.
		skill.ConfigId = configID
		if message, ok := skillComplaint(skill); !ok {
			return errors.New(message)
		}
		names = append(names, strings.TrimSpace(skill.Name))
	}

	stored, err := s.store.SkillsNamed(ctx, customerID, configID, names)
	if err != nil {
		return err
	}
	known := map[string]store.Skill{}
	for _, skill := range stored {
		known[skill.Name] = skill
	}

	for _, skill := range skills {
		skill.ConfigId = configID
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
