package api

import (
	"context"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// noConfigs is what the config and skill paths say on a deployment without a database.
// They are stored rather than computed, so there is nothing to serve without one.
const noConfigs = "agent configs are not available: no database configured"

// ListAgentConfigs returns the calling customer's configs, newest first.
func (s *Server) ListAgentConfigs(ctx context.Context, _ ListAgentConfigsRequestObject) (ListAgentConfigsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListAgentConfigs401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return ListAgentConfigs400JSONResponse{badRequest(noConfigs)}, nil
	}

	stored, err := s.store.CustomerAgentConfigs(ctx, customerID)
	if err != nil {
		return nil, err
	}

	listed := make([]AgentConfig, 0, len(stored))
	for _, config := range stored {
		listed = append(listed, agentConfigOf(config))
	}
	return ListAgentConfigs200JSONResponse(listed), nil
}

// CreateAgentConfig stores a configuration sessions can be created from.
func (s *Server) CreateAgentConfig(ctx context.Context, request CreateAgentConfigRequestObject) (CreateAgentConfigResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return CreateAgentConfig401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return CreateAgentConfig400JSONResponse{badRequest(noConfigs)}, nil
	}
	if request.Body == nil {
		return CreateAgentConfig400JSONResponse{badRequest("a request body is required")}, nil
	}
	if strings.TrimSpace(request.Body.Name) == "" {
		return CreateAgentConfig400JSONResponse{badRequest("an agent config needs a name")}, nil
	}

	config := storedConfig(*request.Body, customerID)
	if err := s.store.CreateAgentConfig(ctx, &config); err != nil {
		return CreateAgentConfig400JSONResponse{badRequest(err.Error())}, nil
	}
	return CreateAgentConfig201JSONResponse(agentConfigOf(config)), nil
}

// GetAgentConfig returns one config.
func (s *Server) GetAgentConfig(ctx context.Context, request GetAgentConfigRequestObject) (GetAgentConfigResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetAgentConfig401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return GetAgentConfig400JSONResponse{badRequest(noConfigs)}, nil
	}

	config, err := s.store.AgentConfig(ctx, customerID, request.Id)
	if err != nil {
		return GetAgentConfig404JSONResponse{NotFoundJSONResponse{Error: unknownConfig}}, nil
	}
	return GetAgentConfig200JSONResponse(agentConfigOf(config)), nil
}

// UpdateAgentConfig replaces a config with what it now is.
func (s *Server) UpdateAgentConfig(ctx context.Context, request UpdateAgentConfigRequestObject) (UpdateAgentConfigResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return UpdateAgentConfig401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return UpdateAgentConfig400JSONResponse{badRequest(noConfigs)}, nil
	}
	if request.Body == nil {
		return UpdateAgentConfig400JSONResponse{badRequest("a request body is required")}, nil
	}
	if strings.TrimSpace(request.Body.Name) == "" {
		return UpdateAgentConfig400JSONResponse{badRequest("an agent config needs a name")}, nil
	}

	existing, err := s.store.AgentConfig(ctx, customerID, request.Id)
	if err != nil {
		return UpdateAgentConfig404JSONResponse{NotFoundJSONResponse{Error: unknownConfig}}, nil
	}

	config := storedConfig(*request.Body, customerID)
	config.ID = existing.ID
	config.CreatedAt = existing.CreatedAt
	if err := s.store.UpdateAgentConfig(ctx, &config); err != nil {
		return UpdateAgentConfig400JSONResponse{badRequest(err.Error())}, nil
	}
	return UpdateAgentConfig200JSONResponse(agentConfigOf(config)), nil
}

// DeleteAgentConfig stops a config being usable.
func (s *Server) DeleteAgentConfig(ctx context.Context, request DeleteAgentConfigRequestObject) (DeleteAgentConfigResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return DeleteAgentConfig401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return DeleteAgentConfig400JSONResponse{badRequest(noConfigs)}, nil
	}

	if err := s.store.DeleteAgentConfig(ctx, customerID, request.Id); err != nil {
		return DeleteAgentConfig404JSONResponse{NotFoundJSONResponse{Error: unknownConfig}}, nil
	}
	return DeleteAgentConfig204Response{}, nil
}

// ListSkills returns the calling customer's skills, newest first.
func (s *Server) ListSkills(ctx context.Context, _ ListSkillsRequestObject) (ListSkillsResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return ListSkills401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return ListSkills400JSONResponse{badRequest(noConfigs)}, nil
	}

	stored, err := s.store.CustomerSkills(ctx, customerID)
	if err != nil {
		return nil, err
	}

	listed := make([]Skill, 0, len(stored))
	for _, skill := range stored {
		listed = append(listed, skillOf(skill))
	}
	return ListSkills200JSONResponse(listed), nil
}

// CreateSkill defines a kind of work worth handing to the slower model.
func (s *Server) CreateSkill(ctx context.Context, request CreateSkillRequestObject) (CreateSkillResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return CreateSkill401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return CreateSkill400JSONResponse{badRequest(noConfigs)}, nil
	}
	if request.Body == nil {
		return CreateSkill400JSONResponse{badRequest("a request body is required")}, nil
	}
	if message, ok := skillComplaint(*request.Body); !ok {
		return CreateSkill400JSONResponse{badRequest(message)}, nil
	}

	skill := storedSkill(*request.Body, customerID)
	if err := s.store.CreateSkill(ctx, &skill); err != nil {
		return CreateSkill400JSONResponse{badRequest(err.Error())}, nil
	}
	return CreateSkill201JSONResponse(skillOf(skill)), nil
}

// GetSkill returns one skill.
func (s *Server) GetSkill(ctx context.Context, request GetSkillRequestObject) (GetSkillResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return GetSkill401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return GetSkill400JSONResponse{badRequest(noConfigs)}, nil
	}

	skill, err := s.store.Skill(ctx, customerID, request.Id)
	if err != nil {
		return GetSkill404JSONResponse{NotFoundJSONResponse{Error: unknownSkill}}, nil
	}
	return GetSkill200JSONResponse(skillOf(skill)), nil
}

// UpdateSkill replaces a skill with what it now is.
func (s *Server) UpdateSkill(ctx context.Context, request UpdateSkillRequestObject) (UpdateSkillResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return UpdateSkill401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return UpdateSkill400JSONResponse{badRequest(noConfigs)}, nil
	}
	if request.Body == nil {
		return UpdateSkill400JSONResponse{badRequest("a request body is required")}, nil
	}
	if message, ok := skillComplaint(*request.Body); !ok {
		return UpdateSkill400JSONResponse{badRequest(message)}, nil
	}

	existing, err := s.store.Skill(ctx, customerID, request.Id)
	if err != nil {
		return UpdateSkill404JSONResponse{NotFoundJSONResponse{Error: unknownSkill}}, nil
	}

	skill := storedSkill(*request.Body, customerID)
	skill.ID = existing.ID
	skill.CreatedAt = existing.CreatedAt
	if err := s.store.UpdateSkill(ctx, &skill); err != nil {
		return UpdateSkill400JSONResponse{badRequest(err.Error())}, nil
	}
	return UpdateSkill200JSONResponse(skillOf(skill)), nil
}

// DeleteSkill stops a skill being usable.
func (s *Server) DeleteSkill(ctx context.Context, request DeleteSkillRequestObject) (DeleteSkillResponseObject, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return DeleteSkill401JSONResponse{missingCustomer()}, nil
	}
	if s.store == nil {
		return DeleteSkill400JSONResponse{badRequest(noConfigs)}, nil
	}

	if err := s.store.DeleteSkill(ctx, customerID, request.Id); err != nil {
		return DeleteSkill404JSONResponse{NotFoundJSONResponse{Error: unknownSkill}}, nil
	}
	return DeleteSkill204Response{}, nil
}

// unknownConfig and unknownSkill are what a caller is told about a resource that is not
// theirs, which is the same thing they are told about one that never existed.
const (
	unknownConfig = "no such agent config"
	unknownSkill  = "no such skill"
)

// skillComplaint reports what is wrong with a skill, if anything. A skill without a
// description is one the fast model would never know when to reach for.
func skillComplaint(request SkillRequest) (string, bool) {
	if strings.TrimSpace(request.Name) == "" {
		return "a skill needs a name", false
	}
	if strings.TrimSpace(request.Description) == "" {
		return "a skill needs a description, which is how the model decides when to use it", false
	}
	if strings.TrimSpace(request.Instructions) == "" {
		return "a skill needs instructions, which is what the subagent answers under", false
	}
	return "", true
}

// storedConfig turns a request into a row. The customer comes from the trusted header
// rather than the body, the same way a session's does.
func storedConfig(request AgentConfigRequest, customerID string) store.AgentConfig {
	config := store.AgentConfig{
		CustomerID:         customerID,
		Name:               strings.TrimSpace(request.Name),
		STT:                value(request.Stt),
		TTS:                value(request.Tts),
		Voice:              value(request.Voice),
		LLM:                value(request.Llm),
		Subagent:           value(request.Subagent),
		Instructions:       value(request.Instructions),
		Greeting:           value(request.Greeting),
		KnowledgeNamespace: value(request.KnowledgeNamespace),
	}
	if request.Skills != nil {
		config.Skills = *request.Skills
	}
	if request.Tags != nil {
		config.Tags = *request.Tags
	}
	return config
}

// storedSkill turns a request into a row.
func storedSkill(request SkillRequest, customerID string) store.Skill {
	return store.Skill{
		CustomerID:   customerID,
		Name:         strings.TrimSpace(request.Name),
		Description:  request.Description,
		Instructions: request.Instructions,
		DeadlineMs:   value(request.DeadlineMs),
	}
}

// agentConfigOf renders a config for the wire. Empty strings are left out rather than
// sent as blanks, so a config that says nothing about a model reads as saying nothing.
func agentConfigOf(config store.AgentConfig) AgentConfig {
	rendered := AgentConfig{
		Id:        config.ID,
		Name:      config.Name,
		CreatedAt: config.CreatedAt,
		UpdatedAt: config.UpdatedAt,
	}
	rendered.Stt = optional(config.STT)
	rendered.Tts = optional(config.TTS)
	rendered.Voice = optional(config.Voice)
	rendered.Llm = optional(config.LLM)
	rendered.Subagent = optional(config.Subagent)
	rendered.Instructions = optional(config.Instructions)
	rendered.Greeting = optional(config.Greeting)
	rendered.KnowledgeNamespace = optional(config.KnowledgeNamespace)
	if len(config.Skills) > 0 {
		skills := config.Skills
		rendered.Skills = &skills
	}
	if len(config.Tags) > 0 {
		tags := config.Tags
		rendered.Tags = &tags
	}
	return rendered
}

// skillOf renders a skill for the wire.
func skillOf(skill store.Skill) Skill {
	rendered := Skill{
		Id:           skill.ID,
		Name:         skill.Name,
		Description:  skill.Description,
		Instructions: skill.Instructions,
		CreatedAt:    skill.CreatedAt,
		UpdatedAt:    skill.UpdatedAt,
	}
	if skill.DeadlineMs > 0 {
		deadline := skill.DeadlineMs
		rendered.DeadlineMs = &deadline
	}
	return rendered
}

// optional carries a string only when there is one, which is how an unset field stays
// unset on the way back out.
func optional(text string) *string {
	if text == "" {
		return nil
	}
	return &text
}
