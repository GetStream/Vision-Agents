package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"
)

// CreateAgentConfig stores a new config and fills in its id and timestamps.
func (s *Store) CreateAgentConfig(ctx context.Context, config *AgentConfig) error {
	if config.CustomerID == "" {
		return errors.New("store: customer id is required")
	}
	if config.Name == "" {
		return errors.New("store: an agent config needs a name")
	}

	config.ID = newID()
	now := time.Now().UTC()
	config.CreatedAt = now
	config.UpdatedAt = now
	config.DeletedAt = nil
	normalizeConfig(config)

	if _, err := s.db.NewInsert().Model(config).Exec(ctx); err != nil {
		return fmt.Errorf("store: create agent config: %w", err)
	}
	return nil
}

// UpdateAgentConfig replaces a config a customer holds. Every field is written, so an
// update is what the config now is rather than what changed about it.
func (s *Store) UpdateAgentConfig(ctx context.Context, config *AgentConfig) error {
	if config.CustomerID == "" || config.ID == "" {
		return errors.New("store: a customer and a config id are required")
	}
	if config.Name == "" {
		return errors.New("store: an agent config needs a name")
	}

	config.UpdatedAt = time.Now().UTC()
	normalizeConfig(config)

	result, err := s.db.NewUpdate().Model(config).
		Column("name", "stt", "tts", "voice", "llm", "subagent", "search", "instructions",
			"greeting", "skills", "keyterms", "knowledge_namespace", "tags", "sync_hash", "updated_at").
		Where("id = ?", config.ID).
		Where("customer_id = ?", config.CustomerID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: update agent config: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: update agent config: %w", err)
	}
	if affected == 0 {
		return unknownAgentConfig(config.ID)
	}
	return nil
}

// DeleteAgentConfig marks a config as gone. The row stays, because the calls that ran
// under it still name it.
func (s *Store) DeleteAgentConfig(ctx context.Context, customerID, id string) error {
	if customerID == "" || id == "" {
		return errors.New("store: a customer and a config id are required")
	}

	result, err := s.db.NewUpdate().Model((*AgentConfig)(nil)).
		Set("deleted_at = ?", time.Now().UTC()).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: delete agent config: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: delete agent config: %w", err)
	}
	if affected == 0 {
		return unknownAgentConfig(id)
	}
	return nil
}

// AgentConfig returns one config a customer holds.
func (s *Store) AgentConfig(ctx context.Context, customerID, id string) (AgentConfig, error) {
	if customerID == "" || id == "" {
		return AgentConfig{}, errors.New("store: a customer and a config id are required")
	}

	var config AgentConfig
	err := s.db.NewSelect().Model(&config).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return AgentConfig{}, unknownAgentConfig(id)
	}
	if err != nil {
		return AgentConfig{}, fmt.Errorf("store: agent config: %w", err)
	}
	return config, nil
}

// AgentConfigByName returns the config a customer holds under this name.
func (s *Store) AgentConfigByName(ctx context.Context, customerID, name string) (AgentConfig, bool, error) {
	if customerID == "" || name == "" {
		return AgentConfig{}, false, errors.New("store: a customer and a config name are required")
	}

	var config AgentConfig
	err := s.db.NewSelect().Model(&config).
		Where("customer_id = ?", customerID).
		Where("name = ?", name).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return AgentConfig{}, false, nil
	}
	if err != nil {
		return AgentConfig{}, false, fmt.Errorf("store: agent config by name: %w", err)
	}
	return config, true, nil
}

// CustomerAgentConfigs returns the configs a customer holds, newest first.
func (s *Store) CustomerAgentConfigs(ctx context.Context, customerID string) ([]AgentConfig, error) {
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}

	var configs []AgentConfig
	err := s.db.NewSelect().Model(&configs).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Order("created_at DESC").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: customer agent configs: %w", err)
	}
	return configs, nil
}

// CreateSkill stores a new skill and fills in its id and timestamps.
func (s *Store) CreateSkill(ctx context.Context, skill *Skill) error {
	if skill.CustomerID == "" {
		return errors.New("store: customer id is required")
	}
	if skill.Name == "" {
		return errors.New("store: a skill needs a name")
	}

	skill.ID = newID()
	now := time.Now().UTC()
	skill.CreatedAt = now
	skill.UpdatedAt = now
	skill.DeletedAt = nil

	if _, err := s.db.NewInsert().Model(skill).Exec(ctx); err != nil {
		return fmt.Errorf("store: create skill: %w", err)
	}
	return nil
}

// UpdateSkill replaces a skill a customer holds.
func (s *Store) UpdateSkill(ctx context.Context, skill *Skill) error {
	if skill.CustomerID == "" || skill.ID == "" {
		return errors.New("store: a customer and a skill id are required")
	}
	if skill.Name == "" {
		return errors.New("store: a skill needs a name")
	}

	skill.UpdatedAt = time.Now().UTC()

	result, err := s.db.NewUpdate().Model(skill).
		Column("name", "description", "instructions", "deadline_ms", "updated_at").
		Where("id = ?", skill.ID).
		Where("customer_id = ?", skill.CustomerID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: update skill: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: update skill: %w", err)
	}
	if affected == 0 {
		return unknownSkill(skill.ID)
	}
	return nil
}

// DeleteSkill marks a skill as gone.
func (s *Store) DeleteSkill(ctx context.Context, customerID, id string) error {
	if customerID == "" || id == "" {
		return errors.New("store: a customer and a skill id are required")
	}

	result, err := s.db.NewUpdate().Model((*Skill)(nil)).
		Set("deleted_at = ?", time.Now().UTC()).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: delete skill: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: delete skill: %w", err)
	}
	if affected == 0 {
		return unknownSkill(id)
	}
	return nil
}

// Skill returns one skill a customer holds.
func (s *Store) Skill(ctx context.Context, customerID, id string) (Skill, error) {
	if customerID == "" || id == "" {
		return Skill{}, errors.New("store: a customer and a skill id are required")
	}

	var skill Skill
	err := s.db.NewSelect().Model(&skill).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return Skill{}, unknownSkill(id)
	}
	if err != nil {
		return Skill{}, fmt.Errorf("store: skill: %w", err)
	}
	return skill, nil
}

// CustomerSkills returns the skills a customer holds, newest first.
func (s *Store) CustomerSkills(ctx context.Context, customerID string) ([]Skill, error) {
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}

	var skills []Skill
	err := s.db.NewSelect().Model(&skills).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Order("created_at DESC").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: customer skills: %w", err)
	}
	return skills, nil
}

// SkillsNamed returns the customer's skills with any of these names. A name nobody
// defined is simply absent, so the caller can report which ones it could not find.
func (s *Store) SkillsNamed(ctx context.Context, customerID string, names []string) ([]Skill, error) {
	if customerID == "" {
		return nil, errors.New("store: customer id is required")
	}
	if len(names) == 0 {
		return nil, nil
	}

	var skills []Skill
	err := s.db.NewSelect().Model(&skills).
		Where("customer_id = ?", customerID).
		Where("name IN (?)", bun.In(names)).
		Where("deleted_at IS NULL").
		Scan(ctx)
	if err != nil {
		return nil, fmt.Errorf("store: skills named: %w", err)
	}
	return skills, nil
}

// normalizeConfig fills in the JSONB columns a nil slice or map would write as null,
// which the columns are not.
func normalizeConfig(config *AgentConfig) {
	if config.Skills == nil {
		config.Skills = []string{}
	}
	if config.Keyterms == nil {
		config.Keyterms = []string{}
	}
	if config.Tags == nil {
		config.Tags = map[string]string{}
	}
}

func unknownAgentConfig(id string) error {
	return fmt.Errorf("store: there is no agent config %s", id)
}

func unknownSkill(id string) error {
	return fmt.Errorf("store: there is no skill %s", id)
}
