package appconfig

import (
	"context"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// found is a row looked up by name together with whether there was one, so that "this
// customer has no agent called that" is itself worth caching: a session opened by name
// asks the same question on every conversation.
type found[T any] struct {
	Value T    `json:"value"`
	Found bool `json:"found"`
}

// AgentConfig returns one config a customer holds.
func (s *Store) AgentConfig(ctx context.Context, customerID, id string) (store.AgentConfig, error) {
	if customerID == "" || id == "" {
		return s.db.AgentConfig(ctx, customerID, id)
	}
	return read(ctx, s, key("agent", customerID, id), func(ctx context.Context) (store.AgentConfig, error) {
		return s.db.AgentConfig(ctx, customerID, id)
	})
}

// AgentConfigByName returns the config a customer holds under this name.
func (s *Store) AgentConfigByName(ctx context.Context, customerID, name string) (store.AgentConfig, bool, error) {
	if customerID == "" || name == "" {
		return s.db.AgentConfigByName(ctx, customerID, name)
	}
	answer, err := read(ctx, s, key("agent-name", customerID, name),
		func(ctx context.Context) (found[store.AgentConfig], error) {
			config, exists, err := s.db.AgentConfigByName(ctx, customerID, name)
			return found[store.AgentConfig]{Value: config, Found: exists}, err
		})
	return answer.Value, answer.Found, err
}

// CreateAgentConfig stores a new config.
func (s *Store) CreateAgentConfig(ctx context.Context, config *store.AgentConfig) error {
	if err := s.db.CreateAgentConfig(ctx, config); err != nil {
		return err
	}
	// Only the name, because the id was minted by the insert and nothing can have asked
	// for it yet. The name may well have been asked for and not found.
	s.forget(ctx, key("agent-name", config.CustomerID, config.Name))
	return nil
}

// UpdateAgentConfig replaces a config a customer holds.
//
// The name it had before is dropped as well as the one it has now, because a rename
// leaves whoever asked for the old name holding a config that no longer answers to it.
func (s *Store) UpdateAgentConfig(ctx context.Context, config *store.AgentConfig) error {
	was, err := s.db.AgentConfig(ctx, config.CustomerID, config.ID)
	if err != nil {
		return err
	}
	if err := s.db.UpdateAgentConfig(ctx, config); err != nil {
		return err
	}
	s.forget(ctx,
		key("agent", config.CustomerID, config.ID),
		key("agent-name", config.CustomerID, config.Name),
		key("agent-name", config.CustomerID, was.Name))
	return nil
}

// AddConnectorBinding appends a binding to a config's connectors and nothing else
// (store.Store.AddConnectorBinding), and forgets the cached config under its id and its name.
func (s *Store) AddConnectorBinding(ctx context.Context, customerID, configID string, binding store.ConnectorBinding) (bool, error) {
	config, added, err := s.db.AddConnectorBinding(ctx, customerID, configID, binding)
	if err != nil || !added {
		return added, err
	}
	s.forget(ctx, key("agent", customerID, configID), key("agent-name", customerID, config.Name))
	return true, nil
}

// DeleteAgentConfig marks a config as gone.
func (s *Store) DeleteAgentConfig(ctx context.Context, customerID, id string) error {
	was, err := s.db.AgentConfig(ctx, customerID, id)
	if err != nil {
		return err
	}
	if err := s.db.DeleteAgentConfig(ctx, customerID, id); err != nil {
		return err
	}
	s.forget(ctx, key("agent", customerID, id), key("agent-name", customerID, was.Name))
	return nil
}

// SkillsNamed returns one config's skills with any of these names.
//
// One entry per name rather than one per set, because a name is what a write knows about
// and a set is not: there is no listing the sets some session once asked for. A name
// nothing defines is cached as absent, which is what a session naming a built-in asks on
// every conversation.
func (s *Store) SkillsNamed(ctx context.Context, customerID, configID string, names []string) ([]store.Skill, error) {
	if customerID == "" || configID == "" || len(names) == 0 {
		return s.db.SkillsNamed(ctx, customerID, configID, names)
	}

	skills := make([]store.Skill, 0, len(names))
	for _, name := range names {
		answer, err := read(ctx, s, key("skill", customerID, configID, name),
			func(ctx context.Context) (found[store.Skill], error) {
				matched, err := s.db.SkillsNamed(ctx, customerID, configID, []string{name})
				if err != nil || len(matched) == 0 {
					return found[store.Skill]{}, err
				}
				return found[store.Skill]{Value: matched[0], Found: true}, nil
			})
		if err != nil {
			return nil, err
		}
		if answer.Found {
			skills = append(skills, answer.Value)
		}
	}
	return skills, nil
}

// CreateSkill stores a new skill.
func (s *Store) CreateSkill(ctx context.Context, skill *store.Skill) error {
	if err := s.db.CreateSkill(ctx, skill); err != nil {
		return err
	}
	s.forget(ctx, key("skill", skill.CustomerID, skill.ConfigID, skill.Name))
	return nil
}

// UpdateSkill replaces a skill. An update may rename it or move it to another config, so
// what it was is dropped along with what it now is.
func (s *Store) UpdateSkill(ctx context.Context, skill *store.Skill) error {
	was, err := s.db.Skill(ctx, skill.CustomerID, skill.ID)
	if err != nil {
		return err
	}
	if err := s.db.UpdateSkill(ctx, skill); err != nil {
		return err
	}
	s.forget(ctx,
		key("skill", skill.CustomerID, skill.ConfigID, skill.Name),
		key("skill", was.CustomerID, was.ConfigID, was.Name))
	return nil
}

// DeleteSkill marks a skill as gone.
func (s *Store) DeleteSkill(ctx context.Context, customerID, id string) error {
	was, err := s.db.Skill(ctx, customerID, id)
	if err != nil {
		return err
	}
	if err := s.db.DeleteSkill(ctx, customerID, id); err != nil {
		return err
	}
	s.forget(ctx, key("skill", customerID, was.ConfigID, was.Name))
	return nil
}
