package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// CreateAgentConfig stores a new config and fills in its id and timestamps. Each connection
// it binds as fixed has to be live, and stays locked until the config is stored
// (lockBoundConnections). A channel connection another live config binds is refused with a
// *ChannelConnectionTakenError (refuseSecondChannelAgent).
func (s *Store) CreateAgentConfig(ctx context.Context, config *AgentConfig) error {
	if config.CustomerID == "" {
		return stack.Wrap(errors.New("store: customer id is required"))
	}
	if config.Name == "" {
		return stack.Wrap(errors.New("store: an agent config needs a name"))
	}

	config.ID = newID()
	now := time.Now().UTC()
	config.CreatedAt = now
	config.UpdatedAt = now
	config.DeletedAt = nil
	normalizeConfig(config)

	return stack.Wrap(s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		missing, err := lockBoundConnections(ctx, tx, config)
		if err != nil {
			return err
		}
		if err := refuseUnbindable(missing, nil); err != nil {
			return err
		}
		undefined, err := lockCustomDefinitions(ctx, tx, config.CustomerID, boundConnectors(config.Connectors))
		if err != nil {
			return err
		}
		if err := refuseUndefined(undefined, nil); err != nil {
			return err
		}
		if err := refuseSecondChannelAgent(ctx, tx, config, nil, false); err != nil {
			return err
		}
		if _, err := tx.NewInsert().Model(config).Exec(ctx); err != nil {
			if constraint(err) == "agent_configs_name_idx" {
				return ErrNameTaken
			}
			return fmt.Errorf("store: create agent config: %w", err)
		}
		return nil
	}))
}

// configColumns are the columns an update writes: everything about a config except who
// owns it and when it was created, which an update cannot change.
//
// It is named rather than written into the call so a test can hold it against the model.
// A field added to AgentConfig and forgotten here is stored on create and silently
// dropped on every update after, which reads as a setting that will not save.
var configColumns = []string{
	"name", "mode", "stt", "tts", "sts", "voice", "llm", "subagent",
	"video_source", "video_max_frames", "search", "instructions", "greeting", "greeting_mode", "guardrail",
	"skills", "plugins", "connectors", "plugin_events", "mcp_servers", "channels", "keyterms", "visible_tools", "knowledge_namespace", "sandbox", "sandbox_options", "harness", "tags",
	"dispatch_incoming_call", "dispatch_text", "episode_cards", "progressive_tools", "sync_hash", "updated_at",
}

// UpdateAgentConfig replaces a config a customer holds. Every field is written, so an
// update is what the config now is rather than what changed about it. A connection it binds
// as fixed is locked as CreateAgentConfig locks it, and has to be live unless the stored
// config binds it already: a forced delete leaves that binding behind on purpose, and a save
// that keeps it is not a new bind. A new bind of a channel connection another live config
// binds is refused as CreateAgentConfig refuses it.
func (s *Store) UpdateAgentConfig(ctx context.Context, config *AgentConfig) error {
	if config.CustomerID == "" || config.ID == "" {
		return stack.Wrap(errors.New("store: a customer and a config id are required"))
	}
	if config.Name == "" {
		return stack.Wrap(errors.New("store: an agent config needs a name"))
	}

	config.UpdatedAt = time.Now().UTC()
	normalizeConfig(config)

	return stack.Wrap(s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		missing, err := lockBoundConnections(ctx, tx, config)
		if err != nil {
			return err
		}
		undefined, err := lockCustomDefinitions(ctx, tx, config.CustomerID, boundConnectors(config.Connectors))
		if err != nil {
			return err
		}
		var stored AgentConfig
		if len(missing) > 0 || len(undefined) > 0 || bindsAnyFixed(config.Connectors) {
			err := tx.NewSelect().Model(&stored).Column("connectors", "tags").
				Where("id = ?", config.ID).
				Where("customer_id = ?", config.CustomerID).
				Where("deleted_at IS NULL").
				Scan(ctx)
			if err != nil && !errors.Is(err, sql.ErrNoRows) {
				return fmt.Errorf("store: update agent config: %w", err)
			}
		}
		if err := refuseUnbindable(missing, stored.Connectors); err != nil {
			return err
		}
		if err := refuseUndefined(undefined, stored.Connectors); err != nil {
			return err
		}
		if err := refuseSecondChannelAgent(ctx, tx, config, stored.Connectors, stored.TestCopy()); err != nil {
			return err
		}
		result, err := tx.NewUpdate().Model(config).
			Column(configColumns...).
			Where("id = ?", config.ID).
			Where("customer_id = ?", config.CustomerID).
			Where("deleted_at IS NULL").
			Exec(ctx)
		if constraint(err) == "agent_configs_name_idx" {
			return ErrNameTaken
		}
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
	}))
}

// AddConnectorBinding appends binding to a live config's connectors and writes that column
// alone, with updated_at, so an edit of any other column that lands meanwhile is kept. The
// config row is locked FOR UPDATE for the read and the write, so two writers of connectors
// take turns. A fixed binding's connection is locked as UpdateAgentConfig locks it and must
// be live. added is false, and nothing is written, when the config already has a binding of
// that name. The config is returned as it is after, for a caller that forgets a cached copy.
// For router plugins migrate (T61 in acceleration/docs/connectors/subtasks.md on
// connectors/planning).
func (s *Store) AddConnectorBinding(ctx context.Context, customerID, configID string, binding ConnectorBinding) (config AgentConfig, added bool, err error) {
	if customerID == "" || configID == "" || binding.Name == "" {
		return AgentConfig{}, false, stack.Wrap(errors.New("store: a customer, a config and a binding name are required"))
	}
	err = s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		missing, err := lockBoundConnections(ctx, tx, &AgentConfig{CustomerID: customerID, Connectors: []ConnectorBinding{binding}})
		if err != nil {
			return err
		}
		if err := refuseUnbindable(missing, nil); err != nil {
			return err
		}
		undefined, err := lockCustomDefinitions(ctx, tx, customerID, []string{binding.ConnectorID})
		if err != nil {
			return err
		}
		if err := refuseUndefined(undefined, nil); err != nil {
			return err
		}
		// The channel owner's lock before the row's, the order UpdateAgentConfig takes them in.
		// refuseSecondChannelAgent takes it again below, which a transaction that holds it is
		// granted at once (https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS).
		if binding.Connection.Type == "fixed" {
			if _, _, err := lockChannelOwner(ctx, tx, customerID, binding.Connection.ConnectionID); err != nil {
				return err
			}
		}
		err = tx.NewSelect().Model(&config).
			Where("id = ?", configID).
			Where("customer_id = ?", customerID).
			Where("deleted_at IS NULL").
			For("UPDATE").
			Scan(ctx)
		if errors.Is(err, sql.ErrNoRows) {
			return unknownAgentConfig(configID)
		}
		if err != nil {
			return fmt.Errorf("store: add connector binding: %w", err)
		}
		if slices.ContainsFunc(config.Connectors, func(b ConnectorBinding) bool { return b.Name == binding.Name }) {
			return nil
		}
		adding := config
		adding.Connectors = []ConnectorBinding{binding}
		if err := refuseSecondChannelAgent(ctx, tx, &adding, config.Connectors, false); err != nil {
			return err
		}
		config.Connectors = append(config.Connectors, binding)
		config.UpdatedAt = time.Now().UTC()
		_, err = tx.NewUpdate().Model(&config).
			Column("connectors", "updated_at").
			Where("id = ?", configID).
			Where("customer_id = ?", customerID).
			Exec(ctx)
		if err != nil {
			return fmt.Errorf("store: add connector binding: %w", err)
		}
		added = true
		return nil
	})
	if err != nil {
		return AgentConfig{}, false, stack.Wrap(err)
	}
	return config, added, nil
}

// lockBoundConnections locks each live connection of the config's customer that the config
// binds as fixed until the transaction ends, and returns the ids it binds that are not live.
// It makes a bind and an unforced delete (DeleteUnboundConnectorConnection, which locks the
// connection FOR UPDATE first) wait for each other, so one of them always sees the other:
//
//   - A delete that locked the connection first is waited for. Under READ COMMITTED the
//     locking read then re-checks its WHERE against the row as the delete left it, so a
//     deleted connection is not returned and the bind is refused
//     (https://www.postgresql.org/docs/current/transaction-iso.html#XACT-READ-COMMITTED).
//   - A delete that comes second waits for this transaction, and then reads the binding it
//     wrote.
//
// FOR KEY SHARE, not FOR SHARE: it conflicts with the delete's FOR UPDATE and not with the
// FOR NO KEY UPDATE a plain UPDATE of a non-key column takes (table 13.3,
// https://www.postgresql.org/docs/current/explicit-locking.html#LOCKING-ROWS), so a config
// save does not hold up a credential save. The connection is locked before the config row,
// as in the delete, so the two never wait on each other in a cycle.
func lockBoundConnections(ctx context.Context, tx bun.Tx, config *AgentConfig) ([]string, error) {
	var bound []string
	for _, binding := range config.Connectors {
		if binding.Connection.Type == "fixed" && !slices.Contains(bound, binding.Connection.ConnectionID) {
			bound = append(bound, binding.Connection.ConnectionID)
		}
	}
	if len(bound) == 0 {
		return nil, nil
	}
	var live []string
	err := tx.NewSelect().Model((*ConnectorConnection)(nil)).Column("id").
		Where("customer_id = ?", config.CustomerID).
		Where("id IN (?)", bun.In(bound)).
		Where("deleted_at IS NULL").
		For("KEY SHARE").
		Scan(ctx, &live)
	if err != nil {
		return nil, fmt.Errorf("store: lock bound connector connections: %w", err)
	}
	return slices.DeleteFunc(bound, func(id string) bool { return slices.Contains(live, id) }), nil
}

// refuseUnbindable refuses a fixed binding to a connection that is not live, unless kept, the
// bindings stored before this write, has it already.
func refuseUnbindable(missing []string, kept []ConnectorBinding) error {
	for _, id := range missing {
		if !slices.ContainsFunc(kept, func(binding ConnectorBinding) bool {
			return binding.Connection.Type == "fixed" && binding.Connection.ConnectionID == id
		}) {
			return stack.Wrap(fmt.Errorf("%w: %s", ErrNoConnectorConnection, id))
		}
	}
	return nil
}

// bindsAnyFixed reports whether a binding has a fixed connection.
func bindsAnyFixed(bindings []ConnectorBinding) bool {
	return slices.ContainsFunc(bindings, func(binding ConnectorBinding) bool { return binding.Connection.Type == "fixed" })
}

// boundConnectors are the connector ids bindings name.
func boundConnectors(bindings []ConnectorBinding) []string {
	ids := make([]string, 0, len(bindings))
	for _, binding := range bindings {
		ids = append(ids, binding.ConnectorID)
	}
	return ids
}

// refuseUndefined refuses a binding to a custom connector the customer has no definition of,
// unless kept, the bindings stored before this write, names it already: a forced connector
// delete leaves its bindings behind on purpose (DeleteConnectorDefinition), and a save that
// keeps one is not a new bind, as refuseUnbindable keeps a forced connection delete's.
func refuseUndefined(undefined []string, kept []ConnectorBinding) error {
	for _, id := range undefined {
		if !slices.ContainsFunc(kept, func(binding ConnectorBinding) bool { return binding.ConnectorID == id }) {
			return stack.Wrap(fmt.Errorf("%w: %s", ErrNoConnectorDefinition, id))
		}
	}
	return nil
}

// DeleteAgentConfig marks a config as gone. The row stays, because the calls that ran
// under it still name it.
func (s *Store) DeleteAgentConfig(ctx context.Context, customerID, id string) error {
	if customerID == "" || id == "" {
		return stack.Wrap(errors.New("store: a customer and a config id are required"))
	}

	result, err := s.db.NewUpdate().Model((*AgentConfig)(nil)).
		Set("deleted_at = ?", time.Now().UTC()).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete agent config: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete agent config: %w", err))
	}
	if affected == 0 {
		return unknownAgentConfig(id)
	}
	return nil
}

// AgentConfig returns one config a customer holds.
func (s *Store) AgentConfig(ctx context.Context, customerID, id string) (AgentConfig, error) {
	if customerID == "" || id == "" {
		return AgentConfig{}, stack.Wrap(errors.New("store: a customer and a config id are required"))
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
		return AgentConfig{}, stack.Wrap(fmt.Errorf("store: agent config: %w", err))
	}
	return config, nil
}

// AgentConfigOwner returns a config by id alone, without being told whose it is.
//
// Every other read here is scoped to a customer, because every other caller already knows
// which customer it is acting for. This one is for the case where the id is all there is:
// an agent channel that names the config answering in it, whose owner has to be worked out
// rather than taken from a header. Finding out who owns a config is the whole point of it,
// so a caller that knows the customer should use AgentConfig instead.
func (s *Store) AgentConfigOwner(ctx context.Context, id string) (AgentConfig, error) {
	if id == "" {
		return AgentConfig{}, errors.New("store: a config id is required")
	}

	var config AgentConfig
	err := s.db.NewSelect().Model(&config).
		Where("id = ?", id).
		Where("deleted_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return AgentConfig{}, unknownAgentConfig(id)
	}
	if err != nil {
		return AgentConfig{}, fmt.Errorf("store: agent config owner: %w", err)
	}
	return config, nil
}

// AgentConfigByName returns the config a customer holds under this name.
func (s *Store) AgentConfigByName(ctx context.Context, customerID, name string) (AgentConfig, bool, error) {
	if customerID == "" || name == "" {
		return AgentConfig{}, false, stack.Wrap(errors.New("store: a customer and a config name are required"))
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
		return AgentConfig{}, false, stack.Wrap(fmt.Errorf("store: agent config by name: %w", err))
	}
	return config, true, nil
}

// CustomerAgentConfigs returns the configs a customer holds, newest first.
func (s *Store) CustomerAgentConfigs(ctx context.Context, customerID string) ([]AgentConfig, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}

	var configs []AgentConfig
	err := s.db.NewSelect().Model(&configs).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Order("created_at DESC").
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: customer agent configs: %w", err))
	}
	return configs, nil
}

// CreateSkill stores a new skill and fills in its id and timestamps.
func (s *Store) CreateSkill(ctx context.Context, skill *Skill) error {
	if skill.CustomerID == "" {
		return stack.Wrap(errors.New("store: customer id is required"))
	}
	if skill.ConfigID == "" {
		return stack.Wrap(errors.New("store: a skill belongs to an agent config"))
	}
	if skill.Name == "" {
		return stack.Wrap(errors.New("store: a skill needs a name"))
	}

	skill.ID = newID()
	now := time.Now().UTC()
	skill.CreatedAt = now
	skill.UpdatedAt = now
	skill.DeletedAt = nil

	if _, err := s.db.NewInsert().Model(skill).Exec(ctx); err != nil {
		if constraint(err) == "skills_name_idx" {
			return stack.Wrap(ErrNameTaken)
		}
		return stack.Wrap(fmt.Errorf("store: create skill: %w", err))
	}
	return nil
}

// UpdateSkill replaces a skill a customer holds.
func (s *Store) UpdateSkill(ctx context.Context, skill *Skill) error {
	if skill.CustomerID == "" || skill.ID == "" {
		return stack.Wrap(errors.New("store: a customer and a skill id are required"))
	}
	if skill.Name == "" {
		return stack.Wrap(errors.New("store: a skill needs a name"))
	}

	skill.UpdatedAt = time.Now().UTC()

	result, err := s.db.NewUpdate().Model(skill).
		Column("config_id", "name", "description", "instructions", "capture_video", "deadline_ms", "updated_at").
		Where("id = ?", skill.ID).
		Where("customer_id = ?", skill.CustomerID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if constraint(err) == "skills_name_idx" {
		return stack.Wrap(ErrNameTaken)
	}
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: update skill: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: update skill: %w", err))
	}
	if affected == 0 {
		return unknownSkill(skill.ID)
	}
	return nil
}

// DeleteSkill marks a skill as gone.
func (s *Store) DeleteSkill(ctx context.Context, customerID, id string) error {
	if customerID == "" || id == "" {
		return stack.Wrap(errors.New("store: a customer and a skill id are required"))
	}

	result, err := s.db.NewUpdate().Model((*Skill)(nil)).
		Set("deleted_at = ?", time.Now().UTC()).
		Where("id = ?", id).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete skill: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete skill: %w", err))
	}
	if affected == 0 {
		return unknownSkill(id)
	}
	return nil
}

// Skill returns one skill a customer holds.
func (s *Store) Skill(ctx context.Context, customerID, id string) (Skill, error) {
	if customerID == "" || id == "" {
		return Skill{}, stack.Wrap(errors.New("store: a customer and a skill id are required"))
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
		return Skill{}, stack.Wrap(fmt.Errorf("store: skill: %w", err))
	}
	return skill, nil
}

// CustomerSkills returns the skills a customer holds, newest first. A config id narrows
// them to that agent's own; empty returns every skill across all of them.
func (s *Store) CustomerSkills(ctx context.Context, customerID, configID string) ([]Skill, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}

	var skills []Skill
	query := s.db.NewSelect().Model(&skills).
		Where("customer_id = ?", customerID).
		Where("deleted_at IS NULL")
	if configID != "" {
		query = query.Where("config_id = ?", configID)
	}
	if err := query.Order("created_at DESC").Scan(ctx); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: customer skills: %w", err))
	}
	return skills, nil
}

// SkillsNamed returns one config's skills with any of these names. A name nobody defined
// is simply absent, so the caller can report which ones it could not find.
func (s *Store) SkillsNamed(ctx context.Context, customerID, configID string, names []string) ([]Skill, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}
	// Skills belong to a config, so a session that was not created from one reaches
	// nothing here and takes the built-in set.
	if configID == "" || len(names) == 0 {
		return nil, nil
	}

	var skills []Skill
	err := s.db.NewSelect().Model(&skills).
		Where("customer_id = ?", customerID).
		Where("config_id = ?", configID).
		Where("name IN (?)", bun.In(names)).
		Where("deleted_at IS NULL").
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: skills named: %w", err))
	}
	return skills, nil
}

// normalizeConfig fills in the JSONB columns a nil slice or map would write as null,
// which the columns are not, and the mode a caller that predates it leaves empty.
func normalizeConfig(config *AgentConfig) {
	if config.VideoMaxFrames == 0 {
		config.VideoMaxFrames = 1
	}
	if config.Mode == "" {
		config.Mode = AgentModeVoice
	}
	if config.GreetingMode == "" {
		config.GreetingMode = GreetingExact
	}
	if config.Skills == nil {
		config.Skills = []string{}
	}
	if config.Plugins == nil {
		config.Plugins = []PluginEntry{}
	}
	if config.Connectors == nil {
		config.Connectors = []ConnectorBinding{}
	}
	if config.PluginEvents == nil {
		config.PluginEvents = []PluginEvent{}
	}
	if config.MCPServers == nil {
		config.MCPServers = []MCPServer{}
	}
	if config.Keyterms == nil {
		config.Keyterms = []string{}
	}
	if config.VisibleTools == nil {
		config.VisibleTools = []string{}
	}
	if config.Tags == nil {
		config.Tags = map[string]string{}
	}
}

// ErrNoAgentConfig is a config id the customer holds no live config by. A sentinel, so a
// caller can tell a deleted config from the database failing. The text is the one these
// errors always had.
var ErrNoAgentConfig = errors.New("store: there is no agent config")

// ErrNoSkill is a skill id the customer holds no live skill by, as ErrNoAgentConfig is for
// a config.
var ErrNoSkill = errors.New("store: there is no skill")

// ErrNameTaken says a create or a rename asked for a name that another live record of the
// same kind already has: an agent config, a router config or a voice of the same customer,
// or a skill of the same agent config. The name is the writer's to change, so unlike the
// database failing it is the caller's to fix.
var ErrNameTaken = errors.New("store: the name is taken")

func unknownAgentConfig(id string) error {
	return stack.Wrap(fmt.Errorf("%w %s", ErrNoAgentConfig, id))
}

func unknownSkill(id string) error {
	return stack.Wrap(fmt.Errorf("%w %s", ErrNoSkill, id))
}
