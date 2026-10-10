package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"slices"

	"github.com/uptrace/bun"
)

// DraftOfTag is the tag that marks an agent config as a test copy: a temporary copy the
// dashboard saves of another config's unsaved changes, to run a test on, naming the config it
// copies. The dashboard writes it (DRAFT_OF_TAG = 'draft_of' in volt-dashboard
// src/components/dashboard/agents/lib/draft-copies.ts:9, set on the copy in
// use-test-draft.ts:85, branch connectors/agents-ui-e at 9441e26fd4) and removes the copy after
// the test, or sweeps it once it is 2 h old (STALE_COPY_MS, agent-copies.ts:117). The router has
// no field of its own for a copy, so it reads the tag.
const DraftOfTag = "draft_of"

// TestCopy reports whether the config is a test copy (DraftOfTag). A test copy answers no
// channel message and subscribes to no event: those belong to the live config it copies, the
// one channel and event connection's owner (AI-1049, AI-1048).
func (c AgentConfig) TestCopy() bool {
	return c.Tags[DraftOfTag] != ""
}

// notTestCopy is the WHERE clause that leaves test copies out of a select of agent_configs.
const notTestCopy = "coalesce(tags->>'" + DraftOfTag + "', '') = ''"

// channelOwnerLockSeed is the second argument to hashtextextended for the lock a channel
// connection's owner is decided under. Any value other than the 731 RecordAgentLog uses and
// definitionLockSeed (833) would do; 1049 is this rule's issue, AI-1049.
const channelOwnerLockSeed = 1049

// ChannelConnectionTakenError is a write that would have a second live agent config bind a
// channel connection as fixed. A channel connection, one whose connector reads messages (its
// manifest's channel block has messages, as slack_bot's has), answers each message with one
// agent, so it belongs to the live config that bound it first.
type ChannelConnectionTakenError struct {
	// Binding is the binding of the refused config that names the connection.
	Binding      string
	ConnectionID string
	ConnectorID  string
	// OwnerID and OwnerName are the live config that already binds the connection.
	OwnerID   string
	OwnerName string
}

func (e *ChannelConnectionTakenError) Error() string {
	return fmt.Sprintf("store: connection %s (%s) is bound by agent config %s already", e.ConnectionID, e.ConnectorID, e.OwnerID)
}

// refuseSecondChannelAgent refuses a write of a live config that newly binds, as fixed, a
// channel connection another live config binds already. kept are the bindings stored before
// this write, and keptCopy whether the stored config was a test copy: a save that keeps a
// binding the live config had is not a new bind, so a config that already shared a connection
// before this rule (the oldest of them answers, take in internal/channelbridge) can still be
// saved. A test copy binds what it likes, since it answers nothing.
//
// The decision runs under a transaction-scoped advisory lock on the customer and connection,
// so two routers saving two configs that bind one connection at once take turns: the second
// select runs after the first's commit, and under READ COMMITTED it sees that config
// (https://www.postgresql.org/docs/current/transaction-iso.html#XACT-READ-COMMITTED). The
// locks are taken in id order, so two writes binding the same two connections never wait on
// each other in a cycle.
func refuseSecondChannelAgent(ctx context.Context, tx bun.Tx, config *AgentConfig, kept []ConnectorBinding, keptCopy bool) error {
	if config.TestCopy() {
		return nil
	}
	named := map[string]string{}
	for _, binding := range config.Connectors {
		id := binding.Connection.ConnectionID
		if _, seen := named[id]; binding.Connection.Type != "fixed" || seen {
			continue
		}
		if !keptCopy && slices.ContainsFunc(kept, func(b ConnectorBinding) bool {
			return b.Connection.Type == "fixed" && b.Connection.ConnectionID == id
		}) {
			continue
		}
		named[id] = binding.Name
	}
	ids := make([]string, 0, len(named))
	for id := range named {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	for _, id := range ids {
		var connection ConnectorConnection
		err := tx.NewSelect().Model(&connection).Column("id", "connector_id", "definition_revision").
			Where("customer_id = ?", config.CustomerID).
			Where("id = ?", id).
			Where("deleted_at IS NULL").
			Scan(ctx)
		if errors.Is(err, sql.ErrNoRows) {
			// refuseUnbindable answers a connection that is not live.
			continue
		}
		if err != nil {
			return fmt.Errorf("store: channel connection owner: %w", err)
		}
		definition, err := connectorDefinition(ctx, tx, config.CustomerID, connection.ConnectorID, connection.DefinitionRevision)
		if errors.Is(err, ErrNoConnectorDefinition) {
			continue
		}
		if err != nil {
			return err
		}
		if definition.Manifest.Channel == nil || definition.Manifest.Channel.Messages.IsZero() {
			continue
		}
		if _, err := tx.ExecContext(ctx, "SELECT pg_advisory_xact_lock(hashtextextended(?, ?))",
			config.CustomerID+"/"+id, channelOwnerLockSeed); err != nil {
			return fmt.Errorf("store: channel connection owner: %w", err)
		}
		var owner AgentConfig
		err = tx.NewSelect().Model(&owner).Column("id", "name").
			Where("customer_id = ?", config.CustomerID).
			Where("id != ?", config.ID).
			Where("deleted_at IS NULL").
			Where(notTestCopy).
			Where(bindsFixedConnection, id).
			Order("created_at", "id").
			Limit(1).
			Scan(ctx)
		if errors.Is(err, sql.ErrNoRows) {
			continue
		}
		if err != nil {
			return fmt.Errorf("store: channel connection owner: %w", err)
		}
		return &ChannelConnectionTakenError{
			Binding: named[id], ConnectionID: id, ConnectorID: connection.ConnectorID,
			OwnerID: owner.ID, OwnerName: owner.Name,
		}
	}
	return nil
}
