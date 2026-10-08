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

// What a change was made to (AuditEntry.ResourceType). Configuration, all of it: what an
// agent does while it runs is traffic and is kept elsewhere
// (20261008120000_audit_log.sql).
const (
	AuditAgentConfig  = "agent_config"
	AuditSkill        = "skill"
	AuditKnowledge    = "knowledge"
	AuditKnowledgeURL = "knowledge_url"
	AuditRouterConfig = "router_config"
	AuditPlugin       = "plugin"
	AuditPolicy       = "policy"
)

// auditResourceTypes is every resource a change may be recorded against, which is the
// migration's CHECK as Go can read it.
var auditResourceTypes = []string{
	AuditAgentConfig, AuditSkill, AuditKnowledge, AuditKnowledgeURL,
	AuditRouterConfig, AuditPlugin, AuditPolicy,
}

// What happened to it (AuditEntry.Action). A sync is kept apart from an update because it
// is the one change a directory made rather than a person: it is what a later sync measures
// the dashboard's edits against.
const (
	AuditCreated = "created"
	AuditUpdated = "updated"
	AuditDeleted = "deleted"
	AuditSynced  = "synced"
)

var auditActions = []string{AuditCreated, AuditUpdated, AuditDeleted, AuditSynced}

// Which client made it (AuditEntry.Source), from X-Stream-Client.
const (
	AuditSourceDashboard = "dashboard"
	AuditSourceCLI       = "cli"
	AuditSourceSDK       = "sdk"
	// AuditSourceAPI is a caller that named no client. It reached the API directly, which
	// is all that can be said about it.
	AuditSourceAPI = "api"
)

var auditSources = []string{AuditSourceDashboard, AuditSourceCLI, AuditSourceSDK, AuditSourceAPI}

// AuditSource reads a client's name into a source, falling back to the API for one that
// named nothing or named something nobody knows.
func AuditSource(client string) string {
	if slices.Contains(auditSources, client) {
		return client
	}
	return AuditSourceAPI
}

// How many audit rows are handed back at once. Neither is measured; they are the sizes
// every other list here uses.
const (
	defaultAuditEntryLimit = 25
	maxAuditEntryLimit     = 200
)

// AuditChange is one field that moved, with what it held before and holds now. The two
// values are the field as the API renders it, so a client reads a change in the shape it
// reads the resource.
type AuditChange struct {
	Field  string `json:"field"`
	Before any    `json:"before,omitempty"`
	After  any    `json:"after,omitempty"`
}

// AuditEntry is one change somebody made to the app's configuration.
type AuditEntry struct {
	bun.BaseModel `bun:"table:audit_log,alias:al"`

	ID           string `bun:"id,pk"`
	CustomerID   string `bun:"customer_id,notnull"`
	ResourceType string `bun:"resource_type,notnull"`
	ResourceID   string `bun:"resource_id,notnull"`
	ResourceName string `bun:"resource_name,notnull"`
	// AgentID is the agent the change was to or under, empty for a resource that belongs
	// to no agent.
	AgentID   string `bun:"agent_id,notnull"`
	Action    string `bun:"action,notnull"`
	Source    string `bun:"source,notnull"`
	ActorID   string `bun:"actor_id,notnull"`
	ActorName string `bun:"actor_name,notnull"`
	RequestID string `bun:"request_id,notnull"`
	// Changes is never empty: a write that moved nothing records no entry.
	Changes   []AuditChange `bun:"changes,type:jsonb,notnull"`
	CreatedAt time.Time     `bun:"created_at,notnull"`
}

// AuditEntryFilter picks a customer's audit rows, a page at a time. The fields are ANDed,
// and each empty one is every row.
type AuditEntryFilter struct {
	ResourceType string
	ResourceID   string
	AgentID      string
	Source       string
	// Actions keeps only these actions. Empty is every action.
	Actions []string
	// Since keeps only what was recorded strictly after it, which is how a sync asks for
	// the changes made since the last one.
	Since time.Time
	Limit int
	// After is the last row of the previous page.
	After *AuditPosition
}

// AuditEntryLimit is the page size an audit query uses for the limit asked for.
// AuditEntries returns one row more than this.
func AuditEntryLimit(asked int) int {
	return clampLimit(asked, defaultAuditEntryLimit, maxAuditEntryLimit)
}

// RecordAuditEntry stores one change, at now.
func (s *Store) RecordAuditEntry(ctx context.Context, entry *AuditEntry) error {
	if entry.CustomerID == "" || entry.ResourceID == "" {
		return stack.Wrap(errors.New("store: an audit entry needs a customer and a resource"))
	}
	if !slices.Contains(auditResourceTypes, entry.ResourceType) {
		return stack.Wrap(fmt.Errorf("store: %q is not an audited resource", entry.ResourceType))
	}
	if !slices.Contains(auditActions, entry.Action) {
		return stack.Wrap(fmt.Errorf("store: %q is not an audit action", entry.Action))
	}
	// A sync is recorded whether or not it moved anything: what it marks is the moment the
	// directory and the stored config agreed, which is what the next sync measures the
	// edits made since against. Every other action is a change, and a change that changed
	// nothing is not one.
	if len(entry.Changes) == 0 && entry.Action != AuditSynced {
		return stack.Wrap(errors.New("store: an audit entry records what moved, and nothing did"))
	}
	if entry.Changes == nil {
		entry.Changes = []AuditChange{}
	}
	entry.ID = newID()
	entry.Source = AuditSource(entry.Source)
	entry.CreatedAt = time.Now().UTC().Truncate(time.Microsecond)
	if _, err := s.db.NewInsert().Model(entry).Exec(ctx); err != nil {
		return stack.Wrap(fmt.Errorf("store: record audit entry: %w", err))
	}
	return nil
}

// AuditEntries lists the customer's changes, newest first, one more than
// AuditEntryLimit(filter.Limit). A deleted resource's rows are listed too: that is the
// point of keeping them.
func (s *Store) AuditEntries(ctx context.Context, customerID string, filter AuditEntryFilter) ([]AuditEntry, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: a customer id is required"))
	}
	entries := []AuditEntry{}
	query := s.db.NewSelect().Model(&entries).Where("customer_id = ?", customerID)
	if filter.ResourceType != "" {
		query = query.Where("resource_type = ?", filter.ResourceType)
	}
	if filter.ResourceID != "" {
		query = query.Where("resource_id = ?", filter.ResourceID)
	}
	if filter.AgentID != "" {
		query = query.Where("agent_id = ?", filter.AgentID)
	}
	if filter.Source != "" {
		query = query.Where("source = ?", filter.Source)
	}
	if len(filter.Actions) > 0 {
		query = query.Where("action IN (?)", bun.In(filter.Actions))
	}
	if !filter.Since.IsZero() {
		query = query.Where("created_at > ?", filter.Since)
	}
	if after := filter.After; after != nil {
		query = query.Where("(created_at, id) < (?, ?)", after.CreatedAt, after.ID)
	}
	err := query.
		Order("created_at DESC", "id DESC").
		Limit(AuditEntryLimit(filter.Limit) + 1).
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: list audit entries: %w", err))
	}
	return entries, nil
}

// LatestAuditEntry returns the newest row the filter matches, and whether there was one. It
// is what a sync reads to find when the agent was last synced, and what resolves the id a
// client acknowledges changes up to.
func (s *Store) LatestAuditEntry(ctx context.Context, customerID string, filter AuditEntryFilter) (AuditEntry, bool, error) {
	filter.Limit = 1
	filter.After = nil
	found, err := s.AuditEntries(ctx, customerID, filter)
	if err != nil {
		return AuditEntry{}, false, err
	}
	if len(found) == 0 {
		return AuditEntry{}, false, nil
	}
	return found[0], true, nil
}

// AuditEntryByID returns one of the customer's audit rows.
func (s *Store) AuditEntryByID(ctx context.Context, customerID, id string) (AuditEntry, bool, error) {
	if customerID == "" || id == "" {
		return AuditEntry{}, false, stack.Wrap(errors.New("store: a customer and an entry id are required"))
	}
	var entry AuditEntry
	err := s.db.NewSelect().Model(&entry).
		Where("customer_id = ?", customerID).
		Where("id = ?", id).
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return AuditEntry{}, false, nil
	}
	if err != nil {
		return AuditEntry{}, false, stack.Wrap(fmt.Errorf("store: audit entry: %w", err))
	}
	return entry, true, nil
}
