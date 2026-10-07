package api

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// errNoAudit is what the audit paths say on a deployment without a database. The log is
// rows; there is nothing to serve without somewhere to keep them.
var errNoAudit = notConfigured("the audit log is not available: no database configured")

// AuditChange is one field that moved.
type AuditChange struct {
	Field  string `json:"field" doc:"The field as the resource's own schema names it."`
	Before any    `json:"before,omitempty" doc:"What it held, in the shape the resource is read in. Absent when it held nothing, which is what a created resource's fields all did."`
	After  any    `json:"after,omitempty" doc:"What it holds now. Absent when it now holds nothing, which is what a deleted resource's fields all do."`
}

func (*AuditChange) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One field that moved, with what it held before and holds now, each " +
		"in the shape the resource itself is read in."
	return schema
}

// AuditEntry is one change somebody made to the app's configuration.
type AuditEntry struct {
	ID           string            `json:"id"`
	ResourceType AuditResourceType `json:"resource_type"`
	ResourceID   string            `json:"resource_id" doc:"The resource, which may since have been deleted."`
	ResourceName string            `json:"resource_name,omitempty" doc:"What it was called when it changed, for a resource that has a name."`
	AgentID      string            `json:"agent_id,omitempty" doc:"The agent the change was to or under. Absent for a resource that belongs to no agent, such as a router."`
	Action       AuditAction       `json:"action"`
	Source       AuditSource       `json:"source"`
	ActorID      string            `json:"actor_id,omitempty" doc:"Who made it, as their client named them. Absent for a change nobody signed, such as a process syncing on startup."`
	ActorName    string            `json:"actor_name,omitempty" doc:"Their name, as their client named them. Never an email address: the router keeps none."`
	RequestID    string            `json:"request_id,omitempty" doc:"The X-Request-Id of the request that made it."`
	Changes      []AuditChange     `json:"changes" nullable:"false" doc:"Every field that moved. A write that moved nothing is not recorded at all, so this is empty only on a synced entry, which marks the moment a directory and an agent agreed whether or not anything moved."`
	CreatedAt    time.Time         `json:"created_at"`
}

func (*AuditEntry) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One change somebody made to the app's configuration: what changed, " +
		"who changed it, which client they used, and the before and after of every field that " +
		"moved. Only configuration is recorded -- agents, skills, knowledge, routers, plugins " +
		"and policies -- never what an agent did while it ran."
	return schema
}

// AuditResourceType is what a change was made to.
type AuditResourceType string

func (AuditResourceType) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "AuditResourceType",
		"What the change was made to. All of them are configuration: what an agent does while "+
			"it runs is traffic, and is read from the sessions and the logs instead.",
		store.AuditAgentConfig, store.AuditSkill, store.AuditKnowledge, store.AuditKnowledgeURL,
		store.AuditRouterConfig, store.AuditPlugin, store.AuditPolicy)
}

// AuditAction is what happened to it.
type AuditAction string

func (AuditAction) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "AuditAction",
		"created, updated or deleted for a change somebody made one at a time. synced is a "+
			"whole agent directory written at once by POST /v1/agents/sync, and is what a later "+
			"sync measures the edits made since against.",
		store.AuditCreated, store.AuditUpdated, store.AuditDeleted, store.AuditSynced)
}

// AuditSource is which client made the change.
type AuditSource string

func (AuditSource) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "AuditSource",
		"Which client made it, from the X-Stream-Client header. api is a caller that named no "+
			"client: it reached the API directly, which is all that can be said about it.",
		store.AuditSourceDashboard, store.AuditSourceCLI, store.AuditSourceSDK, store.AuditSourceAPI)
}

// AuditQuery is which of the app's changes to list.
type AuditQuery struct {
	Filter *AuditFilter `json:"filter,omitempty"`
	Limit  int          `json:"limit,omitempty" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 25."`
	Cursor string       `json:"cursor,omitempty" doc:"The next_cursor of the previous page, sent with the same filter. Omitted is the first page."`
}

func (*AuditQuery) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.AdditionalProperties = false
	return schema
}

// AuditFilter narrows an audit query. Fields are ANDed.
type AuditFilter struct {
	ResourceType *AuditResourceType `json:"resource_type,omitempty"`
	ResourceID   *string            `json:"resource_id,omitempty" doc:"One resource's own history."`
	AgentID      *string            `json:"agent_id,omitempty" doc:"One agent's history: changes to the agent itself and to the skills and knowledge under it."`
	Source       *AuditSource       `json:"source,omitempty"`
	Action       *AuditAction       `json:"action,omitempty"`
}

func (*AuditFilter) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Which changes to list. A field not listed here is refused rather than ignored."
	schema.AdditionalProperties = false
	return schema
}

// AuditPage is a page of changes.
type AuditPage struct {
	Items      []AuditEntry `json:"items" nullable:"false"`
	HasMore    bool         `json:"has_more"`
	NextCursor *string      `json:"next_cursor,omitempty" doc:"Pass as cursor for the next page, with the same filter. Absent on the last one."`
}

// AgentChanges is what was changed about an agent since the last sync of its directory.
type AgentChanges struct {
	Items []AuditEntry `json:"items" nullable:"false" doc:"The changes made since the last sync, newest first, at most 200. Empty when the agent has not been touched since."`
	// LastChange is what a sync hands back as base_change to say it has seen these.
	LastChange *string    `json:"last_change,omitempty" doc:"The newest change's id. Send it as base_change on a sync to say these have been seen, and that sync will not be refused for them."`
	SyncedAt   *time.Time `json:"synced_at,omitempty" doc:"When the directory was last synced onto this agent. Absent for an agent no directory has ever been synced onto."`
}

func (*AgentChanges) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What was changed about an agent since its directory was last synced: " +
		"the edits a sync of that directory would write over."
	return schema
}

type queryAuditRequest struct {
	Body *AuditQuery
}

type queryAuditResponse struct {
	Body AuditPage
}

type agentChangesRequest struct {
	Id string `path:"id" doc:"The agent config, as returned when it was created."`
}

type agentChangesResponse struct {
	Body AgentChanges
}

// registerAudit declares the two ways the log is read: the whole of it, filtered, and one
// agent's unsynced changes, which is the question a sync asks.
//
// Both are server-side only. The log says who in the customer's organization changed what,
// which is the app's business and never an end user's.
func (s *Server) registerAudit(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "queryAudit",
		Method:      http.MethodPost,
		Path:        "/v1/audit/query",
		Summary:     "List the changes made to the app's configuration",
		Description: "Every change somebody made to the app's configuration, newest first: the " +
			"agents, their skills, the knowledge they read, the routers, the plugin logins and " +
			"the policies. Each entry names what changed, who changed it, which client they " +
			"used, and the before and after of every field that moved.\n\n" +
			"What an agent does while it runs is not here: a session, a call and a simulation " +
			"run are traffic rather than configuration, and are read from their own endpoints. " +
			"`resource_type`, `resource_id`, `agent_id`, `source` and `action` narrow the list; " +
			"a deleted resource's entries stay, and its deletion is one of them.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "A page of changes"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.queryAudit)
	huma.Register(api, huma.Operation{
		OperationID: "getAgentChanges",
		Method:      http.MethodGet,
		Path:        "/v1/agents/configs/{id}/changes",
		Summary:     "What was changed about an agent since its directory was last synced",
		Description: "The edits a sync of the agent's directory would write over: everything " +
			"changed about the agent, its skills and its knowledge since the last sync, newest " +
			"first. An agent nobody has touched since answers with an empty list.\n\n" +
			"This is what a sync refused with `unsynced_changes` is asking about. Show the " +
			"changes, let the person decide, and sync again with `base_change` set to " +
			"`last_change` to say they have been seen -- having either written them into the " +
			"directory first, or chosen to write over them.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The changes since the last sync"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getAgentChanges)
}

// queryAudit lists the customer's changes, a page at a time.
func (s *Server) queryAudit(ctx context.Context, request *queryAuditRequest) (*queryAuditResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoAudit
	}

	var sent AuditQuery
	if request.Body != nil {
		sent = *request.Body
	}
	cursor, err := decodeCursor[store.AuditPosition](&sent.Cursor)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	filter := store.AuditEntryFilter{Limit: sent.Limit, After: cursor}
	if sent.Filter != nil {
		filter.ResourceType = value((*string)(sent.Filter.ResourceType))
		filter.ResourceID = value(sent.Filter.ResourceID)
		filter.AgentID = value(sent.Filter.AgentID)
		filter.Source = value((*string)(sent.Filter.Source))
		if action := value((*string)(sent.Filter.Action)); action != "" {
			filter.Actions = []string{action}
		}
	}

	found, err := s.store.AuditEntries(ctx, customerID, filter)
	if err != nil {
		return nil, err
	}
	kept, more := page(found, store.AuditEntryLimit(sent.Limit))
	listed := AuditPage{Items: auditEntriesOf(kept), HasMore: more}
	if more {
		last := kept[len(kept)-1]
		listed.NextCursor = encodeCursor(store.AuditPosition{CreatedAt: last.CreatedAt, ID: last.ID})
	}
	return &queryAuditResponse{Body: listed}, nil
}

// getAgentChanges lists what was changed about an agent since its directory was last
// synced, which is what a sync refused for them is asking about.
func (s *Server) getAgentChanges(ctx context.Context, request *agentChangesRequest) (*agentChangesResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoAudit
	}
	config, err := s.configs.AgentConfig(ctx, customerID, request.Id)
	if err != nil {
		return nil, errUnknownConfig
	}

	changes, syncedAt, err := s.unsyncedChanges(ctx, customerID, config.ID, "")
	if err != nil {
		return nil, err
	}
	answer := AgentChanges{Items: auditEntriesOf(changes)}
	if !syncedAt.IsZero() {
		answer.SyncedAt = &syncedAt
	}
	if len(changes) > 0 {
		answer.LastChange = &changes[0].ID
	}
	return &agentChangesResponse{Body: answer}, nil
}

// unsyncedChanges reads what was changed about an agent that the last sync of its directory
// did not make, newest first, along with when that sync was.
//
// base is a change the caller says it has already seen: everything up to and including it
// is left out, which is how a caller that was shown the changes and chose what to do about
// them gets its next sync through. An id the customer does not hold is ignored rather than
// refused, because the only thing it can mean is a client holding an entry from somewhere
// else, and that is not a reason to refuse a sync.
func (s *Server) unsyncedChanges(ctx context.Context, customerID, configID, base string) ([]store.AuditEntry, time.Time, error) {
	since := time.Time{}
	synced, found, err := s.store.LatestAuditEntry(ctx, customerID, store.AuditEntryFilter{
		AgentID: configID, Actions: []string{store.AuditSynced},
	})
	if err != nil {
		return nil, time.Time{}, err
	}
	if found {
		since = synced.CreatedAt
	}
	syncedAt := since

	if base != "" {
		acknowledged, found, err := s.store.AuditEntryByID(ctx, customerID, base)
		if err != nil {
			return nil, time.Time{}, err
		}
		if found && acknowledged.CreatedAt.After(since) {
			since = acknowledged.CreatedAt
		}
	}

	changes, err := s.store.AuditEntries(ctx, customerID, store.AuditEntryFilter{
		AgentID: configID,
		Actions: []string{store.AuditCreated, store.AuditUpdated, store.AuditDeleted},
		Since:   since,
		Limit:   maxAuditEntryPage,
	})
	if err != nil {
		return nil, time.Time{}, err
	}
	kept, _ := page(changes, maxAuditEntryPage)
	return kept, syncedAt, nil
}

// maxAuditEntryPage is store.AuditEntryLimit's ceiling, which the changes list asks for
// outright: a person deciding what to do about their edits wants all of them, and an agent
// edited 200 times since its last sync has a bigger problem than a truncated list.
const maxAuditEntryPage = 200

func auditEntriesOf(rows []store.AuditEntry) []AuditEntry {
	listed := make([]AuditEntry, 0, len(rows))
	for _, row := range rows {
		entry := AuditEntry{
			ID: row.ID, ResourceType: AuditResourceType(row.ResourceType), ResourceID: row.ResourceID,
			ResourceName: row.ResourceName, AgentID: row.AgentID, Action: AuditAction(row.Action),
			Source: AuditSource(row.Source), ActorID: row.ActorID, ActorName: row.ActorName,
			RequestID: row.RequestID, Changes: make([]AuditChange, 0, len(row.Changes)),
			CreatedAt: row.CreatedAt,
		}
		for _, change := range row.Changes {
			entry.Changes = append(entry.Changes, AuditChange{
				Field: change.Field, Before: change.Before, After: change.After,
			})
		}
		listed = append(listed, entry)
	}
	return listed
}

// auditRecord is one change, as the handler that made it describes it.
type auditRecord struct {
	ResourceType string
	ResourceID   string
	ResourceName string
	// AgentID is the agent the change was to or under. A change to an agent config leaves
	// it empty and it is filled in from ResourceID, so the agent's own history and its
	// skills' are one index seek.
	AgentID string
	Action  string
	Changes []store.AuditChange
}

// audit writes down a change somebody made, with who they were and which client they used.
//
// A failure is logged rather than returned, the way recordUser's is: the change the caller
// asked for has already been made, and refusing to report a write that happened would be
// the wrong trade. A record with no changes writes nothing: a save that moved nothing is
// not a change, and a log full of them is a log nobody reads.
func (s *Server) audit(ctx context.Context, record auditRecord) {
	if s.store == nil || (len(record.Changes) == 0 && record.Action != store.AuditSynced) {
		return
	}
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return
	}
	agentID := record.AgentID
	if agentID == "" && record.ResourceType == store.AuditAgentConfig {
		agentID = record.ResourceID
	}
	actor := ActorFrom(ctx)
	entry := store.AuditEntry{
		CustomerID: customerID, ResourceType: record.ResourceType, ResourceID: record.ResourceID,
		ResourceName: record.ResourceName, AgentID: agentID, Action: record.Action,
		Source: actor.Source, ActorID: actor.ID, ActorName: actor.Name,
		RequestID: RequestIDFrom(ctx), Changes: record.Changes,
	}
	if err := s.store.RecordAuditEntry(ctx, &entry); err != nil {
		s.logger.Error("could not record a configuration change",
			"customer", customerID, "resource", record.ResourceType, "id", record.ResourceID,
			"error", err)
	}
}

// auditBookkeeping is what no diff reports: the fields the router writes itself. Nobody
// changed them, and a log saying updated_at moved on every save is a log with the change
// buried in it.
var auditBookkeeping = []string{"id", "created_at", "updated_at", "sync_hash"}

// auditDiff is the fields that moved between two renderings of a resource.
//
// It compares the API's own shape of the thing rather than the stored row, so what a change
// names is what a client reads the resource by: `thinking_llm`, not `subagent`. A field
// absent from one side and present on the other has moved, which is what makes a create
// every field at once and a delete every field back to nothing.
func auditDiff(before, after any) []store.AuditChange {
	was := auditFields(before)
	now := auditFields(after)

	fields := make([]string, 0, len(was)+len(now))
	for field := range was {
		fields = append(fields, field)
	}
	for field := range now {
		if _, named := was[field]; !named {
			fields = append(fields, field)
		}
	}
	slices.Sort(fields)

	changes := make([]store.AuditChange, 0, len(fields))
	for _, field := range fields {
		if slices.Contains(auditBookkeeping, field) {
			continue
		}
		left, had := was[field]
		right, has := now[field]
		if had == has && string(left) == string(right) {
			continue
		}
		change := store.AuditChange{Field: field}
		if had {
			change.Before = auditValue(left)
		}
		if has {
			change.After = auditValue(right)
		}
		changes = append(changes, change)
	}
	return changes
}

// auditFields renders a resource as its JSON object, which is the one description of it
// both halves of a diff can be read from. A value that will not marshal is no fields,
// which makes the diff empty rather than wrong.
func auditFields(resource any) map[string]json.RawMessage {
	if resource == nil {
		return nil
	}
	raw, err := json.Marshal(resource)
	if err != nil {
		return nil
	}
	fields := map[string]json.RawMessage{}
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil
	}
	return fields
}

// auditValue reads a field back into the value a client is handed, so a change carries the
// field in the shape the resource itself has it rather than as a quoted blob of JSON.
func auditValue(raw json.RawMessage) any {
	var held any
	if err := json.Unmarshal(raw, &held); err != nil {
		return string(raw)
	}
	return held
}

// auditFieldsOf is which fields a diff touched, for a caller deciding whether two changes
// are about the same thing.
func auditFieldsOf(changes []store.AuditChange) []string {
	fields := make([]string, 0, len(changes))
	for _, change := range changes {
		fields = append(fields, change.Field)
	}
	return fields
}
