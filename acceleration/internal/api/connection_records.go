package api

import (
	"context"
	"net/http"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ConnectionInvocation is one connector tool call a session ran through a connection. It
// holds no argument and no result, for any session.
type ConnectionInvocation struct {
	ID           string               `json:"id"`
	ConnectionID string               `json:"connection_id"`
	ConnectorID  string               `json:"connector_id"`
	ConfigID     string               `json:"config_id" doc:"The agent config whose binding the call went through."`
	Binding      string               `json:"binding" doc:"The alias the config binds the connector under."`
	Tool         string               `json:"tool" doc:"The tool's name at the provider, without the alias."`
	SessionID    string               `json:"session_id,omitempty" doc:"The session that called it. Absent for an incognito session, whose calls are tied to no conversation."`
	StartedAt    time.Time            `json:"started_at"`
	LatencyMs    int64                `json:"latency_ms" doc:"From the call reaching the router to its answer, the router's own checks included."`
	ErrorType    *InvocationErrorType `json:"error_type,omitempty" doc:"How the call failed. Absent for a call that answered."`
}

func (*ConnectionInvocation) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One connector tool call a session ran through the connection: the " +
		"binding, the tool, how long it took and how it failed. What the call was asked and " +
		"answered is never kept."
	return schema
}

// InvocationErrorType is how a connector tool call failed.
type InvocationErrorType string

func (InvocationErrorType) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "InvocationErrorType",
		"customer_auth: the provider refused the connection's credential, or it had none; "+
			"reconnect it. external_server: the provider answered with a failure or could not "+
			"be reached. client_timeout: the router stopped waiting before the provider "+
			"answered, and nothing says it got the call. outcome_unknown: the call was sent and "+
			"cut off, by the binding's timeout or an interrupted turn, so it may have been done. "+
			"denied: the router refused it before anything was sent.",
		store.InvocationCustomerAuth, store.InvocationExternalServer, store.InvocationClientTimeout,
		store.InvocationOutcomeUnknown, store.InvocationDenied)
}

// ConnectionInvocationPage is a page of a connection's calls.
type ConnectionInvocationPage struct {
	Items      []ConnectionInvocation `json:"items"`
	HasMore    bool                   `json:"has_more"`
	NextCursor *string                `json:"next_cursor,omitempty" doc:"Pass as cursor for the next page. Absent on the last one."`
}

// ConnectorAuditEvent is one grant a connection got, renewed or lost, as an operator reads it.
type ConnectorAuditEvent struct {
	ID           string               `json:"id"`
	ConnectionID string               `json:"connection_id" doc:"The connection, which may since have been deleted."`
	ConnectorID  string               `json:"connector_id"`
	OwnerType    ConnectionOwnerType  `json:"owner_type"`
	Action       ConnectorAuditAction `json:"action"`
	Reason       string               `json:"reason,omitempty" doc:"Why: consent or credentials for a created grant; deleted or user_deleted for a delete; for a grant the provider ended, its word for why, such as invalid_grant, scope_required or revoked."`
	Revision     int                  `json:"revision,omitempty" doc:"The connection's credential revision once the change was made. Absent when the change names none, as a delete."`
	RequestID    string               `json:"request_id,omitempty" doc:"The X-Request-Id of the API request that caused it."`
	SessionID    string               `json:"session_id,omitempty" doc:"The session whose tool call caused it. Absent for an incognito session."`
	AttemptID    string               `json:"attempt_id,omitempty" doc:"The authorization attempt a consent finished."`
	CreatedAt    time.Time            `json:"created_at"`
}

func (*ConnectorAuditEvent) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One grant a connection got, renewed or lost, with the ids that tie it " +
		"to what caused it. It names no user and no provider account, so it outlives a user's " +
		"connections being deleted."
	return schema
}

// ConnectorAuditAction is what an audit row records.
type ConnectorAuditAction string

func (ConnectorAuditAction) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ConnectorAuditAction",
		"grant_created: a consent or a credentials write gave the connection a grant. "+
			"grant_refreshed: the router renewed its credential. grant_revoked: the grant ended, "+
			"because the provider refused or revoked it or the connection was deleted.",
		store.AuditGrantCreated, store.AuditGrantRefreshed, store.AuditGrantRevoked)
}

// ConnectorAuditPage is a page of audit rows.
type ConnectorAuditPage struct {
	Items      []ConnectorAuditEvent `json:"items"`
	HasMore    bool                  `json:"has_more"`
	NextCursor *string               `json:"next_cursor,omitempty" doc:"Pass as cursor for the next page, with the same connection_id. Absent on the last one."`
}

type listConnectionInvocationsRequest struct {
	ID string `path:"id" doc:"The connection, as returned when it was created."`
	// 200 and 25 are store.InvocationLimit's.
	Limit  int    `query:"limit" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 25."`
	Cursor string `query:"cursor" doc:"The next_cursor of the previous page. Omitted is the first page."`
}

type listConnectionInvocationsResponse struct {
	Body ConnectionInvocationPage
}

type listConnectorAuditRequest struct {
	ConnectionID string `query:"connection_id" doc:"Keeps one connection's rows, deleted or not."`
	// 200 and 25 are store.AuditLimit's.
	Limit  int    `query:"limit" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 25."`
	Cursor string `query:"cursor" doc:"The next_cursor of the previous page. Omitted is the first page."`
}

type listConnectorAuditResponse struct {
	Body ConnectorAuditPage
}

type deleteUserConnectionsRequest struct {
	UserID string `path:"user_id" minLength:"1" doc:"The user whose connections to delete, as owner.user_id named them."`
}

// registerConnectionRecords declares what the router keeps about connections' use: the
// invocation log (T29), the audit (T47), and the user delete that takes a user's connections
// and their log with it. All three are server-side only: a connection's use is the app's
// business, and a device never reaches a connection (registerConnections).
func (s *Server) registerConnectionRecords(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listConnectionInvocations",
		Method:      http.MethodGet,
		Path:        "/v1/agents/connections/{id}/invocations",
		Summary:     "List a connection's tool calls",
		Description: "Every tool call sessions ran through the connection, newest first: the " +
			"binding, the tool, the latency and how it failed. What a call was asked and " +
			"answered is never kept, and an incognito session's calls name no session. Who may " +
			"read them is who may read the connection.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "A page of tool calls"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.listConnectionInvocations)
	huma.Register(api, huma.Operation{
		OperationID: "listConnectorAudit",
		Method:      http.MethodGet,
		Path:        "/v1/agents/connector-audit",
		Summary:     "List the connector audit",
		Description: "Every grant the app's connections got, renewed or lost, newest first, " +
			"with the request, session and authorization attempt that caused each. A deleted " +
			"connection's rows stay, and its deletion is one of them.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "A page of audit rows"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listConnectorAudit)
	huma.Register(api, huma.Operation{
		OperationID:   "deleteUserConnections",
		Method:        http.MethodDelete,
		Path:          "/v1/agents/users/{user_id}/connections",
		Summary:       "Delete every connection of one user",
		DefaultStatus: http.StatusNoContent,
		Description: "For offboarding and erasure requests: deletes every connection the user " +
			"owns, live or deleted before, for good, with its credentials, its pending consents " +
			"and its tool call log, so the user's id and their provider accounts' ids are gone. " +
			"The next session for the user attaches none of them. The provider is not asked to " +
			"revoke what it issued. The audit keeps one grant_revoked row for each connection " +
			"that still held a grant, naming neither the user nor the account. A user with no " +
			"connections is not an error.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"204": {Description: "The user's connections are deleted"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.deleteUserConnections)
}

// listConnectionInvocations lists one connection's calls, a page at a time.
func (s *Server) listConnectionInvocations(ctx context.Context, request *listConnectionInvocationsRequest) (*listConnectionInvocationsResponse, error) {
	connection, err := s.reachableConnection(ctx, request.ID)
	if err != nil {
		return nil, err
	}
	cursor, err := decodeCursor[store.InvocationPosition](&request.Cursor)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	found, err := s.store.ConnectorInvocations(ctx, connection.CustomerID, connection.ID, request.Limit, cursor)
	if err != nil {
		return nil, err
	}
	kept, more := page(found, store.InvocationLimit(request.Limit))
	listed := ConnectionInvocationPage{Items: make([]ConnectionInvocation, 0, len(kept)), HasMore: more}
	for _, row := range kept {
		item := ConnectionInvocation{
			ID: row.ID, ConnectionID: row.ConnectionID, ConnectorID: row.ConnectorID, ConfigID: row.ConfigID,
			Binding: row.Binding, Tool: row.Tool, SessionID: row.SessionID, StartedAt: row.StartedAt,
			LatencyMs: row.LatencyMs,
		}
		if row.ErrorType != "" {
			failure := InvocationErrorType(row.ErrorType)
			item.ErrorType = &failure
		}
		listed.Items = append(listed.Items, item)
	}
	if more {
		last := kept[len(kept)-1]
		listed.NextCursor = encodeCursor(store.InvocationPosition{StartedAt: last.StartedAt, ID: last.ID})
	}
	return &listConnectionInvocationsResponse{Body: listed}, nil
}

// listConnectorAudit lists the customer's audit rows, a page at a time.
func (s *Server) listConnectorAudit(ctx context.Context, request *listConnectorAuditRequest) (*listConnectorAuditResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConnections
	}
	cursor, err := decodeCursor[store.AuditPosition](&request.Cursor)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	found, err := s.store.ConnectorAuditEvents(ctx, customerID, store.AuditFilter{
		ConnectionID: request.ConnectionID, Limit: request.Limit, After: cursor,
	})
	if err != nil {
		return nil, err
	}
	kept, more := page(found, store.AuditLimit(request.Limit))
	listed := ConnectorAuditPage{Items: make([]ConnectorAuditEvent, 0, len(kept)), HasMore: more}
	for _, row := range kept {
		listed.Items = append(listed.Items, ConnectorAuditEvent{
			ID: row.ID, ConnectionID: row.ConnectionID, ConnectorID: row.ConnectorID,
			OwnerType: ConnectionOwnerType(row.OwnerType), Action: ConnectorAuditAction(row.Action),
			Reason: row.Reason, Revision: row.Revision, RequestID: row.RequestID, SessionID: row.SessionID,
			AttemptID: row.AttemptID, CreatedAt: row.CreatedAt,
		})
	}
	if more {
		last := kept[len(kept)-1]
		listed.NextCursor = encodeCursor(store.AuditPosition{CreatedAt: last.CreatedAt, ID: last.ID})
	}
	return &listConnectorAuditResponse{Body: listed}, nil
}

// deleteUserConnections hard deletes every connection of one user of the caller's app, drops
// their outbound clients and audits each grant it revoked.
func (s *Server) deleteUserConnections(ctx context.Context, request *deleteUserConnectionsRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil {
		return nil, errNoConnections
	}
	deleted, err := s.store.DeleteUserConnectorConnections(ctx, customerID, request.UserID)
	if err != nil {
		return nil, err
	}
	for _, connection := range deleted {
		if s.connectorTransports != nil {
			s.connectorTransports.Close(core.ConnectionRef{CustomerID: customerID, ConnectionID: connection.ID})
		}
		if connection.HadGrant {
			s.auditGrant(ctx, customerID, connection.ID, connection.ConnectorID, store.OwnerUser,
				store.AuditGrantRevoked, store.AuditReasonUserDeleted, 0, "")
		}
	}
	return nil, nil
}
