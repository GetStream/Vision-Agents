package api

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net/http"
	"slices"
	"strings"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// maxApprovedTools is how many tools one approve names: the maxItems of a binding's tools
// (AgentConnectorBinding.Tools), so the tools one binding offers fit in one call.
const maxApprovedTools = 128

// errToolPinMoved is the answer to an approve that lost a race (store.ErrConnectorToolPinMoved).
// 409, as errStaleRevision: the caller reads again and retries (RFC 9110 section 15.5.10).
var errToolPinMoved = conflict("a tool's pin or the connection's grant changed while the tools were approved, " +
	"so nothing was approved: read GET .../tools again and retry")

// ConnectionToolApproval is one tool an approve names, at the digest its caller read.
type ConnectionToolApproval struct {
	Name         string `json:"name" minLength:"1" doc:"The tool's name at the provider, as GET .../tools lists it."`
	SchemaDigest string `json:"schema_digest" minLength:"1" doc:"Its schema_digest as GET .../tools lists it now: what the caller approves."`
}

// ConnectionToolsApproval is the body of an approve.
type ConnectionToolsApproval struct {
	Tools []ConnectionToolApproval `json:"tools" minItems:"1" maxItems:"128" doc:"The tools to approve, each named once. Up to 128."`
}

func (*ConnectionToolsApproval) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The tools of a connection to approve at the schema the provider lists now. An " +
		"unknown field is refused rather than ignored."
	schema.AdditionalProperties = false
	return schema
}

type approveConnectionToolsRequest struct {
	ID   string `path:"id" doc:"The connection, as returned when it was created."`
	Body ConnectionToolsApproval
}

// ApprovedConnectionTool is one tool an approve pinned (store.ApprovedTool).
type ApprovedConnectionTool struct {
	Name                 string `json:"name"`
	SchemaDigest         string `json:"schema_digest" doc:"The digest the tool is pinned at now."`
	PreviousSchemaDigest string `json:"previous_schema_digest,omitempty" doc:"The digest it was pinned at before. Absent when it was not pinned for the connection's grant."`
}

// ApprovedConnectionTools is what an approve pinned.
type ApprovedConnectionTools struct {
	ConnectionID string                   `json:"connection_id"`
	Tools        []ApprovedConnectionTool `json:"tools" doc:"The tools approved, by name, sorted."`
}

type approvedConnectionToolsResponse struct {
	Body ApprovedConnectionTools
}

// registerToolApprovals declares the approve, server-side only like every connection
// operation.
func (s *Server) registerToolApprovals(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "approveConnectionTools",
		Method:      http.MethodPost,
		Path:        "/v1/agents/connections/{id}/tools/approve",
		Summary:     "Approve a connection's tools again",
		Description: "Pins tools of the connection at the schema the provider lists now, for the " +
			"connection's current grant. A session pins a tool granted by name or by pattern the " +
			"first time it offers it, and leaves it out while the provider lists it at another " +
			"schema, which validate reports as needs_reapproval with the changed_tools. This is " +
			"how a bearer or api_key connection gets them back; an OAuth one may reconnect " +
			"instead, which begins a new grant.\n\n" +
			"Each tool named must be one GET .../tools lists, which a validate refreshes, at the " +
			"schema_digest sent: a tool it does not list is a 400, and one it lists at another " +
			"digest a 409. The pins move together, and only while each is still what it was when " +
			"read and the grant has not changed since; otherwise a 409. A refused approve writes " +
			"nothing. The next session offers the tools, and the next validate answers connected " +
			"when no other tool changed; last_validation keeps what the last validate found until " +
			"then. One audit row, tool_reapproved, names the tools and their digests. A " +
			"schema_digest an agent config grants is not a pin: change the config instead. Who " +
			"may approve them is who may read the connection.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The tools approved"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound, http.StatusConflict},
	}, s.approveConnectionTools)
}

// approveConnectionTools pins the tools named at the digests the connection's last validate
// listed, when those are the digests sent, all or none (store.ReapproveConnectorTools).
func (s *Server) approveConnectionTools(ctx context.Context, request *approveConnectionToolsRequest) (*approvedConnectionToolsResponse, error) {
	connection, err := s.reachableConnection(ctx, request.ID)
	if err != nil {
		return nil, err
	}
	// What GET .../tools shows: the session compares its pins with what the provider lists
	// when it opens, and a validate is how the caller reads that list.
	listed := make(map[string]string, len(connection.CachedTools))
	for _, tool := range connection.CachedTools {
		listed[tool.Name] = tool.SchemaDigest
	}
	pins, err := s.store.ConnectorToolPins(ctx, connection.ID, connection.ConnectedAt)
	if err != nil {
		return nil, err
	}
	approved := make([]store.ApprovedTool, 0, len(request.Body.Tools))
	named := map[string]bool{}
	var moved []string
	for _, tool := range request.Body.Tools {
		if named[tool.Name] {
			return nil, invalidRequest(fmt.Sprintf("tool %q is named twice", tool.Name))
		}
		named[tool.Name] = true
		digest, found := listed[tool.Name]
		if !found {
			return nil, invalidRequest(fmt.Sprintf("the connection lists no tool named %q: validate it to "+
				"read the tools the provider lists now", tool.Name))
		}
		if digest != tool.SchemaDigest {
			moved = append(moved, tool.Name)
		}
		approved = append(approved, store.ApprovedTool{Name: tool.Name, SchemaDigest: digest, PreviousSchemaDigest: pins[tool.Name]})
	}
	if len(moved) > 0 {
		return nil, conflict(fmt.Sprintf("the connection lists %s at another schema_digest than the one sent, "+
			"so nothing was approved: read GET .../tools again", strings.Join(moved, ", ")))
	}
	err = s.store.ReapproveConnectorTools(ctx, connection, core.CorrelationOf(ctx).RequestID, approved)
	if errors.Is(err, store.ErrNoConnectorConnection) {
		return nil, errNoSuchConnection
	}
	if errors.Is(err, store.ErrConnectorToolPinMoved) {
		return nil, errToolPinMoved
	}
	if err != nil {
		return nil, err
	}
	answer := ApprovedConnectionTools{ConnectionID: connection.ID, Tools: approvedTools(approved)}
	slices.SortFunc(answer.Tools, func(a, b ApprovedConnectionTool) int { return strings.Compare(a.Name, b.Name) })
	s.logger.Info("connector tools approved", "connection", connection.ID, "connector", connection.ConnectorID,
		"tools", strings.Join(slices.Sorted(maps.Keys(named)), ","))
	return &approvedConnectionToolsResponse{Body: answer}, nil
}

// approvedTools is tools as the API shows them, nil for none.
func approvedTools(tools []store.ApprovedTool) []ApprovedConnectionTool {
	if tools == nil {
		return nil
	}
	shown := make([]ApprovedConnectionTool, 0, len(tools))
	for _, tool := range tools {
		shown = append(shown, ApprovedConnectionTool(tool))
	}
	return shown
}
