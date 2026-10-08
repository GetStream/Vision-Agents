package api

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/danielgtaylor/huma/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// errTokenExportOff is the answer to an export on a deployment with connectors off, which
// cmd/router builds with no resolver and no scheme (newConnectorResolver, newConnectorRegistry).
var errTokenExportOff = notConfigured("connection tokens cannot be exported: connectors are not enabled on this deployment")

// errNotCustomersApp is the answer for a grant issued to an app that is not the customer's
// own: Stream's (operator), or one the router created for the customer (managed). Only the
// customer's own provider app exports (T45 in subtasks.md on connectors/planning, «A
// Stream-owned app refuses export»).
var errNotCustomersApp = forbidden("this connection's grant was not issued to the app's own OAuth client, so its token is not exported: " +
	"put the app's own client for the connector and connect again")

// ConnectionToken is a connection's access credential, as the app's backend sends it to the
// provider itself.
type ConnectionToken struct {
	ConnectionID string     `json:"connection_id"`
	Header       string     `json:"header" doc:"The HTTP field to send it in: Authorization for an OAuth access token, the connection's own header for an API key."`
	Value        string     `json:"value" doc:"The whole field value: Bearer and the access token for an OAuth access token (RFC 6750 section 2.1), the key for an API key."`
	ExpiresAt    *time.Time `json:"expires_at,omitempty" doc:"When it stops working. Absent when the provider gave no expiry. Export again for a fresh one."`
}

func (*ConnectionToken) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A connection's access credential, for the app's backend to call the " +
		"provider with directly. It holds no refresh token."
	return schema
}

type connectionTokenResponse struct {
	// RFC 6749 section 5.1: a response holding a token «MUST include the HTTP "Cache-Control"
	// response header field with a value of "no-store"».
	CacheControl string `header:"Cache-Control" doc:"Always no-store: keep the token out of every cache."`
	Body         ConnectionToken
}

// exportConnectionToken hands the app's backend a connection's access credential, through the
// resolver so it is fresh, when the credential is the customer's own: an API key, or an OAuth
// grant issued to the customer's own client. Each export writes one token_export audit row
// before the credential leaves.
func (s *Server) exportConnectionToken(ctx context.Context, request *connectionRequest) (*connectionTokenResponse, error) {
	connection, err := s.reachableConnection(ctx, request.ID)
	if err != nil {
		return nil, err
	}
	if s.connectorResolver == nil || len(s.connectors.Schemes) == 0 {
		return nil, errTokenExportOff
	}
	exporter, ok := s.connectors.Schemes[connection.AuthScheme].(core.Exporter)
	if !ok {
		return nil, forbidden(fmt.Sprintf("a %s connection's credential is not exported", connection.AuthScheme))
	}
	ref := core.ConnectionRef{CustomerID: connection.CustomerID, ConnectionID: connection.ID}
	credential, err := s.connectorResolver.Resolve(ctx, ref, core.CredentialRequest{})
	switch {
	case errors.Is(err, store.ErrNoConnectorConnection):
		return nil, errNoSuchConnection
	case errors.Is(err, resolver.ErrNotConnected):
		return nil, conflict("the connection has no credential to export: it is pending or needs a reconnect")
	case errors.Is(err, resolver.ErrTemporarilyUnavailable):
		return nil, unavailable("the provider could not renew the credential just now: try again")
	case err != nil:
		return nil, err
	}
	exported, err := exporter.Export(credential)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	if exported.Client != "" {
		if err := s.customersOwnClient(ctx, connection, exported.Client); err != nil {
			return nil, err
		}
	}
	// No export leaves without its row.
	err = s.store.RecordConnectorAudit(ctx, &store.ConnectorAuditEvent{
		CustomerID: connection.CustomerID, ConnectionID: connection.ID, ConnectorID: connection.ConnectorID,
		OwnerType: connection.OwnerType, Action: store.AuditTokenExport, Revision: credential.Revision,
		RequestID: core.CorrelationOf(ctx).RequestID,
	})
	if err != nil {
		return nil, err
	}
	token := ConnectionToken{ConnectionID: connection.ID, Header: exported.Header, Value: exported.Value}
	if !exported.ExpiresAt.IsZero() {
		token.ExpiresAt = &exported.ExpiresAt
	}
	return &connectionTokenResponse{CacheControl: "no-store", Body: token}, nil
}

// customersOwnClient refuses a grant unless it was issued to the customer's own client and
// the customer's record for the connector is still that kind of client. A connection names no
// client record (store.ConnectorConnection), so the record is looked up by connector: none at
// all means the operator's client in the environment.
func (s *Server) customersOwnClient(ctx context.Context, connection store.ConnectorConnection, issuedTo core.ClientRegistrationMethod) error {
	if issuedTo != core.ClientCustomer {
		return errNotCustomersApp
	}
	record, err := s.store.ConnectorOAuthClient(ctx, connection.CustomerID, connection.ConnectorID)
	if errors.Is(err, store.ErrNoConnectorOAuthClient) {
		return errNotCustomersApp
	}
	if err != nil {
		return err
	}
	if record.Registration != core.ClientCustomer {
		return errNotCustomersApp
	}
	return nil
}
