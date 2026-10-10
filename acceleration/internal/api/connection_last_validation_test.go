//go:build integration

package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"time"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// rejectedStaticError is what a validate says of a bearer or api_key connection the provider
// refused (resolver.rejectedStatic).
const rejectedStaticError = "The provider rejected the stored token or key; replace it with PUT /v1/agents/connections/{id}/credentials"

// TestA4xxOnAStaticTokenMovesItToNeedsReauthorization is AI-1052 (E2E F60): GitHub's MCP server
// answers a well-formed but wrong token with 400, not 401, and the connection stayed connected.
// Any 4xx but 429 now moves a bearer or api_key connection as a 401 does, and the last
// validation says so on GET.
func (s *ConnectionToolsSuite) TestA4xxOnAStaticTokenMovesItToNeedsReauthorization() {
	for _, status := range []int{http.StatusBadRequest, http.StatusForbidden, http.StatusNotFound} {
		s.Run(strconv.Itoa(status), func() {
			id := s.connected(bearer.Name)
			s.answerMCP(status)
			before := time.Now()

			validation := s.validate(id)

			s.Equal(validationNeedsReauthorization, string(validation.Status))
			s.Equal(codeCredentialRejected, validation.Code)
			s.Equal(rejectedStaticError, validation.Error)
			got := s.get(id)
			s.Equal(ConnectionStatus(store.ConnectionNeedsReauthorization), got.Status)
			s.Require().NotNil(got.LastValidation)
			s.Equal(validationNeedsReauthorization, string(got.LastValidation.Status))
			s.Equal(codeCredentialRejected, got.LastValidation.Code)
			s.Equal(rejectedStaticError, got.LastValidation.Error)
			s.WithinDuration(before, got.LastValidation.CheckedAt, time.Minute)
		})
	}
}

// TestAMoveOnValidateIsAuditedAsTheMoveOnA401Is: the move is the resolver's Invalidate, so it
// leaves the grant_revoked row a 401 leaves.
func (s *ConnectionToolsSuite) TestAMoveOnValidateIsAuditedAsTheMoveOnA401Is() {
	id := s.connected(bearer.Name)
	s.answerMCP(http.StatusBadRequest)

	s.validate(id)

	var page ConnectorAuditPage
	s.Require().Eventually(func() bool {
		return s.serverClient.do(http.MethodGet, "/v1/agents/connector-audit?connection_id="+url.QueryEscape(id), nil, &page) == http.StatusOK &&
			len(page.Items) >= 2
	}, settleFor, 20*time.Millisecond)
	s.Equal(ConnectorAuditAction(store.AuditGrantRevoked), page.Items[0].Action)
	s.Equal(string(core.OutcomeInvalidGrant), page.Items[0].Reason)
	s.Equal(ConnectorAuditAction(store.AuditGrantCreated), page.Items[1].Action)
}

// TestANewTokenAndAValidateBringAMovedConnectionBack pins Decision 3 of AI-1052: saving a new
// token connects it again, as before, and the next validate replaces the last validation.
func (s *ConnectionToolsSuite) TestANewTokenAndAValidateBringAMovedConnectionBack() {
	id := s.connected(bearer.Name)
	s.answerMCP(http.StatusBadRequest)
	s.Require().Equal(validationNeedsReauthorization, string(s.validate(id).Status))
	s.provider.AnswerMCP(0)

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(s.get(id).Revision, s.token), nil))
	s.Equal(ConnectionStatus(store.ConnectionConnected), s.get(id).Status)
	validation := s.validate(id)

	s.Equal(validationConnected, string(validation.Status))
	got := s.get(id).LastValidation
	s.Require().NotNil(got)
	s.Equal(validationConnected, string(got.Status))
	s.Empty(got.Code)
	s.Empty(got.Error)
	s.Require().NotNil(validation.CheckedAt)
	s.True(validation.CheckedAt.Equal(got.CheckedAt), "the time the tools were listed")
}

// TestA429OrA5xxKeepsAStaticTokenConnected: a 429 says to wait and a 5xx says nothing about the
// credential, so the connection stays connected; the last validation keeps the failure and the
// provider's status.
func (s *ConnectionToolsSuite) TestA429OrA5xxKeepsAStaticTokenConnected() {
	for _, status := range []int{http.StatusTooManyRequests, http.StatusInternalServerError, http.StatusServiceUnavailable} {
		s.Run(strconv.Itoa(status), func() {
			id := s.connected(bearer.Name)
			s.answerMCP(status)

			validation := s.validate(id)

			s.Equal(validationFailed, string(validation.Status))
			got := s.get(id)
			s.Equal(ConnectionStatus(store.ConnectionConnected), got.Status)
			s.Require().NotNil(got.LastValidation)
			s.Equal(validationFailed, string(got.LastValidation.Status))
			s.Equal(strconv.Itoa(status), got.LastValidation.Code)
			s.Equal(validation.Error, got.LastValidation.Error)
			s.NotEmpty(got.LastValidation.Error)
		})
	}
}

// TestAProviderThatDoesNotAnswerKeepsAStaticTokenConnected: no answer at all (a closed port)
// says nothing about the credential.
func (s *ConnectionToolsSuite) TestAProviderThatDoesNotAnswerKeepsAStaticTokenConnected() {
	closed := httptest.NewTLSServer(http.NotFoundHandler())
	closed.Close()
	id := s.connectionTo(s.connectorAt(closed.URL + fakeprovider.PathMCP))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(1, s.token), nil))

	validation := s.validate(id)

	s.Equal(validationFailed, string(validation.Status))
	got := s.get(id)
	s.Equal(ConnectionStatus(store.ConnectionConnected), got.Status)
	s.Require().NotNil(got.LastValidation)
	s.Equal(validationFailed, string(got.LastValidation.Status))
	s.Empty(got.LastValidation.Code, "no status to name")
	s.Equal(validation.Error, got.LastValidation.Error)
}

// TestTheAnswerAValidateFailedOnDecidesNotTheSessionsDelete: a stateful MCP server answers
// tools/list with 503, then the DELETE that ends the session with 405, as MCP 2025-11-25 lets it
// («Session Management»). The 503 is what the validate failed on, so the token stays connected.
func (s *ConnectionToolsSuite) TestTheAnswerAValidateFailedOnDecidesNotTheSessionsDelete() {
	server := mcpsdk.NewServer(&mcpsdk.Implementation{Name: "stateful", Version: "1"}, nil)
	handler := mcpsdk.NewStreamableHTTPHandler(func(*http.Request) *mcpsdk.Server { return server }, &mcpsdk.StreamableHTTPOptions{JSONResponse: true})
	deletes := 0
	// httptest servers share one certificate, so the fake's client trusts this one too.
	stateful := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewReader(raw))
		switch {
		case r.Method == http.MethodDelete:
			deletes++
			w.WriteHeader(http.StatusMethodNotAllowed)
		case bytes.Contains(raw, []byte(`"method":"tools/list"`)):
			w.WriteHeader(http.StatusServiceUnavailable)
		default:
			handler.ServeHTTP(w, r)
		}
	}))
	s.T().Cleanup(stateful.Close)
	id := s.connectionTo(s.connectorAt(stateful.URL + "/mcp"))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(1, s.token), nil))

	validation := s.validate(id)

	s.Equal(validationFailed, string(validation.Status))
	s.Equal(1, deletes, "the session was ended with a DELETE")
	got := s.get(id)
	s.Equal(ConnectionStatus(store.ConnectionConnected), got.Status)
	s.Require().NotNil(got.LastValidation)
	s.Equal("503", got.LastValidation.Code)
}

// TestA400OnAnOAuthGrantKeepsItConnected is a control: an OAuth grant keeps the 401-only rule
// on validate, as on base (probe on 599298c6: status connected, validate failed).
func (s *ConnectionToolsSuite) TestA400OnAnOAuthGrantKeepsItConnected() {
	id := s.connection(oauth2code.Name)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.importedGrant(s.token, "chat:write"), nil))
	s.answerMCP(http.StatusBadRequest)

	validation := s.validate(id)

	s.Equal(validationFailed, string(validation.Status))
	got := s.get(id)
	s.Equal(ConnectionStatus(store.ConnectionConnected), got.Status)
	s.Require().NotNil(got.LastValidation)
	s.Equal("400", got.LastValidation.Code)
}

// TestAConnectionNeverValidatedShowsNoLastValidation is the control for GET and list: a
// connection no validate has checked reads as on base, with no last_validation key (probe on
// 599298c6: the key is absent).
func (s *ConnectionToolsSuite) TestAConnectionNeverValidatedShowsNoLastValidation() {
	connector := s.connector(bearer.Name, "")
	id := s.connectionTo(connector)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(1, s.token), nil))

	var one map[string]json.RawMessage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections/"+id, nil, &one))
	var page struct {
		Items []map[string]json.RawMessage `json:"items"`
	}
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections?owner_type=app&connector_id="+connector, nil, &page))

	s.NotContains(one, "last_validation")
	s.Require().Len(page.Items, 1)
	s.NotContains(page.Items[0], "last_validation")
}

// TestTheListShowsEachConnectionsLastValidation: the list reads the last validations for its
// page, each on its own connection.
func (s *ConnectionToolsSuite) TestTheListShowsEachConnectionsLastValidation() {
	connector := s.connector(bearer.Name, "")
	failing := s.connectionTo(connector)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+failing+"/credentials", s.bearerToken(1, s.token), nil))
	working := s.connectionTo(connector)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+working+"/credentials", s.bearerToken(1, s.token), nil))
	untouched := s.connectionTo(connector)
	s.validate(working)
	s.answerMCP(http.StatusServiceUnavailable)
	s.validate(failing)

	var page ConnectionPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections?owner_type=app&connector_id="+connector, nil, &page))

	byID := map[string]*ConnectionLastValidation{}
	for _, item := range page.Items {
		byID[item.ID] = item.LastValidation
	}
	s.Require().Len(byID, 3)
	s.Require().NotNil(byID[working])
	s.Equal(validationConnected, string(byID[working].Status))
	s.Require().NotNil(byID[failing])
	s.Equal(validationFailed, string(byID[failing].Status))
	s.Equal("503", byID[failing].Code)
	s.Nil(byID[untouched])
}

// TestAPendingConnectionsValidateIsRecordedWithoutAskingTheProvider: what the validate answers
// is what is kept, a refusal before anything is sent included.
func (s *ConnectionToolsSuite) TestAPendingConnectionsValidateIsRecordedWithoutAskingTheProvider() {
	id := s.connection(bearer.Name)

	validation := s.validate(id)

	got := s.get(id).LastValidation
	s.Require().NotNil(got)
	s.Equal(validationPending, string(got.Status))
	s.Equal(validation.Error, got.Error)
	s.Empty(got.Code)
}

// connectorAt stores a bearer connector of the suite's app whose MCP endpoint is mcp.
func (s *ConnectionToolsSuite) connectorAt(mcp string) string {
	connector := "custom_tools" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	yaml := strings.Replace(s.manifest(connector, 1, bearer.Name, ""),
		"mcp: "+s.provider.URL+fakeprovider.PathMCP, "mcp: "+mcp, 1)
	manifest, err := core.ParseManifest([]byte(yaml))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	return connector
}

// answerMCP has the fake answer every MCP request with status until the test ends.
func (s *ConnectionToolsSuite) answerMCP(status int) {
	s.provider.AnswerMCP(status)
	s.T().Cleanup(func() { s.provider.AnswerMCP(0) })
}
