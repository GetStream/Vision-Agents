//go:build integration

package api

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/apikey"
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

// TestANewTokenHidesTheLastValidationOfTheOldOne (R1.1 of PR #874): what a validate found of
// a token says nothing of the token saved after it, so GET and list show no last validation
// until the next validate. The connection stays connected throughout (a 503), so only the
// revision tells the two tokens apart.
func (s *ConnectionToolsSuite) TestANewTokenHidesTheLastValidationOfTheOldOne() {
	connector := s.connector(bearer.Name, "")
	id := s.connectionTo(connector)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(1, s.token), nil))
	s.answerMCP(http.StatusServiceUnavailable)
	s.validate(id)
	s.Require().NotNil(s.get(id).LastValidation)

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(s.get(id).Revision, s.token+"-new"), nil))

	got := s.get(id)
	s.Equal(3, got.Revision, "new credentials")
	s.Nil(got.LastValidation)
	var page ConnectionPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections?owner_type=app&connector_id="+connector, nil, &page))
	s.Require().Len(page.Items, 1)
	s.Nil(page.Items[0].LastValidation)
}

// TestSavingARefusedTokenAgainHidesItsLastValidation (R1.1 of PR #874, the reviewer's probe):
// the same token saved again connects the connection again at the same revision, and GET and
// list no longer show needs_reauthorization beside status connected.
func (s *ConnectionToolsSuite) TestSavingARefusedTokenAgainHidesItsLastValidation() {
	connector := s.connector(bearer.Name, "")
	id := s.connectionTo(connector)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(1, s.token), nil))
	s.answerMCP(http.StatusBadRequest)
	s.Require().Equal(validationNeedsReauthorization, string(s.validate(id).Status))
	s.provider.AnswerMCP(0)

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(s.get(id).Revision, s.token), nil))

	got := s.get(id)
	s.Equal(ConnectionStatus(store.ConnectionConnected), got.Status)
	s.Equal(2, got.Revision, "the same credentials")
	s.Nil(got.LastValidation)
	var page ConnectionPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections?owner_type=app&connector_id="+connector, nil, &page))
	s.Require().Len(page.Items, 1)
	s.Nil(page.Items[0].LastValidation)
}

// TestATokenSavedDuringAValidateIsNotMovedNorShownTheOldResult (R1.1, R1.3 of PR #874): the
// provider refuses the old token with 400 only after a new one was saved. The new token stays
// connected (Invalidate's revision guard, given the credential the validate sent), and the
// late result, of the old token, is not shown beside it.
func (s *ConnectionToolsSuite) TestATokenSavedDuringAValidateIsNotMovedNorShownTheOldResult() {
	arrived, release := make(chan struct{}, 1), make(chan struct{})
	blocking := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		select {
		case arrived <- struct{}{}:
		default:
		}
		<-release
		http.Error(w, "Bad Request", http.StatusBadRequest)
	}))
	s.T().Cleanup(blocking.Close)
	id := s.connectionTo(s.connectorAt(blocking.URL + "/mcp"))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(1, s.token), nil))
	done := make(chan int, 1)
	go func() {
		done <- s.serverClient.do(http.MethodPost, "/v1/agents/connections/"+id+"/validate", nil, nil)
	}()
	<-arrived

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(s.get(id).Revision, s.token+"-new"), nil))
	close(release)
	s.Require().Equal(http.StatusOK, <-done)

	got := s.get(id)
	s.Equal(ConnectionStatus(store.ConnectionConnected), got.Status, "the new token is not moved by the old one's 400")
	s.Nil(got.LastValidation, "the old token's result is not the new one's")
}

// TestAValidateThatRenewsTheGrantShowsWhatItFound (R1.1 of PR #874): an expired access token
// is renewed by the validate's own Resolve, which moves the revision. What the validate found
// is of the renewed token, so it is shown, not hidden as a result of the token before.
func (s *ConnectionToolsSuite) TestAValidateThatRenewsTheGrantShowsWhatItFound() {
	access, refresh := s.codeGrant()
	id := s.connection(oauth2code.Name)
	grant := s.importedGrant(access, "chat:write")
	grant["values"].(map[string]string)[oauth2code.SuppliedRefreshToken] = refresh
	grant["values"].(map[string]string)[oauth2code.SuppliedExpiresAt] = time.Now().Add(-time.Minute).UTC().Format(time.RFC3339)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", grant, nil))
	refreshes := s.provider.Refreshes()

	s.Require().Equal(validationConnected, string(s.validate(id).Status))

	s.Require().Equal(refreshes+1, s.provider.Refreshes(), "the validate renewed the grant")
	got := s.get(id)
	s.Equal(3, got.Revision, "the renewed credentials")
	s.Require().NotNil(got.LastValidation)
	s.Equal(validationConnected, string(got.LastValidation.Status))
}

// TestAnErrorThatEchoesTheCredentialIsStoredWithoutItAndCut (R1.2 of PR #874): go-sdk puts a
// non-transient error answer's body in the error, and a provider can echo the request's
// Authorization header in it, padded to megabytes. The last validation keeps neither the token
// nor more than maxStoredErrorBytes, in the row, on GET and on list, and cuts no character in
// two: the "x" prefixes move the cut across each byte of the 3-byte "€".
func (s *ConnectionToolsSuite) TestAnErrorThatEchoesTheCredentialIsStoredWithoutItAndCut() {
	for _, scheme := range []string{bearer.Name, oauth2code.Name} {
		for offset := range 3 {
			s.Run(scheme+"/"+strconv.Itoa(offset), func() {
				// 501: a non-transient answer (go-sdk keeps the body) that moves no credential, so
				// both schemes stay connected and the validate fails.
				echo := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					_, _ = io.ReadAll(r.Body)
					authorization := r.Header.Get("Authorization")
					_, token, _ := strings.Cut(authorization, " ")
					message := "bad credentials: " + authorization + " token=" + token + " " + strings.Repeat("x", offset) + strings.Repeat("€", 1<<20)
					body, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": 1, "error": map[string]any{"code": -32600, "message": message}})
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusNotImplemented)
					_, _ = w.Write(body)
				}))
				s.T().Cleanup(echo.Close)
				connector := s.connectorWith(scheme, echo.URL+"/mcp")
				id := s.connectionTo(connector)
				credentials := s.bearerToken(1, s.token)
				if scheme == oauth2code.Name {
					credentials = s.importedGrant(s.token, "chat:write")
				}
				s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", credentials, nil))

				s.Require().Equal(validationFailed, string(s.validate(id).Status))

				var row string
				s.Require().NoError(s.store.DB().NewRaw("SELECT error FROM connector_connection_validations WHERE connection_id = ?", id).Scan(s.T().Context(), &row))
				one := s.get(id).LastValidation
				s.Require().NotNil(one)
				var page ConnectionPage
				s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections?owner_type=app&connector_id="+connector, nil, &page))
				s.Require().Len(page.Items, 1)
				s.Require().NotNil(page.Items[0].LastValidation)
				for where, text := range map[string]string{"row": row, "get": one.Error, "list": page.Items[0].LastValidation.Error} {
					// s.False on strings.Contains, not s.NotContains, so a failure never prints the token.
					s.False(strings.Contains(text, s.token), where+" holds the token")
					s.True(strings.Contains(text, "bad credentials: "+storedRedacted+" token="+storedRedacted+" "+strings.Repeat("x", offset)+"€"),
						where+" keeps the start of the provider's error")
					s.LessOrEqual(len(text), maxStoredErrorBytes, where)
					s.True(utf8.ValidString(text), where+" cuts no character")
					s.True(strings.HasSuffix(text, storedErrorCut), where)
				}
			})
		}
	}
}

// TestATokenSavedDuringAValidateIsNotStoredFromTheProvidersError (R2.1 of PR #874): the
// transport resolves its own credential on every request, so a token saved while the validate
// runs is what the provider sees next, and a provider can echo it in its error. That token is
// not the one the validate cuts out, so a validate whose connection's revision moved keeps no
// provider text: B is nowhere in its row, and GET and list still show no last validation.
func (s *ConnectionToolsSuite) TestATokenSavedDuringAValidateIsNotStoredFromTheProvidersError() {
	tokenA, tokenB := s.token, s.token+"-saved-during-the-validate"
	arrived, release := make(chan struct{}, 1), make(chan struct{})
	echo := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var request struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
			Params struct {
				ProtocolVersion string `json:"protocolVersion"`
			} `json:"params"`
		}
		_ = json.Unmarshal(raw, &request)
		w.Header().Set("Content-Type", "application/json")
		if request.Method == "initialize" {
			select {
			case arrived <- struct{}{}:
			default:
			}
			<-release
			_, _ = fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":%q,"capabilities":{"tools":{}},"serverInfo":{"name":"echo","version":"1"}}}`,
				request.ID, request.Params.ProtocolVersion)
			return
		}
		// 501: a non-transient answer whose body go-sdk keeps, and that moves no credential.
		body, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": 1, "error": map[string]any{"code": -32600, "message": "refused: " + r.Header.Get("Authorization")}})
		w.WriteHeader(http.StatusNotImplemented)
		_, _ = w.Write(body)
	}))
	s.T().Cleanup(echo.Close)
	connector := s.connectorAt(echo.URL + "/mcp")
	id := s.connectionTo(connector)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(1, tokenA), nil))
	done := make(chan int, 1)
	go func() {
		done <- s.serverClient.do(http.MethodPost, "/v1/agents/connections/"+id+"/validate", nil, nil)
	}()
	<-arrived
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(s.get(id).Revision, tokenB), nil))
	close(release)
	s.Require().Equal(http.StatusOK, <-done)

	var row, errorText string
	var revision int
	s.Require().NoError(s.store.DB().NewRaw("SELECT row_to_json(ccv)::text, ccv.error, ccv.revision FROM connector_connection_validations AS ccv WHERE ccv.connection_id = ?", id).
		Scan(s.T().Context(), &row, &errorText, &revision))
	// s.False on strings.Contains, not s.NotContains, so a failure never prints the token.
	s.False(strings.Contains(row, tokenB), "the row holds the token saved during the validate")
	s.Empty(errorText, "a validate whose revision moved keeps no provider text")
	s.Equal(2, revision, "the revision the validate checked")
	got := s.get(id)
	s.Equal(ConnectionStatus(store.ConnectionConnected), got.Status)
	s.Equal(3, got.Revision, "token B")
	s.Nil(got.LastValidation, "token A's result is not token B's")
	var page ConnectionPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections?owner_type=app&connector_id="+connector, nil, &page))
	s.Require().Len(page.Items, 1)
	s.Nil(page.Items[0].LastValidation)
}

// TestAnAPIKeyEchoedFromItsNamedHeaderIsNotStored (R2.2 of PR #874): an api_key goes in the
// header the connection names, not in Authorization, and a provider can echo that header in its
// error. The last validation keeps the provider's text without the key, in the row, on GET and
// on list.
func (s *ConnectionToolsSuite) TestAnAPIKeyEchoedFromItsNamedHeaderIsNotStored() {
	const header = "X-Api-Key"
	echo := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		// 501: a non-transient answer whose body go-sdk keeps, and that moves no credential.
		body, _ := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": 1, "error": map[string]any{"code": -32600, "message": "bad key: " + r.Header.Get(header)}})
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotImplemented)
		_, _ = w.Write(body)
	}))
	s.T().Cleanup(echo.Close)
	connector := s.connectorWith(apikey.Name, echo.URL+"/mcp")
	id := s.connectionTo(connector)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials",
		map[string]any{"expected_revision": 1, "values": map[string]string{apikey.SuppliedKey: s.token, apikey.SuppliedHeader: header}}, nil))

	s.Require().Equal(validationFailed, string(s.validate(id).Status))

	var row string
	s.Require().NoError(s.store.DB().NewRaw("SELECT error FROM connector_connection_validations WHERE connection_id = ?", id).Scan(s.T().Context(), &row))
	one := s.get(id).LastValidation
	s.Require().NotNil(one)
	var page ConnectionPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections?owner_type=app&connector_id="+connector, nil, &page))
	s.Require().Len(page.Items, 1)
	s.Require().NotNil(page.Items[0].LastValidation)
	for where, text := range map[string]string{"row": row, "get": one.Error, "list": page.Items[0].LastValidation.Error} {
		// s.False on strings.Contains, not s.NotContains, so a failure never prints the key.
		s.False(strings.Contains(text, s.token), where+" holds the key")
		s.True(strings.Contains(text, "bad key: "+storedRedacted), where+" keeps the provider's error")
	}
}

// codeGrant is an access and a refresh token the fake issued to its preregistered client
// through an authorization code grant with PKCE (RFC 6749 section 4.1, RFC 7636), so the
// router's oauth2_code can renew the access token.
func (s *ConnectionToolsSuite) codeGrant() (string, string) {
	verifier := "verifier-" + strings.Repeat("x", 43)
	digest := sha256.Sum256([]byte(verifier))
	callback, err := s.provider.Consent(s.provider.URL + fakeprovider.PathAuthorize + "?" + url.Values{
		"response_type": {"code"}, "client_id": {s.provider.ClientID}, "redirect_uri": {fakeprovider.RedirectURI},
		"code_challenge": {base64.RawURLEncoding.EncodeToString(digest[:])}, "code_challenge_method": {"S256"},
		"state": {"state"}, "resource": {s.provider.URL + fakeprovider.PathMCP},
	}.Encode())
	s.Require().NoError(err)
	form := url.Values{"grant_type": {"authorization_code"}, "code": {callback.Query().Get("code")},
		"redirect_uri": {fakeprovider.RedirectURI}, "code_verifier": {verifier}, "resource": {s.provider.URL + fakeprovider.PathMCP}}
	request, err := http.NewRequest(http.MethodPost, s.provider.URL+fakeprovider.PathToken, strings.NewReader(form.Encode()))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	request.SetBasicAuth(s.provider.ClientID, s.provider.ClientSecret)
	response, err := s.provider.Client().Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	var tokens struct {
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
	}
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&tokens))
	s.Require().NotEmpty(tokens.RefreshToken)
	return tokens.AccessToken, tokens.RefreshToken
}

// connectorAt stores a bearer connector of the suite's app whose MCP endpoint is mcp.
func (s *ConnectionToolsSuite) connectorAt(mcp string) string {
	return s.connectorWith(bearer.Name, mcp)
}

// connectorWith stores a connector of the suite's app taking scheme whose MCP endpoint is mcp.
func (s *ConnectionToolsSuite) connectorWith(scheme, mcp string) string {
	connector := "custom_tools" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	yaml := strings.Replace(s.manifest(connector, 1, scheme, ""),
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
