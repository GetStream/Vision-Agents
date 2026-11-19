package client

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// standInConnectionsPage is the stand-in's page size, below what List asks for, so a list
// longer than it shows the cursor is followed.
const standInConnectionsPage = 2

// serveConnections answers the connection routes with the router's owner rule
// (acceleration/internal/api/connections.go, ownerOf and mayReach): a user's connection is
// created and reached only while acting for that user, named by X-Stream-User-Id; the app's by
// any backend.
func (r *router) serveConnections(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/agents/connections", func(w http.ResponseWriter, request *http.Request) {
		r.record(request)
		r.mu.Lock()
		defer r.mu.Unlock()
		sent := r.bodies[request.Method+" "+request.URL.Path]
		owner, _ := sent["owner"].(map[string]any)
		userID, named := owner["user_id"].(string)
		switch {
		case owner["type"] == "app" && named:
			refuseConnection(w, "an app-owned connection has no owner.user_id")
			return
		case owner["type"] == "user" && (userID == "" || userID != request.Header.Get(stream.UserHeader)):
			refuseConnection(w, "owner.user_id must be the user this backend acts for, named by "+stream.UserHeader)
			return
		case owner["type"] != "app" && owner["type"] != "user":
			refuseConnection(w, "owner.type must be app or user")
			return
		}
		connectorID, _ := sent["connector_id"].(string)
		created := acceleration.Connection{
			Id:          ptr(fmt.Sprintf("connection-%d", len(r.connections)+1)),
			ConnectorId: connectorID,
			Owner:       acceleration.ConnectionOwner{Type: acceleration.ConnectionOwnerType(owner["type"].(string))},
			Status:      acceleration.ConnectionStatusPending,
		}
		if named {
			created.Owner.UserId = &userID
		}
		if label, ok := sent["label"].(string); ok {
			created.Label = &label
		}
		r.connections = append(r.connections, created)
		answer(w, http.StatusCreated, created)
	})

	mux.HandleFunc("GET /v1/agents/connections", func(w http.ResponseWriter, request *http.Request) {
		r.record(request)
		r.mu.Lock()
		defer r.mu.Unlock()
		query := request.URL.Query()
		ownerType := cmp.Or(query.Get("owner_type"), "app")
		acting := request.Header.Get(stream.UserHeader)
		var matched []acceleration.Connection
		for _, connection := range slices.Backward(r.connections) {
			if string(connection.Owner.Type) != ownerType {
				continue
			}
			if ownerType == "user" && acting != "" && *connection.Owner.UserId != acting {
				continue
			}
			if connector := query.Get("connector_id"); connector != "" && connection.ConnectorId != connector {
				continue
			}
			matched = append(matched, connection)
		}
		from, _ := strconv.Atoi(strings.TrimPrefix(query.Get("cursor"), "after-"))
		to := min(from+standInConnectionsPage, len(matched))
		page := acceleration.ConnectionPage{Items: ptr(matched[from:to]), HasMore: to < len(matched)}
		if page.HasMore {
			page.NextCursor = ptr(fmt.Sprintf("after-%d", to))
		}
		answer(w, http.StatusOK, page)
	})

	mux.HandleFunc("GET /v1/agents/connections/{id}", func(w http.ResponseWriter, request *http.Request) {
		r.record(request)
		r.mu.Lock()
		defer r.mu.Unlock()
		at := r.reachable(request)
		if at < 0 {
			connectionNotFound(w)
			return
		}
		answer(w, http.StatusOK, r.connections[at])
	})

	mux.HandleFunc("DELETE /v1/agents/connections/{id}", func(w http.ResponseWriter, request *http.Request) {
		r.record(request)
		r.mu.Lock()
		defer r.mu.Unlock()
		at := r.reachable(request)
		if at < 0 {
			connectionNotFound(w)
			return
		}
		r.connections = slices.Delete(r.connections, at, at+1)
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("POST /v1/agents/connections/{id}/authorizations", func(w http.ResponseWriter, request *http.Request) {
		r.record(request)
		r.mu.Lock()
		defer r.mu.Unlock()
		at := r.reachable(request)
		if at < 0 {
			connectionNotFound(w)
			return
		}
		answer(w, http.StatusCreated, acceleration.Authorization{
			Id:           "attempt-1",
			Kind:         acceleration.Consent,
			LaunchUrl:    "https://router.example.com/v1/agents/connections/" + *r.connections[at].Id + "/launch",
			HandoffToken: "handoff-1",
			ExpiresAt:    pinnedAt.Add(10 * time.Minute),
		})
	})
}

// reachable is the index of the connection a request names, if the request may reach it, and
// -1 otherwise. The caller holds r.mu.
func (r *router) reachable(request *http.Request) int {
	acting := request.Header.Get(stream.UserHeader)
	for at, connection := range r.connections {
		if *connection.Id != request.PathValue("id") {
			continue
		}
		if connection.Owner.Type == acceleration.ConnectionOwnerTypeApp ||
			acting != "" && *connection.Owner.UserId == acting {
			return at
		}
	}
	return -1
}

// actingFor is the X-Stream-User-Id of every request to a path, whatever the query, in order.
func (r *router) actingFor(method, path string) []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var named []string
	for at, line := range r.asked {
		if before, _, _ := strings.Cut(line, "?"); before == method+" "+path {
			named = append(named, r.heard[at].Get(stream.UserHeader))
		}
	}
	return named
}

func refuseConnection(w http.ResponseWriter, message string) {
	answer(w, http.StatusBadRequest, acceleration.ErrorResponse{Error: acceleration.ErrorDetail{
		Type: "invalid_request", Code: "invalid_request", Message: message,
	}})
}

func connectionNotFound(w http.ResponseWriter) {
	answer(w, http.StatusNotFound, acceleration.ErrorResponse{Error: acceleration.ErrorDetail{
		Type: "not_found", Code: "connection_not_found", Message: "no such connection",
	}})
}

// connectFor creates a connection to connector for user, failing the test if it cannot.
func connectFor(t *testing.T, connections *Connections, connector, user string) acceleration.Connection {
	t.Helper()
	created, err := connections.Create(t.Context(), ConnectionRequest{Connector: connector, UserID: user})
	if err != nil {
		t.Fatal(err)
	}
	return created
}

// Catches a user connection whose owner and X-Stream-User-Id differ, or that names no user in
// one of them: the router refuses either.
func TestAUsersConnectionNamesThemAsOwnerAndActingUser(t *testing.T) {
	backend := newRouter(t)

	created, err := backend.client(t).Connections().Create(t.Context(),
		ConnectionRequest{Connector: "linear", Label: "work", UserID: "anya"})
	if err != nil {
		t.Fatal(err)
	}

	want := map[string]any{"connector_id": "linear", "label": "work", "owner": map[string]any{"type": "user", "user_id": "anya"}}
	if sent := backend.body(t, "POST", "/v1/agents/connections"); !reflect.DeepEqual(sent, want) {
		t.Errorf("the router was sent %v", sent)
	}
	if acting := backend.actingFor("POST", "/v1/agents/connections"); !slices.Equal(acting, []string{"anya"}) {
		t.Errorf("the create acted for %q", acting)
	}
	if *created.Owner.UserId != "anya" || created.Status != acceleration.ConnectionStatusPending {
		t.Errorf("the connection came back as %+v", created)
	}
}

// Catches an app connection sent with an owner.user_id, or with the user the client otherwise
// acts for in X-Stream-User-Id, which would make each read depend on who the client was built for.
func TestAnAppConnectionNamesNoUser(t *testing.T) {
	backend := newRouter(t)
	api, err := New(stream.Backend{URL: backend.URL, CustomerID: "acme", ActingFor: "bob"}) //nolint:staticcheck // SA1019: the client acts for a user, which a call that names none must not send
	if err != nil {
		t.Fatal(err)
	}

	created := connectFor(t, api.Connections(), "github", "")
	if _, err := api.Connections().Get(t.Context(), *created.Id, ConnectionOptions{UserID: ""}); err != nil {
		t.Fatal(err)
	}

	want := map[string]any{"connector_id": "github", "owner": map[string]any{"type": "app"}}
	if sent := backend.body(t, "POST", "/v1/agents/connections"); !reflect.DeepEqual(sent, want) {
		t.Errorf("the router was sent %v", sent)
	}
	for _, path := range []string{"POST /v1/agents/connections", "GET /v1/agents/connections/" + *created.Id} {
		method, route, _ := strings.Cut(path, " ")
		if acting := backend.actingFor(method, route); !slices.Equal(acting, []string{""}) {
			t.Errorf("%s acted for %q", path, acting)
		}
	}
}

// Catches an authorization that drops the return URL or the user, or comes back without the
// launch URL, the handoff token or when the attempt ends.
func TestAuthorizingAnswersWhereAndHowLongTheConsentRuns(t *testing.T) {
	backend := newRouter(t)
	connections := backend.client(t).Connections()
	created := connectFor(t, connections, "linear", "anya")
	path := "/v1/agents/connections/" + *created.Id + "/authorizations"

	started, err := connections.Authorize(t.Context(), *created.Id,
		AuthorizeOptions{ReturnURL: "https://app.example.com/settings", UserID: "anya"})
	if err != nil {
		t.Fatal(err)
	}

	if sent := backend.body(t, "POST", path); !reflect.DeepEqual(sent, map[string]any{"return_url": "https://app.example.com/settings"}) {
		t.Errorf("the router was sent %v", sent)
	}
	if acting := backend.actingFor("POST", path); !slices.Equal(acting, []string{"anya"}) {
		t.Errorf("the authorization acted for %q", acting)
	}
	want := acceleration.Authorization{
		Id: "attempt-1", Kind: acceleration.Consent, HandoffToken: "handoff-1", ExpiresAt: pinnedAt.Add(10 * time.Minute),
		LaunchUrl: "https://router.example.com/v1/agents/connections/" + *created.Id + "/launch",
	}
	if !reflect.DeepEqual(started, want) {
		t.Errorf("the authorization came back as %+v", started)
	}

	if _, err := connections.Authorize(t.Context(), *created.Id, AuthorizeOptions{UserID: "anya"}); err != nil {
		t.Fatal(err)
	}
	if sent := backend.body(t, "POST", path); len(sent) != 0 {
		t.Errorf("an authorization with no return URL sent %v", sent)
	}
}

// Catches a read, an authorization or a delete that sends no user, or another than the one
// named: the owner reaches their connection, and anybody else is told it does not exist.
func TestAUsersConnectionIsReachedOnlyAsThem(t *testing.T) {
	backend := newRouter(t)
	connections := backend.client(t).Connections()
	created := connectFor(t, connections, "linear", "anya")

	got, err := connections.Get(t.Context(), *created.Id, ConnectionOptions{UserID: "anya"})
	if err != nil {
		t.Fatal(err)
	}
	if *got.Id != *created.Id || got.Status != acceleration.ConnectionStatusPending {
		t.Errorf("the connection came back as %+v", got)
	}
	for _, user := range []string{"bob", ""} {
		_, read := connections.Get(t.Context(), *created.Id, ConnectionOptions{UserID: user})
		_, authorized := connections.Authorize(t.Context(), *created.Id, AuthorizeOptions{UserID: user})
		deleted := connections.Delete(t.Context(), *created.Id, ConnectionOptions{UserID: user})
		for name, err := range map[string]error{"read": read, "authorized": authorized, "deleted": deleted} {
			var refused *stream.RouterError
			if !errors.As(err, &refused) || refused.Status != http.StatusNotFound || refused.Code != "connection_not_found" {
				t.Errorf("anya's connection %s as %q: %v", name, user, err)
			}
		}
	}
	if _, err := connections.Get(t.Context(), *created.Id, ConnectionOptions{UserID: "anya"}); err != nil {
		t.Errorf("another user deleted anya's connection: %v", err)
	}
}

// Catches a list that reads one page only, lists the app's when a user is named or a user's when
// none is, or drops the connector.
func TestListingReadsTheOwnersConnectionsEveryPage(t *testing.T) {
	backend := newRouter(t)
	connections := backend.client(t).Connections()
	first := connectFor(t, connections, "linear", "anya")
	second := connectFor(t, connections, "github", "anya")
	third := connectFor(t, connections, "linear", "anya")
	connectFor(t, connections, "linear", "bob")
	app := connectFor(t, connections, "linear", "")

	for _, c := range []struct {
		filter ConnectionFilter
		want   []string
		query  url.Values
		acting string
	}{
		{ConnectionFilter{UserID: "anya"}, []string{*third.Id, *second.Id, *first.Id},
			url.Values{"owner_type": {"user"}, "limit": {"200"}}, "anya"},
		{ConnectionFilter{UserID: "anya", Connector: "linear"}, []string{*third.Id, *first.Id},
			url.Values{"owner_type": {"user"}, "connector_id": {"linear"}, "limit": {"200"}}, "anya"},
		{ConnectionFilter{}, []string{*app.Id}, url.Values{"owner_type": {"app"}, "limit": {"200"}}, ""},
	} {
		backend.mu.Lock()
		backend.asked, backend.heard = nil, nil
		backend.mu.Unlock()

		listed, err := connections.List(t.Context(), c.filter)
		if err != nil {
			t.Fatal(err)
		}
		var ids []string
		for _, connection := range listed {
			ids = append(ids, *connection.Id)
		}
		if !slices.Equal(ids, c.want) {
			t.Errorf("%+v listed %v", c.filter, ids)
		}
		if query := backend.query(t, "GET", "/v1/agents/connections"); !reflect.DeepEqual(query, c.query) {
			t.Errorf("%+v asked %v", c.filter, query)
		}
		pages := (len(c.want) + standInConnectionsPage - 1) / standInConnectionsPage
		if acting := backend.actingFor("GET", "/v1/agents/connections"); !slices.Equal(acting, slices.Repeat([]string{c.acting}, pages)) {
			t.Errorf("%+v acted for %q", c.filter, acting)
		}
	}
}

// Catches a delete that is not sent, is sent as somebody else, or reports success for a refusal.
func TestDeletingAConnectionRemovesIt(t *testing.T) {
	backend := newRouter(t)
	connections := backend.client(t).Connections()
	created := connectFor(t, connections, "linear", "anya")

	if err := connections.Delete(t.Context(), *created.Id, ConnectionOptions{UserID: "anya"}); err != nil {
		t.Fatal(err)
	}

	path := "/v1/agents/connections/" + *created.Id
	if acting := backend.actingFor("DELETE", path); !slices.Equal(acting, []string{"anya"}) {
		t.Errorf("the delete acted for %q", acting)
	}
	if query := backend.query(t, "DELETE", path); len(query) != 0 {
		t.Errorf("the delete asked %v", query)
	}
	var refused *stream.RouterError
	if _, err := connections.Get(t.Context(), *created.Id, ConnectionOptions{UserID: "anya"}); !errors.As(err, &refused) || refused.Status != http.StatusNotFound {
		t.Errorf("the deleted connection was read: %v", err)
	}
	if err := connections.Delete(t.Context(), *created.Id, ConnectionOptions{UserID: "anya"}); !errors.As(err, &refused) || refused.Code != "connection_not_found" {
		t.Errorf("deleting it again answered %v", err)
	}
}

// Catches an empty connection id sent to the router, as /v1/agents/connections/. Counted at the
// server, since no route of the stand-in router matches that path to record it.
func TestAConnectionWithNoIDIsRefusedBeforeTheRouter(t *testing.T) {
	var arrived atomic.Int32
	counting := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		arrived.Add(1)
		connectionNotFound(w)
	}))
	t.Cleanup(counting.Close)
	api, err := New(stream.Backend{URL: counting.URL, CustomerID: "acme"})
	if err != nil {
		t.Fatal(err)
	}
	connections := api.Connections()

	if _, err := connections.Get(t.Context(), "", ConnectionOptions{UserID: "anya"}); err == nil {
		t.Error("a connection with no id was read")
	}
	if _, err := connections.Authorize(t.Context(), "", AuthorizeOptions{UserID: "anya"}); err == nil {
		t.Error("a connection with no id was authorized")
	}
	if err := connections.Delete(t.Context(), "", ConnectionOptions{UserID: "anya"}); err == nil {
		t.Error("a connection with no id was deleted")
	}
	if got := arrived.Load(); got != 0 {
		t.Errorf("the router was asked %d times", got)
	}
}

// Catches a refusal or a failed request that comes back as an empty connection rather than as
// the router's error, or without the context's.
func TestAConnectionRefusalIsTheRoutersError(t *testing.T) {
	refusing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		answer(w, http.StatusServiceUnavailable, acceleration.ErrorResponse{Error: acceleration.ErrorDetail{
			Type: "unavailable", Code: "connectors_unavailable", Message: "connections are not configured",
		}})
	}))
	t.Cleanup(refusing.Close)
	api, err := New(stream.Backend{URL: refusing.URL, CustomerID: "acme"})
	if err != nil {
		t.Fatal(err)
	}
	connections := api.Connections()
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()

	for _, ctx := range []context.Context{t.Context(), cancelled} {
		_, created := connections.Create(ctx, ConnectionRequest{Connector: "linear", UserID: "anya"})
		_, authorized := connections.Authorize(ctx, "connection-1", AuthorizeOptions{UserID: "anya"})
		_, read := connections.Get(ctx, "connection-1", ConnectionOptions{UserID: "anya"})
		_, listed := connections.List(ctx, ConnectionFilter{UserID: "anya"})
		deleted := connections.Delete(ctx, "connection-1", ConnectionOptions{UserID: "anya"})
		for name, err := range map[string]error{"create": created, "authorize": authorized, "get": read, "list": listed, "delete": deleted} {
			var refused *stream.RouterError
			switch {
			case ctx == cancelled && !errors.Is(err, context.Canceled):
				t.Errorf("a cancelled %s answered %v", name, err)
			case ctx != cancelled && (!errors.As(err, &refused) || refused.Status != http.StatusServiceUnavailable || refused.Code != "connectors_unavailable"):
				t.Errorf("a refused %s answered %v", name, err)
			}
		}
	}
}

// Catches docs and code that disagree on reach: a call naming a user reaches the app's
// connections, as the router lets every backend, while its List holds the user's only.
func TestACallNamingAUserReachesAnAppConnectionButListsOnlyTheUsers(t *testing.T) {
	backend := newRouter(t)
	connections := backend.client(t).Connections()
	app := connectFor(t, connections, "github", "")
	connectFor(t, connections, "linear", "alice")

	if _, err := connections.Get(t.Context(), *app.Id, ConnectionOptions{UserID: "alice"}); err != nil {
		t.Errorf("a call naming alice could not read the app's connection: %v", err)
	}
	listed, err := connections.List(t.Context(), ConnectionFilter{UserID: "alice"})
	if err != nil {
		t.Fatal(err)
	}
	if len(listed) != 1 || listed[0].ConnectorId != "linear" {
		t.Errorf("alice's list held %+v, not only her linear connection", listed)
	}
}

// Catches a call that names the user in the owner and not in X-Stream-User-Id (or the reverse),
// reads included, and an app call that sends a header.
func TestEveryCallSendsTheUserItNamesAsTheActingUser(t *testing.T) {
	backend := newRouter(t)
	connections := backend.client(t).Connections()
	app := connectFor(t, connections, "github", "")
	created := connectFor(t, connections, "linear", "alice")
	one := "/v1/agents/connections/" + *created.Id
	if _, err := connections.Authorize(t.Context(), *created.Id, AuthorizeOptions{UserID: "alice"}); err != nil {
		t.Fatal(err)
	}
	if _, err := connections.Get(t.Context(), *created.Id, ConnectionOptions{UserID: "alice"}); err != nil {
		t.Fatal(err)
	}
	if _, err := connections.List(t.Context(), ConnectionFilter{UserID: "alice"}); err != nil {
		t.Fatal(err)
	}
	if err := connections.Delete(t.Context(), *created.Id, ConnectionOptions{UserID: "alice"}); err != nil {
		t.Fatal(err)
	}
	for _, call := range []string{"POST /v1/agents/connections", "POST " + one + "/authorizations", "GET " + one, "GET /v1/agents/connections", "DELETE " + one} {
		method, path, _ := strings.Cut(call, " ")
		want := []string{"alice"}
		if call == "POST /v1/agents/connections" {
			want = []string{"", "alice"} // the app's connection came first
		}
		if acting := backend.actingFor(method, path); !slices.Equal(acting, want) {
			t.Errorf("%s acted for %q", call, acting)
		}
	}
	if owner := backend.body(t, "POST", "/v1/agents/connections")["owner"]; !reflect.DeepEqual(owner, map[string]any{"type": "user", "user_id": "alice"}) {
		t.Errorf("the owner was %v", owner)
	}

	if _, err := connections.List(t.Context(), ConnectionFilter{}); err != nil {
		t.Fatal(err)
	}
	if err := connections.Delete(t.Context(), *app.Id, ConnectionOptions{UserID: ""}); err != nil {
		t.Fatal(err)
	}
	if acting := backend.actingFor("DELETE", "/v1/agents/connections/"+*app.Id); !slices.Equal(acting, []string{""}) {
		t.Errorf("an app delete acted for %q", acting)
	}
	if acting := backend.actingFor("GET", "/v1/agents/connections"); !slices.Equal(acting, []string{"alice", ""}) {
		t.Errorf("the lists acted for %q", acting)
	}
}
