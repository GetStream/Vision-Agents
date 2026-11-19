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

// asUser is a client acting for user through ForUser, failing the test if it cannot.
func (r *router) asUser(t *testing.T, user string) *Client {
	t.Helper()
	acting, err := r.client(t).ForUser(user)
	if err != nil {
		t.Fatal(err)
	}
	return acting
}

// connect creates a connection to connector for whoever connections acts for, failing the test
// if it cannot.
func connect(t *testing.T, connections *Connections, connector string) acceleration.Connection {
	t.Helper()
	created, err := connections.Create(t.Context(), ConnectionRequest{Connector: connector})
	if err != nil {
		t.Fatal(err)
	}
	return created
}

// Catches a user connection whose owner and X-Stream-User-Id differ, or that names no user in
// one of them: the router refuses either.
func TestAUsersConnectionNamesThemAsOwnerAndActingUser(t *testing.T) {
	backend := newRouter(t)

	created, err := backend.asUser(t, "anya").Connections().Create(t.Context(),
		ConnectionRequest{Connector: "linear", Label: "work"})
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

// Catches an app connection sent with an owner.user_id or X-Stream-User-Id, reads included.
func TestAnAppConnectionNamesNoUser(t *testing.T) {
	backend := newRouter(t)
	connections := backend.client(t).Connections()

	created := connect(t, connections, "github")
	if _, err := connections.Get(t.Context(), *created.Id); err != nil {
		t.Fatal(err)
	}
	if _, err := connections.List(t.Context(), ""); err != nil {
		t.Fatal(err)
	}

	want := map[string]any{"connector_id": "github", "owner": map[string]any{"type": "app"}}
	if sent := backend.body(t, "POST", "/v1/agents/connections"); !reflect.DeepEqual(sent, want) {
		t.Errorf("the router was sent %v", sent)
	}
	for _, path := range []string{"POST /v1/agents/connections", "GET /v1/agents/connections/" + *created.Id, "GET /v1/agents/connections"} {
		method, route, _ := strings.Cut(path, " ")
		if acting := backend.actingFor(method, route); !slices.Equal(acting, []string{""}) {
			t.Errorf("%s acted for %q", path, acting)
		}
	}
}

// Catches a client for a user that sends the user on some requests only, or that changes the
// client it was made from: that one must stay the app's.
func TestForUserActsOnEveryConnectionRequestAndLeavesTheParent(t *testing.T) {
	backend := newRouter(t)
	app := backend.client(t)
	anya, err := app.ForUser("anya")
	if err != nil {
		t.Fatal(err)
	}
	connections := anya.Connections()
	created := connect(t, connections, "linear")
	id := *created.Id

	if _, err := connections.Get(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	if _, err := connections.List(t.Context(), ""); err != nil {
		t.Fatal(err)
	}
	if _, err := connections.Authorize(t.Context(), id, ""); err != nil {
		t.Fatal(err)
	}
	if err := connections.Delete(t.Context(), id); err != nil {
		t.Fatal(err)
	}
	connect(t, app.Connections(), "github")

	for _, path := range []string{
		"POST /v1/agents/connections", "GET /v1/agents/connections", "GET /v1/agents/connections/" + id,
		"POST /v1/agents/connections/" + id + "/authorizations", "DELETE /v1/agents/connections/" + id,
	} {
		method, route, _ := strings.Cut(path, " ")
		want := []string{"anya"}
		if path == "POST /v1/agents/connections" {
			want = []string{"anya", ""}
		}
		if acting := backend.actingFor(method, route); !slices.Equal(acting, want) {
			t.Errorf("%s acted for %q, want %q", path, acting, want)
		}
	}
	if !app.ServerSide() || app.Backend().ActingFor != "" || anya.Backend().ActingFor != "anya" {
		t.Errorf("the parent acts for %q, the child for %q", app.Backend().ActingFor, anya.Backend().ActingFor)
	}
}

// Catches a ForUser that hands the app's credential to a device client, or accepts no user.
func TestForUserIsRefusedOnADeviceAndForNobody(t *testing.T) {
	backend := newRouter(t)
	device, err := backend.client(t).SetUser("anya", "anya-token")
	if err != nil {
		t.Fatal(err)
	}

	if _, err := device.ForUser("bob"); err == nil {
		t.Error("a device client worked for another user")
	}
	if _, err := backend.client(t).ForUser(""); err == nil {
		t.Error("a client worked for no user")
	}
}

// Catches an authorization that drops the return URL or the user, or comes back without the
// launch URL, the handoff token or when the attempt ends.
func TestAuthorizingAnswersWhereAndHowLongTheConsentRuns(t *testing.T) {
	backend := newRouter(t)
	connections := backend.asUser(t, "anya").Connections()
	created := connect(t, connections, "linear")
	path := "/v1/agents/connections/" + *created.Id + "/authorizations"

	started, err := connections.Authorize(t.Context(), *created.Id, "https://app.example.com/settings")
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

	if _, err := connections.Authorize(t.Context(), *created.Id, ""); err != nil {
		t.Fatal(err)
	}
	if sent := backend.body(t, "POST", path); len(sent) != 0 {
		t.Errorf("an authorization with no return URL sent %v", sent)
	}
}

// Catches a read, an authorization or a delete that sends no user, or another than the one
// the client acts for: the owner reaches their connection, and anybody else is told it does not
// exist.
func TestAUsersConnectionIsReachedOnlyAsThem(t *testing.T) {
	backend := newRouter(t)
	anya := backend.asUser(t, "anya").Connections()
	created := connect(t, anya, "linear")

	got, err := anya.Get(t.Context(), *created.Id)
	if err != nil {
		t.Fatal(err)
	}
	if *got.Id != *created.Id || got.Status != acceleration.ConnectionStatusPending {
		t.Errorf("the connection came back as %+v", got)
	}
	for user, others := range map[string]*Connections{
		"bob": backend.asUser(t, "bob").Connections(), "nobody": backend.client(t).Connections(),
	} {
		_, read := others.Get(t.Context(), *created.Id)
		_, authorized := others.Authorize(t.Context(), *created.Id, "")
		deleted := others.Delete(t.Context(), *created.Id)
		for name, err := range map[string]error{"read": read, "authorized": authorized, "deleted": deleted} {
			var refused *stream.RouterError
			if !errors.As(err, &refused) || refused.Status != http.StatusNotFound || refused.Code != "connection_not_found" {
				t.Errorf("anya's connection %s as %s: %v", name, user, err)
			}
		}
	}
	if _, err := anya.Get(t.Context(), *created.Id); err != nil {
		t.Errorf("another user deleted anya's connection: %v", err)
	}
}

// Catches a list that reads one page only, lists the app's when a user is acted for or a user's
// when none is, or drops the connector.
func TestListingReadsTheOwnersConnectionsEveryPage(t *testing.T) {
	backend := newRouter(t)
	anya := backend.asUser(t, "anya").Connections()
	first := connect(t, anya, "linear")
	second := connect(t, anya, "github")
	third := connect(t, anya, "linear")
	connect(t, backend.asUser(t, "bob").Connections(), "linear")
	app := backend.client(t).Connections()
	appOwned := connect(t, app, "linear")

	for _, c := range []struct {
		connections *Connections
		connector   string
		want        []string
		query       url.Values
		acting      string
	}{
		{anya, "", []string{*third.Id, *second.Id, *first.Id},
			url.Values{"owner_type": {"user"}, "limit": {"200"}}, "anya"},
		{anya, "linear", []string{*third.Id, *first.Id},
			url.Values{"owner_type": {"user"}, "connector_id": {"linear"}, "limit": {"200"}}, "anya"},
		{app, "", []string{*appOwned.Id}, url.Values{"owner_type": {"app"}, "limit": {"200"}}, ""},
	} {
		backend.mu.Lock()
		backend.asked, backend.heard = nil, nil
		backend.mu.Unlock()

		listed, err := c.connections.List(t.Context(), c.connector)
		if err != nil {
			t.Fatal(err)
		}
		var ids []string
		for _, connection := range listed {
			ids = append(ids, *connection.Id)
		}
		if !slices.Equal(ids, c.want) {
			t.Errorf("%q for %q listed %v", c.connector, c.acting, ids)
		}
		if query := backend.query(t, "GET", "/v1/agents/connections"); !reflect.DeepEqual(query, c.query) {
			t.Errorf("%q for %q asked %v", c.connector, c.acting, query)
		}
		pages := (len(c.want) + standInConnectionsPage - 1) / standInConnectionsPage
		if acting := backend.actingFor("GET", "/v1/agents/connections"); !slices.Equal(acting, slices.Repeat([]string{c.acting}, pages)) {
			t.Errorf("%q for %q acted for %q", c.connector, c.acting, acting)
		}
	}
}

// Catches a delete that is not sent, is sent as somebody else, or reports success for a refusal.
func TestDeletingAConnectionRemovesIt(t *testing.T) {
	backend := newRouter(t)
	connections := backend.asUser(t, "anya").Connections()
	created := connect(t, connections, "linear")

	if err := connections.Delete(t.Context(), *created.Id); err != nil {
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
	if _, err := connections.Get(t.Context(), *created.Id); !errors.As(err, &refused) || refused.Status != http.StatusNotFound {
		t.Errorf("the deleted connection was read: %v", err)
	}
	if err := connections.Delete(t.Context(), *created.Id); !errors.As(err, &refused) || refused.Code != "connection_not_found" {
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

	if _, err := connections.Get(t.Context(), ""); err == nil {
		t.Error("a connection with no id was read")
	}
	if _, err := connections.Authorize(t.Context(), "", ""); err == nil {
		t.Error("a connection with no id was authorized")
	}
	if err := connections.Delete(t.Context(), ""); err == nil {
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
	anya, err := api.ForUser("anya")
	if err != nil {
		t.Fatal(err)
	}
	connections := anya.Connections()
	cancelled, cancel := context.WithCancel(t.Context())
	cancel()

	for _, ctx := range []context.Context{t.Context(), cancelled} {
		_, created := connections.Create(ctx, ConnectionRequest{Connector: "linear"})
		_, authorized := connections.Authorize(ctx, "connection-1", "")
		_, read := connections.Get(ctx, "connection-1")
		_, listed := connections.List(ctx, "")
		deleted := connections.Delete(ctx, "connection-1")
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

// Catches a ForUser client that keeps the parent's team and usage categories, which counts the
// user's work against the app client's own team.
func TestForUserCountsNothingAgainstTheParentsTeam(t *testing.T) {
	backend := newRouter(t)
	api, err := New(stream.Backend{URL: backend.URL, CustomerID: "acme", Team: "finance", UsageCategories: []string{"project:x"}})
	if err != nil {
		t.Fatal(err)
	}

	alice, err := api.ForUser("alice")
	if err != nil {
		t.Fatal(err)
	}
	connect(t, alice.Connections(), "linear")

	sent := backend.heard[len(backend.heard)-1]
	if got := sent.Get(stream.UsageCategoriesHeader); got != "" {
		t.Errorf("alice's request carried the usage categories %q", got)
	}
}

// Catches docs and code that disagree on reach: a backend acting for a user reaches the app's
// connections, as the router lets every backend, while its List holds the user's only.
func TestAUserClientReachesAnAppConnectionButListsOnlyItsOwn(t *testing.T) {
	backend := newRouter(t)
	app := connect(t, backend.client(t).Connections(), "github")
	alice := backend.asUser(t, "alice").Connections()

	if _, err := alice.Get(t.Context(), *app.Id); err != nil {
		t.Errorf("a user's client could not read the app's connection: %v", err)
	}
	listed, err := alice.List(t.Context(), "")
	if err != nil {
		t.Fatal(err)
	}
	if len(listed) != 0 {
		t.Errorf("alice's list held %d connections of the app's", len(listed))
	}
}
