package client

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// connectionsPage is the page size List reads in, the most the router answers.
const connectionsPage = 200

// Connections are the accounts the app and its users hold at connectors, such as a user's
// Linear: what an agent's tools reach that account through. Only a backend may ask.
//
// The user is named once, on the client: alice, err := api.ForUser("alice"), then
// alice.Connections(). The user goes to the router as the owner and as X-Stream-User-Id alike,
// so the two never differ. A user-owned connection is reached only while acting for that user;
// an app-owned one is reached by every backend client, one acting for a user included. On a
// client acting for nobody, Create and List are the app's own.
type Connections struct {
	client *Client
}

// Connections is the connections resource of this client: Create and List are the user's this
// client acts for (ForUser), or the app's own when it acts for nobody.
func (c *Client) Connections() *Connections { return &Connections{client: c} }

// ForUser is a client that runs work for one end user while keeping the app's credential: a
// backend's, so it is refused on a device client, which holds its user's token (SetUser).
//
// It returns a client rather than mutating this one, as SetUser does, so which user a request
// was for never depends on when it happened to run. The new client carries none of this one's
// team or usage categories: the user's work is not counted against them, as with SetUser.
func (c *Client) ForUser(userID string) (*Client, error) {
	if !c.ServerSide() {
		return nil, errors.New("client: working for a user keeps the app's credential, which only a " +
			"backend holds; a device holds its user's token (SetUser)")
	}
	if userID == "" {
		return nil, errors.New("client: working for a user needs the user's id")
	}
	backend := c.backend
	backend.ActingFor = userID
	backend.Team, backend.UsageCategories = "", nil
	resolved, err := backend.Resolve()
	if err != nil {
		return nil, err
	}
	return &Client{backend: resolved, watcher: c.watcher}, nil
}

// ConnectionRequest is a connection to create, pending until an account is connected.
type ConnectionRequest struct {
	// Connector is a built-in, such as linear, or one of the app's own.
	Connector string
	// Label tells the owner's connections to the same connector apart. Empty is none.
	Label string
}

// Create stores a pending connection to Connector, owned by the user this client acts for or
// else by the app.
func (c *Connections) Create(ctx context.Context, request ConnectionRequest) (acceleration.Connection, error) {
	owner := acceleration.ConnectionOwner{Type: acceleration.ConnectionOwnerTypeApp}
	if user := c.user(); user != "" {
		owner = acceleration.ConnectionOwner{Type: acceleration.ConnectionOwnerTypeUser, UserId: &user}
	}
	api, err := c.client.backend.Client()
	if err != nil {
		return acceleration.Connection{}, err
	}
	what := "creating a connection to " + request.Connector
	created, err := api.CreateConnectionWithResponse(ctx, acceleration.ConnectionRequest{
		ConnectorId: request.Connector, Label: pointer(request.Label), Owner: owner,
	})
	if err != nil {
		return acceleration.Connection{}, fmt.Errorf("client: %s: %w", what, err)
	}
	if created.JSON201 == nil {
		return acceleration.Connection{}, failure(what, created.HTTPResponse, created.Body)
	}
	return *created.JSON201, nil
}

// Authorize starts the provider's consent for a connection: a first one for a pending
// connection, a reconnect for one connected before. returnURL is the app's page the browser
// goes back to when the consent ends, with connection_id and status added to its query; empty
// goes back to the dashboard. Open the LaunchUrl it returns in a popup from a page of
// returnURL's origin and post it the HandoffToken; the attempt ends at ExpiresAt.
func (c *Connections) Authorize(ctx context.Context, id, returnURL string) (acceleration.Authorization, error) {
	api, err := c.reach(id)
	if err != nil {
		return acceleration.Authorization{}, err
	}
	what := "authorizing the connection " + id
	started, err := api.CreateAuthorizationWithResponse(ctx, id, acceleration.AuthorizationRequest{
		ReturnUrl: pointer(returnURL),
	})
	if err != nil {
		return acceleration.Authorization{}, fmt.Errorf("client: %s: %w", what, err)
	}
	if started.JSON201 == nil {
		return acceleration.Authorization{}, failure(what, started.HTTPResponse, started.Body)
	}
	return *started.JSON201, nil
}

// Get is one connection. An app's is reached by every backend client; a user's only while
// acting for that user, and for anybody else it is not found, the same as one that does not
// exist.
func (c *Connections) Get(ctx context.Context, id string) (acceleration.Connection, error) {
	api, err := c.reach(id)
	if err != nil {
		return acceleration.Connection{}, err
	}
	what := "reading the connection " + id
	got, err := api.GetConnectionWithResponse(ctx, id)
	if err != nil {
		return acceleration.Connection{}, fmt.Errorf("client: %s: %w", what, err)
	}
	if got.JSON200 == nil {
		return acceleration.Connection{}, failure(what, got.HTTPResponse, got.Body)
	}
	return *got.JSON200, nil
}

// List is every connection of the user this client acts for (only theirs, not the app's), or of
// the app's when it acts for nobody, newest first,
// read page by page. connector keeps one connector's; empty is every connector's.
func (c *Connections) List(ctx context.Context, connector string) ([]acceleration.Connection, error) {
	owner := acceleration.ConnectionOwnerTypeApp
	if c.user() != "" {
		owner = acceleration.ConnectionOwnerTypeUser
	}
	params := acceleration.ListConnectionsParams{
		OwnerType: &owner, ConnectorId: pointer(connector), Limit: pointer(int64(connectionsPage)),
	}
	api, err := c.client.backend.Client()
	if err != nil {
		return nil, err
	}
	var connections []acceleration.Connection
	for {
		page, err := api.ListConnectionsWithResponse(ctx, &params)
		if err != nil {
			return nil, fmt.Errorf("client: listing connections: %w", err)
		}
		if page.JSON200 == nil {
			return nil, failure("listing connections", page.HTTPResponse, page.Body)
		}
		if page.JSON200.Items != nil {
			connections = append(connections, *page.JSON200.Items...)
		}
		if !page.JSON200.HasMore || page.JSON200.NextCursor == nil || *page.JSON200.NextCursor == "" {
			return connections, nil
		}
		params.Cursor = page.JSON200.NextCursor
	}
}

// Delete disconnects a connection of the user this client acts for, or of the app's, and drops
// its credentials. One an agent config binds as its fixed connection is refused with a 409.
func (c *Connections) Delete(ctx context.Context, id string) error {
	api, err := c.reach(id)
	if err != nil {
		return err
	}
	what := "deleting the connection " + id
	deleted, err := api.DeleteConnectionWithResponse(ctx, id, &acceleration.DeleteConnectionParams{})
	if err != nil {
		return fmt.Errorf("client: %s: %w", what, err)
	}
	if deleted.StatusCode() != http.StatusNoContent {
		return failure(what, deleted.HTTPResponse, deleted.Body)
	}
	return nil
}

// reach is the generated client for one connection's requests. An empty id is refused before
// anything is sent: it would ask for /v1/agents/connections/, which names no connection.
func (c *Connections) reach(id string) (*acceleration.ClientWithResponses, error) {
	if id == "" {
		return nil, errors.New("client: a connection needs its id")
	}
	return c.client.backend.Client()
}

// user is who the connections are for, sent as X-Stream-User-Id by the generated client from
// the same value: empty is the app.
func (c *Connections) user() string { return c.client.backend.ActingFor }
