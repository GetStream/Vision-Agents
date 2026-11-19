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
// Every method names the user once. A user-owned connection is that user's, and is reached
// only while acting for them: the user goes to the router as the owner and as X-Stream-User-Id
// alike, so the two never differ. No user is the app's own connection.
type Connections struct {
	client *Client
}

// Connections is the app's and its users' connections.
func (c *Client) Connections() *Connections { return &Connections{client: c} }

// ConnectionRequest is a connection to create, pending until an account is connected.
type ConnectionRequest struct {
	// Connector is a built-in, such as linear, or one of the app's own.
	Connector string
	// Label tells the owner's connections to the same connector apart. Empty is none.
	Label string
	// User is the app's user the connection is for. Empty is the app's own.
	User string
}

// AuthorizeOptions is what starting a consent asks.
type AuthorizeOptions struct {
	// ReturnURL is the app's page the browser goes back to when the consent ends, with
	// connection_id and status added to its query. Empty goes back to the dashboard.
	ReturnURL string
	// User is the user the connection belongs to. Empty is an app connection.
	User string
}

// ConnectionFilter is which connections List reads.
type ConnectionFilter struct {
	// User reads that user's connections. Empty reads the app's own.
	User string
	// Connector keeps one connector's. Empty is every connector's.
	Connector string
}

// Create stores a pending connection to Connector, owned by User or else by the app.
func (c *Connections) Create(ctx context.Context, request ConnectionRequest) (acceleration.Connection, error) {
	owner := acceleration.ConnectionOwner{Type: acceleration.ConnectionOwnerTypeApp}
	if request.User != "" {
		owner = acceleration.ConnectionOwner{Type: acceleration.ConnectionOwnerTypeUser, UserId: &request.User}
	}
	api, err := c.api(request.User)
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
// connection, a reconnect for one connected before. Open the LaunchUrl it returns in a popup
// from a page of ReturnURL's origin and post it the HandoffToken; the attempt ends at
// ExpiresAt.
func (c *Connections) Authorize(ctx context.Context, id string, options AuthorizeOptions) (acceleration.Authorization, error) {
	api, err := c.reach(id, options.User)
	if err != nil {
		return acceleration.Authorization{}, err
	}
	what := "authorizing the connection " + id
	started, err := api.CreateAuthorizationWithResponse(ctx, id, acceleration.AuthorizationRequest{
		ReturnUrl: pointer(options.ReturnURL),
	})
	if err != nil {
		return acceleration.Authorization{}, fmt.Errorf("client: %s: %w", what, err)
	}
	if started.JSON201 == nil {
		return acceleration.Authorization{}, failure(what, started.HTTPResponse, started.Body)
	}
	return *started.JSON201, nil
}

// Get is one connection of user's, or of the app's when user is empty. Another user's is not
// found, the same as one that does not exist.
func (c *Connections) Get(ctx context.Context, id, user string) (acceleration.Connection, error) {
	api, err := c.reach(id, user)
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

// List is every connection the filter matches, newest first, read page by page.
func (c *Connections) List(ctx context.Context, filter ConnectionFilter) ([]acceleration.Connection, error) {
	owner := acceleration.ConnectionOwnerTypeApp
	if filter.User != "" {
		owner = acceleration.ConnectionOwnerTypeUser
	}
	params := acceleration.ListConnectionsParams{
		OwnerType: &owner, ConnectorId: pointer(filter.Connector), Limit: pointer(int64(connectionsPage)),
	}
	api, err := c.api(filter.User)
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

// Delete disconnects a connection of user's, or of the app's when user is empty, and drops
// its credentials. One an agent config binds as its fixed connection is refused with a 409.
func (c *Connections) Delete(ctx context.Context, id, user string) error {
	api, err := c.reach(id, user)
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
func (c *Connections) reach(id, user string) (*acceleration.ClientWithResponses, error) {
	if id == "" {
		return nil, errors.New("client: a connection needs its id")
	}
	return c.api(user)
}

// api is the generated client acting for user, or for nobody when user is empty, whoever this
// client acts for otherwise.
func (c *Connections) api(user string) (*acceleration.ClientWithResponses, error) {
	backend := c.client.backend
	backend.ActingFor = user
	return backend.Client()
}
