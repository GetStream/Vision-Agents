package api

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/danielgtaylor/huma/v2"
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/eventforward"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// errNoEventForwarding is every destination operation on a deployment that forwards nothing:
// without connectors there is no key to seal a destination's secret with.
var errNoEventForwarding = notConfigured("event destinations are not available: connectors are not enabled on this deployment")

const noEventDestination = "the connector has no such event destination"

// ConnectorEventDestination is one URL a connector's raw provider events are forwarded to.
// Its signing secret is never shown after it is made.
type ConnectorEventDestination struct {
	ID          string                `json:"id" readOnly:"true"`
	ConnectorID string                `json:"connector_id" readOnly:"true"`
	URL         string                `json:"url"`
	Forward     ConnectorEventForward `json:"forward"`
	// PreviousSecretUntil is store.EventDestination.PreviousUntil while it is ahead.
	PreviousSecretUntil *time.Time `json:"previous_secret_until,omitempty" readOnly:"true" doc:"Until when the secret the last rotation replaced still signs beside the current one. Absent when only one secret signs."`
	CreatedAt           time.Time  `json:"created_at" readOnly:"true"`
	UpdatedAt           time.Time  `json:"updated_at" readOnly:"true" doc:"When the secret was last rotated, or the destination made."`
}

func (*ConnectorEventDestination) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A URL of the app's own that a connector's raw provider events are forwarded to, " +
		"such as Slack's block_actions or reaction_added. Each forward is a POST of the provider's body as " +
		"it came, with the provider's own Content-Type, signature and timestamp headers, signed on top in " +
		"the Standard Webhooks shape (webhook-id, webhook-timestamp, webhook-signature) with the " +
		"destination's own secret. A 2xx answer is taken; a 5xx, a 429 or no answer is sent again after " +
		"5 s, 5 min, 30 min and 2 h; any other answer is not sent again."
	return schema
}

// ConnectorEventForward is which of a connector's deliveries a destination is sent
// (store.ForwardUnhandled, store.ForwardAll).
type ConnectorEventForward string

func (ConnectorEventForward) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ConnectorEventForward",
		"Which deliveries a destination is sent. unhandled: the ones the router acts on in no way, such as "+
			"a Slack button click, a reaction or a modal submission, and a message no agent of the app "+
			"answers: the app's own code next to the router's agent. all: every verified delivery, "+
			"messages and grant events included, but the provider's URL handshake: the app runs its own "+
			"agent. Either way a message an agent of the app answers is still answered there.",
		store.ForwardUnhandled, store.ForwardAll)
}

// ConnectorEventDestinationRequest is a destination to create.
//
// The URL's 2048 is the bound CustomConnectorRequest.Endpoint and ConnectorOAuthClientRequest
// already use here, not a measured one.
type ConnectorEventDestinationRequest struct {
	URL     string                `json:"url" minLength:"1" maxLength:"2048" doc:"A public https URL. One that is or resolves to a private, loopback or link-local address is refused."`
	Forward ConnectorEventForward `json:"forward"`
}

func (*ConnectorEventDestinationRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "An event destination to create. An unknown field is refused rather than ignored."
	schema.AdditionalProperties = false
	return schema
}

// ConnectorEventDestinationSecret is a destination with the signing secret just made for it.
type ConnectorEventDestinationSecret struct {
	Destination ConnectorEventDestination `json:"destination"`
	Secret      string                    `json:"secret" doc:"The Standard Webhooks signing secret, whsec_ and 32 random bytes in base64. Shown this once: keep it, no later response carries it."`
}

func (*ConnectorEventDestinationSecret) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "An event destination and the secret its forwards are signed with, which no other " +
		"response carries."
	return schema
}

// ConnectorEventDestinationPage is a page of a connector's event destinations.
type ConnectorEventDestinationPage struct {
	Items      []ConnectorEventDestination `json:"items"`
	HasMore    bool                        `json:"has_more"`
	NextCursor *string                     `json:"next_cursor,omitempty" doc:"Pass as cursor for the next page. Absent on the last one."`
}

type createEventDestinationRequest struct {
	ID   string `path:"id" doc:"The connector, such as slack_bot."`
	Body ConnectorEventDestinationRequest
}

type listEventDestinationsRequest struct {
	ID string `path:"id" doc:"The connector, such as slack_bot."`
	// 200 and 25 are store.EventDestinationLimit's.
	Limit  int    `query:"limit" minimum:"1" maximum:"200" doc:"Up to 200. Omitted is 25."`
	Cursor string `query:"cursor" doc:"The next_cursor of the previous page. Omitted is the first page."`
}

type eventDestinationRequest struct {
	ID            string `path:"id" doc:"The connector, such as slack_bot."`
	DestinationID string `path:"destination_id" doc:"The destination, as returned when it was created."`
}

type eventDestinationSecretResponse struct {
	Body ConnectorEventDestinationSecret
}

type listEventDestinationsResponse struct {
	Body ConnectorEventDestinationPage
}

// registerEventDestinations declares the operations on a connector's event destinations. All
// four are server-side only: a destination's secret is the app's backend's to hold, as every
// connector operation is (registerConnectors).
func (s *Server) registerEventDestinations(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID:   "createConnectorEventDestination",
		Method:        http.MethodPost,
		Path:          "/v1/agents/connectors/{id}/event-destinations",
		Summary:       "Forward a connector's provider events to a URL",
		DefaultStatus: http.StatusCreated,
		Description: "Adds a URL the connector's raw provider events are forwarded to, for the " +
			"deliveries of the app's own provider app, such as its Slack app. A connector takes " +
			fmt.Sprint(store.MaxEventDestinations) + " destinations at most. The response carries " +
			"the destination's signing secret, once.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"201": {Description: "The destination and its secret"}},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
			http.StatusNotFound, http.StatusConflict},
	}, s.createEventDestination)
	huma.Register(api, huma.Operation{
		OperationID: "listConnectorEventDestinations",
		Method:      http.MethodGet,
		Path:        "/v1/agents/connectors/{id}/event-destinations",
		Summary:     "List a connector's event destinations",
		Description: "The connector's event destinations, newest first, without their secrets.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "A page of destinations"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listEventDestinations)
	huma.Register(api, huma.Operation{
		OperationID:   "deleteConnectorEventDestination",
		Method:        http.MethodDelete,
		Path:          "/v1/agents/connectors/{id}/event-destinations/{destination_id}",
		Summary:       "Stop forwarding to an event destination",
		DefaultStatus: http.StatusNoContent,
		Description: "Removes the destination, and every forward to it not yet sent.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"204": {Description: "The destination is removed"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.deleteEventDestination)
	huma.Register(api, huma.Operation{
		OperationID: "rotateConnectorEventDestinationSecret",
		Method:      http.MethodPost,
		Path:        "/v1/agents/connectors/{id}/event-destinations/{destination_id}/rotate-secret",
		Summary:     "Rotate an event destination's signing secret",
		Description: "Makes a new signing secret for the destination and returns it, once. For the " +
			"next 24 hours every forward is signed with both the new and the old secret, space-" +
			"separated in webhook-signature, so the receiver can move to the new one without a " +
			"forward failing its check. A rotation during another drops the oldest secret.\n\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{"200": {Description: "The destination and its new secret"}},
		Errors:    []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.rotateEventDestinationSecret)
}

// createEventDestination adds a destination for a connector that takes provider events.
func (s *Server) createEventDestination(ctx context.Context, request *createEventDestinationRequest) (*eventDestinationSecretResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil || s.eventForwarder == nil {
		return nil, errNoEventForwarding
	}
	definition, err := s.store.LatestConnectorDefinition(ctx, customerID, request.ID)
	if errors.Is(err, store.ErrNoConnectorDefinition) {
		return nil, notFound("no such connector")
	}
	if err != nil {
		return nil, err
	}
	if definition.Manifest.Channel == nil {
		return nil, invalidRequest(fmt.Sprintf("%s takes no provider events: its manifest has no channel block", definition.ID))
	}
	sent := request.Body
	if err := s.eventForwarder.CheckURL(ctx, sent.URL); err != nil {
		return nil, invalidRequest(err.Error())
	}
	destination := store.EventDestination{
		ID: uuid.NewString(), CustomerID: customerID, ConnectorID: definition.ID, URL: sent.URL, Forward: string(sent.Forward),
	}
	secret, err := s.eventForwarder.NewSecret(customerID, destination.ConnectorID, destination.ID)
	if err != nil {
		return nil, err
	}
	destination.SecretSealed, destination.KEKVersion = secret.Sealed, secret.Version
	err = s.store.CreateEventDestination(ctx, &destination)
	// 409: the request conflicts with the state of the resource (RFC 9110 section 15.5.10).
	if errors.Is(err, store.ErrEventDestinationsFull) {
		return nil, conflict(fmt.Sprintf("%s already has %d event destinations, the most it takes: delete one first",
			destination.ConnectorID, store.MaxEventDestinations))
	}
	if err != nil {
		return nil, err
	}
	return &eventDestinationSecretResponse{Body: ConnectorEventDestinationSecret{
		Destination: eventDestinationOf(destination), Secret: secret.Plain,
	}}, nil
}

// listEventDestinations lists a connector's destinations, a page at a time.
func (s *Server) listEventDestinations(ctx context.Context, request *listEventDestinationsRequest) (*listEventDestinationsResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil || s.eventForwarder == nil {
		return nil, errNoEventForwarding
	}
	cursor, err := decodeCursor[store.EventDestinationPosition](&request.Cursor)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}
	found, err := s.store.EventDestinations(ctx, customerID, request.ID, request.Limit, cursor)
	if err != nil {
		return nil, err
	}
	kept, more := page(found, store.EventDestinationLimit(request.Limit))
	listed := ConnectorEventDestinationPage{Items: make([]ConnectorEventDestination, 0, len(kept)), HasMore: more}
	for _, destination := range kept {
		listed.Items = append(listed.Items, eventDestinationOf(destination))
	}
	if more {
		last := kept[len(kept)-1]
		listed.NextCursor = encodeCursor(store.EventDestinationPosition{CreatedAt: last.CreatedAt, ID: last.ID})
	}
	return &listEventDestinationsResponse{Body: listed}, nil
}

// deleteEventDestination removes one of a connector's destinations.
func (s *Server) deleteEventDestination(ctx context.Context, request *eventDestinationRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil || s.eventForwarder == nil {
		return nil, errNoEventForwarding
	}
	err := s.store.DeleteEventDestination(ctx, customerID, request.ID, request.DestinationID)
	if errors.Is(err, store.ErrNoEventDestination) {
		return nil, notFound(noEventDestination)
	}
	return nil, err
}

// rotateEventDestinationSecret gives a destination a new secret, the old one signing beside it
// for eventforward.RotationOverlap.
func (s *Server) rotateEventDestinationSecret(ctx context.Context, request *eventDestinationRequest) (*eventDestinationSecretResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, errMissingCustomer
	}
	if s.store == nil || s.eventForwarder == nil {
		return nil, errNoEventForwarding
	}
	secret, err := s.eventForwarder.NewSecret(customerID, request.ID, request.DestinationID)
	if err != nil {
		return nil, err
	}
	destination, err := s.store.RotateEventDestinationSecret(ctx, customerID, request.ID, request.DestinationID,
		secret.Sealed, secret.Version, eventforward.RotationOverlap(time.Now()))
	if errors.Is(err, store.ErrNoEventDestination) {
		return nil, notFound(noEventDestination)
	}
	if err != nil {
		return nil, err
	}
	return &eventDestinationSecretResponse{Body: ConnectorEventDestinationSecret{
		Destination: eventDestinationOf(destination), Secret: secret.Plain,
	}}, nil
}

func eventDestinationOf(destination store.EventDestination) ConnectorEventDestination {
	shown := ConnectorEventDestination{
		ID:          destination.ID,
		ConnectorID: destination.ConnectorID,
		URL:         destination.URL,
		Forward:     ConnectorEventForward(destination.Forward),
		CreatedAt:   destination.CreatedAt,
		UpdatedAt:   destination.UpdatedAt,
	}
	if destination.PreviousUntil != nil && destination.PreviousUntil.After(time.Now()) {
		shown.PreviousSecretUntil = destination.PreviousUntil
	}
	return shown
}
