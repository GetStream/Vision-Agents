package client

import (
	"context"
	"fmt"
	"net/http"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// itemPage is how many items are read per request while unwinding, and itemCeiling is as many
// as the router will hand over at once.
const (
	itemPage    = 200
	itemCeiling = 1000
)

// Items is the things one or more turns were made of, in the order they happened.
//
// Read rather than watched: this is what the backend wrote down, so it reads the same whether
// the conversation is still going or ended last week. Deltas are not here — a hundred
// fragments of one sentence are the sentence — so a caller who wants to watch words arrive
// reads Session.Events and a caller who wants the shape of a turn reads this.
type Items struct {
	client    *Client
	sessionID string
	// responseID is empty for every turn in the session, set for one turn's own items.
	responseID string
}

func newItems(client *Client, sessionID, responseID string) *Items {
	return &Items{client: client, sessionID: sessionID, responseID: responseID}
}

// ItemStream is items arriving a page at a time.
//
// The channel and the error are separate because a channel cannot carry a failure and
// closing on one would look exactly like running out of items. Read Err once the channel has
// closed, the way a scanner is read.
type ItemStream struct {
	items <-chan acceleration.AgentResponseItem
	// failed is written once, before the channel closes, so a reader that has finished
	// ranging is reading a value nothing will touch again.
	failed error
}

// Items is the items themselves, oldest first. It closes when they run out or when reading
// them failed.
func (s *ItemStream) Items() <-chan acceleration.AgentResponseItem { return s.items }

// Err is why the stream stopped early, or nil for one that simply ran out.
func (s *ItemStream) Err() error { return s.failed }

// Unwind reads every item, oldest first, fetching a page at a time.
//
// Paging is inside rather than outside because a conversation's length is not something the
// caller chose: ranging over this reads a turn or a thousand the same way.
//
//	stream := session.Responses.Items.Unwind(ctx, 0)
//	for item := range stream.Items() { ... }
//	if err := stream.Err(); err != nil { ... }
//
// A page of zero takes the default. Abandoning the range leaks nothing once ctx is done, so
// a caller that stops early should cancel it.
func (i *Items) Unwind(ctx context.Context, page int) *ItemStream {
	if page <= 0 {
		page = itemPage
	}
	page = min(page, itemCeiling)

	items := make(chan acceleration.AgentResponseItem, page)
	stream := &ItemStream{items: items}

	go func() {
		defer close(items)
		for offset := 0; ; {
			read, err := i.List(ctx, page, offset)
			if err != nil {
				stream.failed = err
				return
			}
			for _, item := range read {
				select {
				case items <- item:
				case <-ctx.Done():
					stream.failed = ctx.Err()
					return
				}
			}
			// A short page is the last page. Asking again to see an empty one would double
			// the requests for every conversation that happens to be a multiple of the page
			// size, which is not worth avoiding one extra round trip in the rare exact fit.
			if len(read) < page {
				return
			}
			offset += len(read)
		}
	}()
	return stream
}

// List is one page of items, for a caller doing its own paging. A limit or offset of zero
// leaves the router's own.
func (i *Items) List(ctx context.Context, limit, offset int) ([]acceleration.AgentResponseItem, error) {
	api, err := i.client.api()
	if err != nil {
		return nil, err
	}

	params := acceleration.ListResponseItemsParams{
		ResponseId: pointer(i.responseID),
		Limit:      pointer(limit),
		Offset:     pointer(offset),
	}
	listed, err := api.ListResponseItemsWithResponse(ctx, i.sessionID, &params)
	if err != nil {
		return nil, fmt.Errorf("client: reading the items of %s: %w", i.sessionID, err)
	}
	if listed.JSON200 == nil {
		return nil, failure("reading the items of "+i.sessionID, listed.Status(),
			listed.JSON401, listed.JSON403, listed.JSON404)
	}
	return *listed.JSON200, nil
}

// All is everything in one slice, for a conversation short enough to hold.
func (i *Items) All(ctx context.Context) ([]acceleration.AgentResponseItem, error) {
	collected := []acceleration.AgentResponseItem{}
	stream := i.Unwind(ctx, 0)
	for item := range stream.Items() {
		collected = append(collected, item)
	}
	if err := stream.Err(); err != nil {
		return nil, err
	}
	return collected, nil
}

// AgentResponse is one turn, and a way to read what it was made of.
//
// Create returns as soon as the agent has started answering rather than when it has finished,
// because a model takes seconds and a request that waited them out would time out on anything
// worth asking. So this is a handle on an answer in progress: Items reads what has been
// written down so far, and Session.Events is what watches it arrive.
type AgentResponse struct {
	// Created is what the router said when it took the question.
	Created acceleration.AgentResponse
	// Items are this turn's own, as opposed to the whole conversation's.
	Items *Items
}

// ID is the backend's id for the turn, empty for a session that records nothing.
func (r *AgentResponse) ID() string { return r.Created.Id }

// Status is where the turn has got to, as of when it was created or last read.
func (r *AgentResponse) Status() acceleration.AgentResponseStatus { return r.Created.Status }

// Responses is a session's turns.
//
// Items here is the whole conversation flattened, which is how a conversation reads and how
// it gets rendered: the question, what the agent did about it, what it said, then the next
// question. A single turn's items come off the handle Create returns.
type Responses struct {
	// Items is every turn's, oldest first.
	Items *Items

	client    *Client
	sessionID string
}

// Create asks the agent something and names the turn it answers as.
//
// It returns once the agent has started answering. An incognito session records nothing, so
// the turn it hands back has no id: there is nothing to read back afterwards, which is what
// incognito means.
func (r *Responses) Create(ctx context.Context, text string, images ...acceleration.ImageSource) (*AgentResponse, error) {
	api, err := r.client.api()
	if err != nil {
		return nil, err
	}

	request := acceleration.CreateResponseRequest{Text: text}
	if len(images) > 0 {
		request.Images = &images
	}

	created, err := api.CreateResponseWithResponse(ctx, r.sessionID, request)
	if err != nil {
		return nil, fmt.Errorf("client: asking %s: %w", r.sessionID, err)
	}
	if created.JSON202 == nil {
		return nil, failure("asking "+r.sessionID, created.Status(),
			created.JSON400, created.JSON401, created.JSON403, created.JSON404)
	}
	return &AgentResponse{
		Created: *created.JSON202,
		Items:   newItems(r.client, r.sessionID, created.JSON202.Id),
	}, nil
}

// List is the turns so far, oldest first. A limit or offset of zero leaves the router's own.
func (r *Responses) List(ctx context.Context, limit, offset int) ([]acceleration.AgentResponse, error) {
	api, err := r.client.api()
	if err != nil {
		return nil, err
	}

	params := acceleration.ListResponsesParams{Limit: pointer(limit), Offset: pointer(offset)}
	listed, err := api.ListResponsesWithResponse(ctx, r.sessionID, &params)
	if err != nil {
		return nil, fmt.Errorf("client: reading the turns of %s: %w", r.sessionID, err)
	}
	if listed.JSON200 == nil {
		return nil, failure("reading the turns of "+r.sessionID, listed.Status(),
			listed.JSON401, listed.JSON403, listed.JSON404)
	}
	return *listed.JSON200, nil
}

// Rewind goes back to a response and carries on from there.
//
// The reply being spoken is abandoned and the conversation continues as though nothing after
// that response had been said: later turns are no longer listed, and the next question is
// answered from that point. The response itself is kept. Pass a response's ID, or an item's
// ResponseId to go back to the turn it was part of. A persistent conversation cannot be
// rewound, because its transcript lives in Chat; fork it at the response instead.
func (r *Responses) Rewind(ctx context.Context, responseID string) error {
	if responseID == "" {
		return fmt.Errorf("client: rewinding %s: that response has no id, which is what a "+
			"session that records nothing hands back; there is nothing to rewind to", r.sessionID)
	}
	api, err := r.client.api()
	if err != nil {
		return err
	}

	rewound, err := api.RewindSessionWithResponse(ctx, r.sessionID,
		acceleration.RewindSessionRequest{ResponseId: responseID})
	if err != nil {
		return fmt.Errorf("client: rewinding %s: %w", r.sessionID, err)
	}
	if rewound.StatusCode() != http.StatusNoContent {
		return failure("rewinding "+r.sessionID, rewound.Status(),
			rewound.JSON400, rewound.JSON401, rewound.JSON403, rewound.JSON404)
	}
	return nil
}
