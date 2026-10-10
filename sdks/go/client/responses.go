package client

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strings"

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
		for cursor := ""; ; {
			read, err := i.List(ctx, page, cursor)
			if err != nil {
				stream.failed = err
				return
			}
			for _, item := range read.Items {
				select {
				case items <- item:
				case <-ctx.Done():
					stream.failed = ctx.Err()
					return
				}
			}
			if !read.HasMore || read.NextCursor == nil {
				return
			}
			cursor = *read.NextCursor
		}
	}()
	return stream
}

// List is one page of items, for a caller doing its own paging. A limit of zero leaves the
// router's own; an empty cursor is the first page, and the page's NextCursor the next.
func (i *Items) List(ctx context.Context, limit int, cursor string) (*acceleration.AgentResponseItemPage, error) {
	api, err := i.client.api()
	if err != nil {
		return nil, err
	}

	params := acceleration.ListResponseItemsParams{
		ResponseId: pointer(i.responseID),
		Limit:      pointer(limit),
		Cursor:     pointer(cursor),
	}
	listed, err := api.ListResponseItemsWithResponse(ctx, i.sessionID, &params)
	if err != nil {
		return nil, fmt.Errorf("client: reading the items of %s: %w", i.sessionID, err)
	}
	if listed.JSON200 == nil {
		return nil, failure("reading the items of "+i.sessionID, listed.HTTPResponse, listed.Body)
	}
	return listed.JSON200, nil
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

// Input is something sent along with what is asked: an Image or a Clip to show the agent, or
// the RequestID it answers.
type Input interface {
	attachTo(request *acceleration.CreateResponseRequest)
}

// Image is a picture to show the agent, by HTTP(S) URL or as a base64 data URI.
type Image struct {
	URL string
	// Detail is how closely the model looks. Empty lets the model decide.
	Detail acceleration.ImageSourceDetail
}

func (i Image) attachTo(request *acceleration.CreateResponseRequest) {
	source := acceleration.ImageSource{Url: i.URL}
	if i.Detail != "" {
		source.Detail = &i.Detail
	}
	request.Images = appended(request.Images, source)
}

// Clip is a recorded video to show the agent, by public HTTPS URL or as a base64 data URI,
// at most 50 MB either way.
//
// No model takes a video whole, so the router samples it into frames spread evenly across
// the clip and shows the agent those, each with the moment it was taken.
type Clip struct {
	URL string
	// MaxFrames is how many frames to sample, up to 32. Zero is the router's default of 8.
	MaxFrames int
}

func (v Clip) attachTo(request *acceleration.CreateResponseRequest) {
	source := acceleration.VideoSource{Url: v.URL}
	if v.MaxFrames > 0 {
		source.MaxFrames = &v.MaxFrames
	}
	request.Videos = appended(request.Videos, source)
}

// RequestID is the request a dispatch worker was handed, passed back with the text it was
// sent as so the model's answer lands on it. See stream.InboundMessage.RequestID. Every other
// question is given one of its own.
type RequestID string

func (id RequestID) attachTo(request *acceleration.CreateResponseRequest) {
	if id != "" {
		request.RequestId = pointer(string(id))
	}
}

// ClipFile is a video on disk, sent inline so the router needs no way to reach it.
func ClipFile(path string) (Clip, error) {
	clip, err := os.ReadFile(path)
	if err != nil {
		return Clip{}, fmt.Errorf("client: reading %s: %w", path, err)
	}
	kind := mime.TypeByExtension(filepath.Ext(path))
	if !strings.HasPrefix(kind, "video/") {
		kind = "video/mp4"
	}
	return Clip{URL: "data:" + kind + ";base64," + base64.StdEncoding.EncodeToString(clip)}, nil
}

func appended[T any](list *[]T, item T) *[]T {
	if list == nil {
		list = &[]T{}
	}
	*list = append(*list, item)
	return list
}

// Create asks the agent something and names the turn it answers as, showing it any images
// and videos given.
//
// It returns once the agent has started answering. An incognito session records nothing, so
// the turn it hands back has no id: there is nothing to read back afterwards, which is what
// incognito means.
func (r *Responses) Create(ctx context.Context, text string, inputs ...Input) (*AgentResponse, error) {
	api, err := r.client.api()
	if err != nil {
		return nil, err
	}

	request := acceleration.CreateResponseRequest{Text: text}
	for _, input := range inputs {
		input.attachTo(&request)
	}
	// A request id is what makes a retried question answered once. It carries text only, so a
	// question showing the agent something goes without one.
	if request.RequestId == nil && request.Images == nil && request.Videos == nil {
		id := make([]byte, 16)
		if _, err := rand.Read(id); err != nil {
			return nil, err
		}
		request.RequestId = pointer(hex.EncodeToString(id))
	}

	created, err := api.CreateResponseWithResponse(ctx, r.sessionID, request)
	if err != nil {
		return nil, fmt.Errorf("client: asking %s: %w", r.sessionID, err)
	}
	if created.JSON202 == nil {
		return nil, failure("asking "+r.sessionID, created.HTTPResponse, created.Body)
	}
	return &AgentResponse{
		Created: *created.JSON202,
		Items:   newItems(r.client, r.sessionID, created.JSON202.Id),
	}, nil
}

// List is a page of the turns so far, oldest first. A limit of zero leaves the router's own;
// an empty cursor is the first page, and the page's NextCursor the next.
func (r *Responses) List(ctx context.Context, limit int, cursor string) (*acceleration.AgentResponsePage, error) {
	api, err := r.client.api()
	if err != nil {
		return nil, err
	}

	params := acceleration.ListResponsesParams{Limit: pointer(limit), Cursor: pointer(cursor)}
	listed, err := api.ListResponsesWithResponse(ctx, r.sessionID, &params)
	if err != nil {
		return nil, fmt.Errorf("client: reading the turns of %s: %w", r.sessionID, err)
	}
	if listed.JSON200 == nil {
		return nil, failure("reading the turns of "+r.sessionID, listed.HTTPResponse, listed.Body)
	}
	return listed.JSON200, nil
}

// Rewind goes back to a response and carries on from there.
//
// The reply being spoken is abandoned and the conversation continues as though nothing after
// that response had been said: later turns are no longer listed, and the next question is
// answered from that point. The response itself is kept. Pass a response's ID, or an item's
// ResponseId to go back to the turn it was part of. A text conversation cannot be rewound,
// because its transcript lives in Chat; fork it at the response instead.
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
		return failure("rewinding "+r.sessionID, rewound.HTTPResponse, rewound.Body)
	}
	return nil
}
