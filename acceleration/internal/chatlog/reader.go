package chatlog

import (
	"context"
	"errors"
	"slices"
	"sort"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// transcriptLimit is how much of a conversation is read back at once. Stream caps a page
// of messages, and a call long enough to exceed it is long enough that the far end of it
// is a separate question.
const transcriptLimit = 200

// Reader reads conversations back out of Stream Chat.
//
// Writing them is per call and per agent, but reading is not: one reader serves every
// transcript the service is asked for, which is why it is not a Log.
type Reader struct {
	client *getstream.Stream
}

// Spoken is one line of a conversation as it was stored.
type Spoken struct {
	// Speaker is the user the line was written as, which for the agent's own replies is
	// whoever it joined the call as.
	Speaker string
	// Name is that user's display name, when they have one.
	Name string
	// Agent is whether the agent wrote the line rather than somebody it was talking to.
	// It is what the line was stored as, so it holds however the agent was named.
	Agent bool
	Text  string
	At    time.Time
	// ID is the stored message's id. A participant's line is stored under the line id it was
	// heard as, which the session's live frames name too.
	ID string
	// Generating says the line is still being written: a reply still streaming, or words a
	// participant is still saying, stored as they first were.
	Generating bool
	// TurnID is the turn a reply was written for, when it was written for one.
	TurnID string
	// Interrupted says the caller talked over the reply. Its text is what it was going to
	// say, of which the caller heard the start.
	Interrupted bool
	// Purpose is why the agent said the line unasked, empty for an answer: a greeting, a
	// check-in on a quiet caller, or an update on work still running.
	Purpose string
}

// NewReaderFromClient reads through a client the caller built, which is how a test points
// a Reader at something other than Stream.
func NewReaderFromClient(client *getstream.Stream) *Reader { return &Reader{client: client} }

// Read names the part of a conversation to read back.
type Read struct {
	// Channel is the channel id within the agent type: the agent's own channel, or the
	// conversation a call was bound to.
	Channel string
	// Customer is who is reading. A channel stamped for another customer reads as empty.
	Customer string
	// Agent is the agent's id, which is who the conversation service writes its replies
	// as. Those lines carry no source, so this is how they are told apart.
	Agent string
	// From and To bound the lines to the call's own, since a conversation's channel also
	// holds what was typed before and after it. A zero To means the call is still going.
	From, To time.Time
}

// Transcript returns what was said in one channel, oldest first.
//
// A conversation nobody stored comes back empty rather than as an error: a call whose
// transcript was never written still happened, and the caller asking about it is not
// wrong to. Reading never creates the channel, so asking about a call leaves nothing behind.
func (r *Reader) Transcript(ctx context.Context, read Read) ([]Spoken, error) {
	if read.Channel == "" {
		return nil, stack.Wrap(errors.New("chatlog: a channel is required"))
	}

	one, limit := 1, transcriptLimit
	response, err := r.client.Chat().QueryChannels(ctx, &getstream.QueryChannelsRequest{
		FilterConditions: map[string]any{"cid": ChannelType + ":" + read.Channel},
		Limit:            &one,
		MessageLimit:     &limit,
	})
	if err != nil {
		return nil, stack.Wrap(err)
	}
	if len(response.Data.Channels) == 0 {
		return []Spoken{}, nil
	}
	found := response.Data.Channels[0]
	// A channel the router stamped says whose it is. One stamped for somebody else is not
	// this customer's to read, whatever their call row names.
	if found.Channel != nil {
		if stamped, ok := found.Channel.Custom[conversation.CustomerField]; ok && stamped != read.Customer {
			return []Spoken{}, nil
		}
	}

	said := make([]Spoken, 0, len(found.Messages))
	for _, stored := range found.Messages {
		text := conversation.WithoutTimings(stored.Text, stored.Custom)
		interrupted, _ := stored.Custom[interruptedField].(bool)
		if planned, ok := stored.Custom[interruptedTextField].(string); ok && text == "" && interrupted {
			text = planned
		}
		if text == "" || stored.DeletedAt != nil {
			continue
		}
		line := Spoken{Speaker: stored.User.ID, Text: text, ID: stored.ID, Interrupted: interrupted}
		if generating, ok := stored.Custom[generatingField].(bool); ok {
			line.Generating = generating
		}
		if turn, ok := stored.Custom[turnField].(string); ok {
			line.TurnID = turn
		}
		if purpose, ok := stored.Custom[purposeField].(string); ok && slices.Contains(purposes, purpose) {
			line.Purpose = purpose
		}
		if stored.CreatedAt.Time != nil {
			line.At = *stored.CreatedAt.Time
		}
		if !read.From.IsZero() && line.At.Before(read.From) {
			continue
		}
		if !read.To.IsZero() && line.At.After(read.To) {
			continue
		}
		source, sourced := stored.Custom[SourceField]
		line.Agent = source == SourceAgent || (!sourced && read.Agent != "" && stored.User.ID == read.Agent)
		// A user nobody named is named after their own id, which is not a name and is no
		// use to whoever is reading the conversation back.
		if stored.User.Name != nil && *stored.User.Name != stored.User.ID {
			line.Name = *stored.User.Name
		}
		said = append(said, line)
	}

	sort.SliceStable(said, func(i, j int) bool { return said[i].At.Before(said[j].At) })
	return said, nil
}
