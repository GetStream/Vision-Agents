package chatlog

import (
	"context"
	"errors"
	"os"
	"sort"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
)

// transcriptLimit is how much of a conversation is read back at once. Stream caps a page
// of messages, and a call long enough to exceed it is long enough that the far end of it
// is a separate question.
const transcriptLimit = 200

// ReaderOptions configures a Reader. The credentials fall back to the environment the
// same way a Log's do.
type ReaderOptions struct {
	// APIKey defaults to STREAM_API_KEY.
	APIKey string
	// APISecret defaults to STREAM_API_SECRET.
	APISecret string
}

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
}

// NewReader validates the options and returns a Reader.
func NewReader(options ReaderOptions) (*Reader, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APISecret == "" {
		options.APISecret = os.Getenv(apiSecretEnvVar)
	}
	if options.APIKey == "" || options.APISecret == "" {
		return nil, errors.New("chatlog: " + apiKeyEnvVar + " and " + apiSecretEnvVar + " are required")
	}

	client, err := getstream.NewClient(options.APIKey, options.APISecret)
	if err != nil {
		return nil, err
	}
	return &Reader{client: client}, nil
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
		return nil, errors.New("chatlog: a channel is required")
	}

	one, limit := 1, transcriptLimit
	response, err := r.client.Chat().QueryChannels(ctx, &getstream.QueryChannelsRequest{
		FilterConditions: map[string]any{"cid": ChannelType + ":" + read.Channel},
		Limit:            &one,
		MessageLimit:     &limit,
	})
	if err != nil {
		return nil, err
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
		if stored.Text == "" || stored.DeletedAt != nil {
			continue
		}
		line := Spoken{Speaker: stored.User.ID, Text: stored.Text}
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
