package chatlog

import (
	"context"
	"errors"
	"os"
	"sort"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
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
	Text string
	At   time.Time
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

// Transcript returns what was said by one agent, oldest first.
//
// A conversation nobody stored comes back empty rather than as an error: a call whose
// transcript was never written still happened, and the caller asking about it is not
// wrong to.
func (r *Reader) Transcript(ctx context.Context, agentID string) ([]Spoken, error) {
	if agentID == "" {
		return nil, errors.New("chatlog: an agent id is required")
	}

	limit := transcriptLimit
	// Without asking for the state the channel comes back without its messages, which
	// reads as a conversation nobody stored rather than as the error it is.
	state := true
	// A custom channel type such as "agent" refuses server-side create without a
	// creator; "messaging" used to let this through. The channel is named for the
	// agent, so the agent is who created it.
	createdBy := agentID
	response, err := r.client.Chat().GetOrCreateChannel(ctx, ChannelType, agentID,
		&getstream.GetOrCreateChannelRequest{
			State:    &state,
			Messages: &getstream.MessagePaginationParams{Limit: &limit},
			Data:     &getstream.ChannelInput{CreatedByID: &createdBy},
		})
	if err != nil {
		return nil, err
	}

	said := make([]Spoken, 0, len(response.Data.Messages))
	for _, stored := range response.Data.Messages {
		if stored.Text == "" || stored.DeletedAt != nil {
			continue
		}
		line := Spoken{Speaker: stored.User.ID, Text: stored.Text}
		if stored.User.Name != nil {
			line.Name = *stored.User.Name
		}
		if stored.CreatedAt.Time != nil {
			line.At = *stored.CreatedAt.Time
		}
		said = append(said, line)
	}

	sort.SliceStable(said, func(i, j int) bool { return said[i].At.Before(said[j].At) })
	return said, nil
}
