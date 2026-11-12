package conversation

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// ErrRewindConflict is a rewind Stream Chat refused because something was written to the
// channel after the newest message the rewind was told about. Nothing was hidden; the
// caller can rewind again.
var ErrRewindConflict = errors.New("conversation: the channel moved on while it was being rewound")

// ErrRewindOff is a rewind Stream Chat refused for the app: rewinding agent channels is
// turned off there (its kill switch is set).
var ErrRewindOff = errors.New("conversation: rewinding agent channels is turned off for this application")

// ErrRewindRefused is a rewind Stream Chat refused for the message or the channel, such as
// a message that is no longer visible.
var ErrRewindRefused = errors.New("conversation: Stream Chat refused the rewind")

// rewindConflictCode is Stream Chat's rewind_conflict.
const rewindConflictCode = 125

// rewindWait bounds how long a rewind waits for the conversation's writes to reach Stream.
const rewindWait = 10 * time.Second

// rewindRequest is POST /channels/{type}/{id}/rewind, which the SDK this module pins
// predates. message_id stays visible; everything after it is hidden.
type rewindRequest struct {
	MessageID             string `json:"message_id"`
	ExpectedLastMessageID string `json:"expected_last_message_id"`
}

// MessageForTurn is the Stream Chat message a model turn's reply is written to, once the
// turn has been seen.
func (c *Conversation) MessageForTurn(turnID string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	id, bound := c.turns[turnID]
	return id, bound && id != ""
}

// Threaded reports whether the conversation is kept in a thread channel, whose messages
// are an external thread's as well as the agent's.
func (c *Conversation) Threaded() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.thread
}

// Rewind hides every message of the channel after messageID, for everyone and every read,
// and forgets the commands those messages carried. The reply being written must be stopped
// first: its last writes are waited for here, so they land before the rewind rather than
// after it.
//
// When messageID is already the newest message there is nothing to hide, and Stream is not
// asked.
func (c *Conversation) Rewind(ctx context.Context, messageID string) error {
	if messageID == "" {
		return stack.Wrap(errors.New("conversation: a message to rewind to is required"))
	}
	if err := c.awaitWritten(ctx); err != nil {
		return err
	}
	client, err := c.chat(ctx)
	if err != nil {
		return stack.Wrap(err)
	}
	channelType, id := channelOf(c.CID())

	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	limit := 100
	state := true
	read, err := client.Chat().GetOrCreateChannel(ctx, channelType, id, &getstream.GetOrCreateChannelRequest{
		State: &state, Messages: &getstream.MessagePaginationParams{Limit: &limit},
	})
	if err != nil {
		return stack.Wrap(err)
	}
	messages := read.Data.Messages
	if len(messages) == 0 {
		return stack.Wrap(fmt.Errorf("%w: the channel has no messages", ErrRewindRefused))
	}
	newest := messages[len(messages)-1].ID
	if newest != messageID {
		var response getstream.Response
		_, err = getstream.MakeRequest(client.Client, ctx, http.MethodPost, "/api/v2/chat/channels/{type}/{id}/rewind", nil,
			&rewindRequest{MessageID: messageID, ExpectedLastMessageID: newest}, &response,
			map[string]string{"type": channelType, "id": id})
		if err != nil {
			return rewindRefusal(err)
		}
	}

	// What came after the message is hidden now, so the commands it carried are forgotten:
	// a retried request id of one of them starts a new turn rather than replaying a reply
	// nobody can see. Commands older than the page read are all before the message.
	after := map[string]bool{}
	for i := len(messages) - 1; i >= 0 && messages[i].ID != messageID; i-- {
		after[messages[i].ID] = true
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for commandID, record := range c.data.Commands {
		if after[record.UserMessageID] || after[record.AssistantMessageID] || after[commandID] {
			delete(c.data.Commands, commandID)
		}
	}
	if c.data.Current != nil && (after[c.data.Current.ID] || c.data.Current.FinishedAt != nil) {
		c.data.Current = nil
	}
	c.turns = map[string]string{}
	return nil
}

// awaitWritten waits until every write the conversation accepted is stored in Stream. A
// write stays queued until it is, so an empty queue is one nothing is in flight for.
func (c *Conversation) awaitWritten(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, rewindWait)
	defer cancel()
	tick := time.NewTicker(25 * time.Millisecond)
	defer tick.Stop()
	for {
		c.mu.Lock()
		pending, parkedUntil := len(c.data.Pending), c.parkedUntil
		c.mu.Unlock()
		if pending == 0 {
			return nil
		}
		if !parkedUntil.IsZero() {
			return stack.Wrap(errors.New("conversation: its Stream app cannot be written, so it cannot be rewound now"))
		}
		select {
		case <-ctx.Done():
			return stack.Wrap(fmt.Errorf("conversation: the reply's last writes did not reach Stream: %w", ctx.Err()))
		case <-tick.C:
		}
	}
}

// rewindRefusal turns Stream Chat's refusal of a rewind into what the caller can act on.
func rewindRefusal(err error) error {
	var refused *getstream.StreamError
	if !errors.As(err, &refused) {
		return stack.Wrap(err)
	}
	switch {
	case refused.StatusCode == http.StatusConflict && refused.Code == rewindConflictCode:
		return stack.Wrap(ErrRewindConflict)
	case refused.StatusCode == http.StatusForbidden:
		return stack.Wrap(ErrRewindOff)
	case refused.StatusCode >= 400 && refused.StatusCode < 500:
		return stack.Wrap(fmt.Errorf("%w: %s", ErrRewindRefused, refused.Message))
	}
	return stack.Wrap(err)
}
