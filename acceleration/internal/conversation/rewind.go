package conversation

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// The ways Chat refuses a rewind that a caller acts on. Each is matched with errors.Is;
// the *getstream.StreamError Chat answered with stays in the chain for its message.
var (
	// ErrRewindConflict is a channel whose newest visible message is not the expected one:
	// a message was sent, or another rewind landed, after it was read. errors.As with a
	// *RewindConflictError gives the actual newest message.
	ErrRewindConflict = errors.New("conversation: the channel changed since its last message was read")
	// ErrNothingToRewind is a rewind that is not inclusive to the newest visible message,
	// which would hide nothing.
	ErrNothingToRewind = errors.New("conversation: nothing to rewind")
	// ErrRewindAmbiguous is a target sharing its created_at with another visible message:
	// rewinds hide created_at ranges, so the two cannot be told apart.
	ErrRewindAmbiguous = errors.New("conversation: another message has the rewind point's created_at")
	// ErrForkInProgress is a rewind of a fork whose copy is not complete.
	ErrForkInProgress = errors.New("conversation: the channel is a fork still being copied")
	// ErrTooManyRewinds is a rewind that would give the channel more separate hidden ranges
	// than Chat allows. errors.As with a *TooManyRewindsError gives the limit.
	ErrTooManyRewinds = errors.New("conversation: the channel has too many rewound ranges")
	// ErrRewindUnavailable is an app without the rewind feature, or with it frozen: Chat
	// answers both the same way.
	ErrRewindUnavailable = errors.New("conversation: rewind is not enabled for this app")
)

// RewindConflictError is ErrRewindConflict with the channel's actual newest visible
// message, empty when it has none.
type RewindConflictError struct {
	LastMessageID string
	cause         error
}

func (e *RewindConflictError) Error() string {
	return fmt.Sprintf("%s: its newest visible message is %q", ErrRewindConflict, e.LastMessageID)
}
func (e *RewindConflictError) Is(target error) bool { return target == ErrRewindConflict }
func (e *RewindConflictError) Unwrap() error        { return e.cause }

// TooManyRewindsError is ErrTooManyRewinds with the most separate ranges a channel can
// have, zero when Chat did not say.
type TooManyRewindsError struct {
	Limit int
	cause error
}

func (e *TooManyRewindsError) Error() string {
	return fmt.Sprintf("%s (at most %d)", ErrTooManyRewinds, e.Limit)
}
func (e *TooManyRewindsError) Is(target error) bool { return target == ErrTooManyRewinds }
func (e *TooManyRewindsError) Unwrap() error        { return e.cause }

// rewindRefusal is a sentinel refusal that keeps Chat's error in the chain.
type rewindRefusal struct {
	sentinel, cause error
}

func (e *rewindRefusal) Error() string        { return e.sentinel.Error() + ": " + e.cause.Error() }
func (e *rewindRefusal) Is(target error) bool { return target == e.sentinel }
func (e *rewindRefusal) Unwrap() error        { return e.cause }

// RewindResult is the rewind Chat made.
type RewindResult struct {
	ID        string `json:"id"`
	MessageID string `json:"message_id"`
	Inclusive bool   `json:"inclusive"`
	// HiddenCountAtRewind is how many messages the rewind hid that were not hidden already.
	HiddenCountAtRewind int `json:"hidden_count_at_rewind"`
}

// Tail is what a rewind to a message would hide: the channel's visible messages from it to
// the end, oldest first, whole as Chat holds them.
type Tail struct {
	// Messages leaves out soft-deleted messages, which are nothing to bring back.
	Messages []getstream.MessageResponse
	// LastID is the channel's newest visible message as Chat counts it, a soft-deleted one
	// included: the expected_last_message_id of a rewind made from this read. It is empty
	// when nothing was read.
	LastID string
}

// Rewinder rewinds a channel in Chat and reads what a rewind would hide.
type Rewinder interface {
	// Rewind hides every message of the channel created after messageID, and messageID too
	// when inclusive. expectedLastMessageID is the newest visible message the caller read;
	// userID is who the rewind is made as.
	Rewind(ctx context.Context, channelType, channelID, messageID string, inclusive bool, expectedLastMessageID, userID string) (RewindResult, error)
	// ReadFrom reads the channel's visible messages from messageID, included when
	// inclusive, to the end.
	ReadFrom(ctx context.Context, channelType, channelID, messageID string, inclusive bool) (Tail, error)
}

// NewRewinder rewinds through a Chat client, as Chats hands one out.
func NewRewinder(client *getstream.Stream) Rewinder {
	return chatRewinder{client: client}
}

// rewinder is the Rewinder for the app this conversation is kept in, resolved like every
// write so a rotated credential is used from the next one.
func (c *Conversation) rewinder(ctx context.Context) (Rewinder, error) {
	client, err := c.chat(ctx)
	if err != nil {
		return nil, err
	}
	return NewRewinder(client), nil
}

type chatRewinder struct {
	client *getstream.Stream
}

// tailPage is the most messages Chat returns in one channel read.
const tailPage = 100

type rewindRequest struct {
	MessageID             string `json:"message_id"`
	Inclusive             bool   `json:"inclusive"`
	ExpectedLastMessageID string `json:"expected_last_message_id"`
	UserID                string `json:"user_id,omitempty"`
}

// rewindResponse decodes only the rewind. The channel beside it is left alone: a failure
// to decode a field nothing reads would turn a rewind Chat committed into an error.
type rewindResponse struct {
	Rewind RewindResult `json:"rewind"`
}

func (r chatRewinder) Rewind(ctx context.Context, channelType, channelID, messageID string, inclusive bool, expectedLastMessageID, userID string) (RewindResult, error) {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	req := rewindRequest{MessageID: messageID, Inclusive: inclusive, ExpectedLastMessageID: expectedLastMessageID, UserID: userID}
	var resp rewindResponse
	_, err := getstream.MakeRequest[rewindRequest, rewindResponse](r.client.Client, ctx, http.MethodPost,
		"/api/v2/chat/channels/{type}/{id}/rewind", nil, &req, &resp, map[string]string{"type": channelType, "id": channelID})
	if err != nil {
		return RewindResult{}, stack.Wrap(rewindError(err))
	}
	return resp.Rewind, nil
}

// rewindError names Chat's refusal by its exception_fields.reason, which Chat sets so a
// client need not hardcode codes: too_many_rewinds and nothing_to_rewind share a status and
// code. The feature being off has no reason: it is the 403 Chat gives any flag that is off,
// and a server-side call holds every permission, so no other 403 reaches it.
func rewindError(err error) error {
	refused, ok := errors.AsType[*getstream.StreamError](err)
	if !ok {
		return err
	}
	switch refused.ExceptionFields["reason"] {
	case "rewind_conflict":
		return &RewindConflictError{LastMessageID: refused.ExceptionFields["last_message_id"], cause: err}
	case "too_many_rewinds":
		limit, _ := strconv.Atoi(refused.ExceptionFields["limit"])
		return &TooManyRewindsError{Limit: limit, cause: err}
	case "nothing_to_rewind":
		return &rewindRefusal{sentinel: ErrNothingToRewind, cause: err}
	case "rewind_point_ambiguous":
		return &rewindRefusal{sentinel: ErrRewindAmbiguous, cause: err}
	case "fork_in_progress":
		return &rewindRefusal{sentinel: ErrForkInProgress, cause: err}
	case "":
		if refused.StatusCode == http.StatusForbidden && refused.Code == chatNotAllowed {
			return &rewindRefusal{sentinel: ErrRewindUnavailable, cause: err}
		}
	}
	return err
}

// chatNotAllowed is Chat's error code for an action that is not allowed, a feature flag
// that is off included.
const chatNotAllowed = 17

func (r chatRewinder) ReadFrom(ctx context.Context, channelType, channelID, messageID string, inclusive bool) (Tail, error) {
	var tail Tail
	limit := tailPage
	params := &getstream.MessagePaginationParams{Limit: &limit}
	from := messageID
	if inclusive {
		params.IDGte = &from
	} else {
		params.IDGt = &from
	}
	for {
		page, err := r.page(ctx, channelType, channelID, params)
		if err != nil {
			return Tail{}, err
		}
		for _, m := range page {
			tail.LastID = m.ID
			if m.DeletedAt == nil && m.Type != "deleted" {
				tail.Messages = append(tail.Messages, m)
			}
		}
		if len(page) < tailPage {
			return tail, nil
		}
		next := tail.LastID
		params = &getstream.MessagePaginationParams{Limit: &limit, IDGt: &next}
	}
}

// page reads one page of the channel, never creating it: no Data is sent.
func (r chatRewinder) page(ctx context.Context, channelType, channelID string, params *getstream.MessagePaginationParams) ([]getstream.MessageResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	state := true
	resp, err := r.client.Chat().GetOrCreateChannel(ctx, channelType, channelID, &getstream.GetOrCreateChannelRequest{State: &state, Messages: params})
	if err != nil {
		return nil, stack.Wrap(err)
	}
	return resp.Data.Messages, nil
}
