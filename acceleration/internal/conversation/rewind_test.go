package conversation

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
)

// RewindSuite rewinds an agent channel of Chat in memory and reads back what a rewind
// would hide.
type RewindSuite struct {
	suite.Suite
	chat     *chattest.Server
	rewinder Rewinder
	ctx      context.Context
	// channel is the agent channel's id; clock is when the next message is written.
	channel string
	clock   time.Time
}

func TestRewindSuite(t *testing.T) {
	suite.Run(t, new(RewindSuite))
}

func (s *RewindSuite) SetupTest() {
	s.chat = chattest.NewServer(s.T())
	s.rewinder = NewRewinder(s.chat.Client)
	s.ctx = context.Background()
	s.channel = "rewind-" + uuid.NewString()
	s.clock = time.Date(2026, 11, 12, 9, 0, 0, 0, time.UTC)
	agent := "agent-1"
	_, err := s.chat.Client.Chat().GetOrCreateChannel(s.ctx, "agent", s.channel, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{CreatedByID: &agent}})
	s.Require().NoError(err)
}

// wrote writes messages with the given texts, a millisecond apart, and returns their ids.
func (s *RewindSuite) wrote(texts ...string) []string {
	var ids []string
	for _, text := range texts {
		s.clock = s.clock.Add(time.Millisecond)
		s.chat.At(s.clock)
		id := "m-" + uuid.NewString()
		user := "user-1"
		_, err := s.chat.Client.Chat().SendMessage(s.ctx, "agent", s.channel, &getstream.SendMessageRequest{
			Message: getstream.MessageRequest{ID: &id, Text: &text, UserID: &user,
				Custom:      map[string]any{"support_turn_id": "turn-" + text},
				Attachments: []getstream.Attachment{{Type: ptrTo("file"), Title: ptrTo(text + ".txt")}}}})
		s.Require().NoError(err)
		ids = append(ids, id)
	}
	return ids
}

func ptrTo[T any](v T) *T { return &v }

func (s *RewindSuite) rewind(messageID string, inclusive bool, expectedLast string) (RewindResult, error) {
	return s.rewinder.Rewind(s.ctx, "agent", s.channel, messageID, inclusive, expectedLast, "agent-1")
}

func (s *RewindSuite) TestARewindHidesEveryMessageAfterItsPoint() {
	ids := s.wrote("one", "two", "three", "four")

	got, err := s.rewind(ids[1], false, ids[3])

	s.Require().NoError(err)
	s.NotEmpty(got.ID)
	s.Equal(ids[1], got.MessageID)
	s.False(got.Inclusive)
	s.Equal(2, got.HiddenCountAtRewind)
	s.Equal(ids[:2], s.chat.Visible(s.channel))
}

func (s *RewindSuite) TestAnInclusiveRewindHidesItsPointToo() {
	ids := s.wrote("one", "two", "three")

	got, err := s.rewind(ids[1], true, ids[2])

	s.Require().NoError(err)
	s.True(got.Inclusive)
	s.Equal(2, got.HiddenCountAtRewind)
	s.Equal(ids[:1], s.chat.Visible(s.channel))
}

func (s *RewindSuite) TestAStaleLastMessageIsAConflictNamingTheActualOne() {
	ids := s.wrote("one", "two", "three")

	_, err := s.rewind(ids[0], false, ids[1])

	s.Require().ErrorIs(err, ErrRewindConflict)
	conflict, ok := errors.AsType[*RewindConflictError](err)
	s.Require().True(ok)
	s.Equal(ids[2], conflict.LastMessageID)
	chat, ok := errors.AsType[*getstream.StreamError](err)
	s.Require().True(ok, "Chat's own error stays in the chain")
	s.Equal(http.StatusConflict, chat.StatusCode)
	s.Equal(ids, s.chat.Visible(s.channel), "nothing is hidden")
}

func (s *RewindSuite) TestRewindingToTheNewestMessageAndNotItIsNothingToRewind() {
	ids := s.wrote("one", "two")

	_, err := s.rewind(ids[1], false, ids[1])

	s.Require().ErrorIs(err, ErrNothingToRewind)
	s.NotErrorIs(err, ErrTooManyRewinds, "the two share Chat's status and code")
}

func (s *RewindSuite) TestARewindPointSharingItsCreatedAtIsAmbiguous() {
	first := s.wrote("one")[0]
	twin := s.wroteAtTheSameInstant("one's twin")
	last := s.wrote("three")[0]

	_, err := s.rewind(first, false, last)

	s.Require().ErrorIs(err, ErrRewindAmbiguous)
	_, err = s.rewind(twin, false, last)
	s.Require().ErrorIs(err, ErrRewindAmbiguous)
}

// wroteAtTheSameInstant writes one message at the clock as it is, without advancing it.
func (s *RewindSuite) wroteAtTheSameInstant(text string) string {
	id := "m-" + uuid.NewString()
	user := "user-1"
	_, err := s.chat.Client.Chat().SendMessage(s.ctx, "agent", s.channel, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{ID: &id, Text: &text, UserID: &user}})
	s.Require().NoError(err)
	return id
}

func (s *RewindSuite) TestAForkStillBeingCopiedCannotBeRewound() {
	ids := s.wrote("one", "two")
	done := s.chat.ForkInProgress(s.channel)

	_, err := s.rewind(ids[0], false, ids[1])

	s.Require().ErrorIs(err, ErrForkInProgress)
	done()
	_, err = s.rewind(ids[0], false, ids[1])
	s.Require().NoError(err)
}

func (s *RewindSuite) TestARewindAddingARangePastTheCapIsTooManyRewinds() {
	s.chat.CapRewinds(1)
	ids := s.wrote("one", "two")
	_, err := s.rewind(ids[0], false, ids[1])
	s.Require().NoError(err)
	later := s.wrote("three", "four")

	_, err = s.rewind(later[0], false, later[1])

	s.Require().ErrorIs(err, ErrTooManyRewinds)
	tooMany, ok := errors.AsType[*TooManyRewindsError](err)
	s.Require().True(ok)
	s.Equal(1, tooMany.Limit)
	s.Len(s.chat.HiddenRanges(s.channel), 1)
}

// Rewinding to the same message again, as a regenerate does, swallows the range before it,
// so it is never refused by the cap.
func (s *RewindSuite) TestRepeatingARewindKeepsOneRange() {
	s.chat.CapRewinds(1)
	ids := s.wrote("one", "two")
	_, err := s.rewind(ids[0], false, ids[1])
	s.Require().NoError(err)
	again := s.wrote("two again")

	_, err = s.rewind(ids[0], false, again[0])

	s.Require().NoError(err)
	s.Len(s.chat.HiddenRanges(s.channel), 1)
	s.Equal(ids[:1], s.chat.Visible(s.channel))
}

func (s *RewindSuite) TestAnAppWithoutTheFeatureCannotRewind() {
	ids := s.wrote("one", "two")
	s.chat.DisableRewind()

	_, err := s.rewind(ids[0], false, ids[1])

	s.Require().ErrorIs(err, ErrRewindUnavailable)
}

func (s *RewindSuite) TestARefusalWithNoKnownReasonIsChatsOwnError() {
	ids := s.wrote("one")

	_, err := s.rewind("not-a-message", false, ids[0])

	s.Require().Error(err)
	for _, known := range []error{ErrRewindConflict, ErrNothingToRewind, ErrRewindAmbiguous, ErrForkInProgress, ErrTooManyRewinds, ErrRewindUnavailable} {
		s.NotErrorIs(err, known)
	}
	chat, ok := errors.AsType[*getstream.StreamError](err)
	s.Require().True(ok)
	s.Equal(http.StatusNotFound, chat.StatusCode)
}

// A message sent at an instant a rewind hid is moved past the hole, as Chat moves it.
func (s *RewindSuite) TestASendAfterARewindIsNotHidden() {
	ids := s.wrote("one", "two")
	_, err := s.rewind(ids[0], false, ids[1])
	s.Require().NoError(err)

	after := s.wroteAtTheSameInstant("three")

	s.Equal([]string{ids[0], after}, s.chat.Visible(s.channel))
}

func (s *RewindSuite) TestAHiddenMessageCannotBeReadById() {
	ids := s.wrote("one", "two")
	_, err := s.rewind(ids[0], false, ids[1])
	s.Require().NoError(err)

	_, err = s.chat.Client.Chat().GetMessage(s.ctx, ids[1], &getstream.GetMessageRequest{})

	chat, ok := errors.AsType[*getstream.StreamError](err)
	s.Require().True(ok)
	s.Equal(http.StatusNotFound, chat.StatusCode)
}

func (s *RewindSuite) TestReadingFromAMessagePagesToTheEndOfTheChannel() {
	texts := make([]string, 250)
	for i := range texts {
		texts[i] = fmt.Sprintf("line %d", i)
	}
	ids := s.wrote(texts...)
	before := len(s.queries())

	tail, err := s.rewinder.ReadFrom(s.ctx, "agent", s.channel, ids[10], true)

	s.Require().NoError(err)
	s.Require().Len(tail.Messages, 240)
	s.Equal(ids[10], tail.Messages[0].ID)
	s.Equal(ids[249], tail.LastID)
	var read []string
	for _, m := range tail.Messages {
		read = append(read, m.ID)
	}
	s.Equal(ids[10:], read)
	s.Equal("line 10", tail.Messages[0].Text)
	s.Equal("turn-line 10", tail.Messages[0].Custom["support_turn_id"])
	s.Require().Len(tail.Messages[0].Attachments, 1)
	s.Equal("line 10.txt", *tail.Messages[0].Attachments[0].Title)
	s.Len(s.queries()[before:], 3, "100 messages a page")
}

func (s *RewindSuite) queries() []chattest.Request {
	var queries []chattest.Request
	for _, r := range s.chat.Requests(s.chat.Key) {
		if strings.HasSuffix(r.Path, "/query") {
			queries = append(queries, r)
		}
	}
	return queries
}

func (s *RewindSuite) TestReadingFromAMessageNotInclusiveLeavesItOut() {
	ids := s.wrote("one", "two", "three")

	tail, err := s.rewinder.ReadFrom(s.ctx, "agent", s.channel, ids[0], false)

	s.Require().NoError(err)
	s.Require().Len(tail.Messages, 2)
	s.Equal(ids[1], tail.Messages[0].ID)
	s.Equal(ids[2], tail.LastID)
}

func (s *RewindSuite) TestReadingFromTheNewestMessageNotInclusiveReadsNothing() {
	ids := s.wrote("one", "two")

	tail, err := s.rewinder.ReadFrom(s.ctx, "agent", s.channel, ids[1], false)

	s.Require().NoError(err)
	s.Empty(tail.Messages)
	s.Empty(tail.LastID)
}

func (s *RewindSuite) TestReadingLeavesOutWhatARewindHid() {
	ids := s.wrote("one", "two", "three")
	_, err := s.rewind(ids[1], true, ids[2])
	s.Require().NoError(err)
	after := s.wrote("four", "five")

	tail, err := s.rewinder.ReadFrom(s.ctx, "agent", s.channel, ids[0], true)

	s.Require().NoError(err)
	var read []string
	for _, m := range tail.Messages {
		read = append(read, m.ID)
	}
	s.Equal([]string{ids[0], after[0], after[1]}, read)
	s.Equal(after[1], tail.LastID)
}

// Chat counts a soft-deleted message as the channel's newest, so the last id a rewind
// expects is that message, though there is nothing of it to bring back.
func (s *RewindSuite) TestASoftDeletedLastMessageIsTheLastIDButNotRead() {
	ids := s.wrote("one", "two", "three")
	_, err := s.chat.Client.Chat().DeleteMessage(s.ctx, ids[2], &getstream.DeleteMessageRequest{})
	s.Require().NoError(err)

	tail, err := s.rewinder.ReadFrom(s.ctx, "agent", s.channel, ids[0], false)

	s.Require().NoError(err)
	s.Require().Len(tail.Messages, 1)
	s.Equal(ids[1], tail.Messages[0].ID)
	s.Equal(ids[2], tail.LastID)
	_, err = s.rewind(ids[0], false, tail.LastID)
	s.Require().NoError(err)
}

// A conversation rewinds in the app it is kept in, with the client it writes with.
func (s *RewindSuite) TestAConversationRewindsThroughItsOwnChat() {
	service := NewForChat(s.chat.Client)
	s.T().Cleanup(service.Close)
	c, _, _, err := service.Open(s.ctx, "customer", "agent-1", "")
	s.Require().NoError(err)
	defer c.Release()
	_, s.channel = channelOf(c.CID())
	ids := s.wrote("one", "two")

	rewinder, err := c.rewinder(s.ctx)
	s.Require().NoError(err)
	_, err = rewinder.Rewind(s.ctx, "agent", s.channel, ids[0], false, ids[1], "agent-1")

	s.Require().NoError(err)
	s.Equal(ids[:1], s.chat.Visible(s.channel))
}
