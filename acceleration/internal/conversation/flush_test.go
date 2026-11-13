package conversation

import (
	"context"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
)

// FlushSuite writes a conversation's queued writes to Chat in memory while something else
// takes writes off the queue.
type FlushSuite struct {
	suite.Suite
	chat    *chattest.Server
	service *Service
	conv    *Conversation
	channel string
}

func TestFlushSuite(t *testing.T) {
	suite.Run(t, new(FlushSuite))
}

func (s *FlushSuite) SetupTest() {
	s.chat = chattest.NewServer(s.T())
	s.service = NewForChat(s.chat.Client)
	s.T().Cleanup(s.service.Close)
	s.channel = "flush-" + uuid.NewString()
	conv, _, _, err := s.service.CreateInApp(context.Background(), 0, "customer", "agent", "agent:"+s.channel, "", "", nil, "")
	s.Require().NoError(err)
	s.conv = conv
}

// A write dropped while the one before it is being sent is taken off once that send is
// answered, and every other write is still sent, each once.
func (s *FlushSuite) TestDroppingTheWriteBeingSentNeverDropsTheOneAfterIt() {
	waiting, release := s.chat.Hold("/message")
	receipt, err := s.conv.BeginCommand("first", "hello", "")
	s.Require().NoError(err)
	<-waiting

	dropped := make(chan struct{})
	go func() {
		s.conv.dropPending(map[string]bool{receipt.UserMessageID: true})
		close(dropped)
	}()
	select {
	case <-dropped:
		s.Fail("the drop did not wait for the write being sent")
	case <-time.After(100 * time.Millisecond):
	}
	release()
	<-dropped

	s.Eventually(func() bool { return len(s.service.pending("customer", "agent:"+s.channel)) == 0 }, 5*time.Second, 10*time.Millisecond)
	s.Equal([]string{receipt.UserMessageID, receipt.AssistantMessageID}, s.chat.Visible(s.channel))
}
