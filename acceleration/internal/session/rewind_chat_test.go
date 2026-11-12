package session

import (
	"net/http"

	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// keptInChat prepares a manager whose text sessions keep their conversation in a Chat
// served from memory, recording turns where the test can read them back.
func (s *SessionSuite) keptInChat() (*chattest.Server, *heldRecorder) {
	chat := chattest.NewServer(s.T())
	service := persistent.NewForChat(chat.Client)
	s.T().Cleanup(service.Close)
	s.conversations = service
	s.gated = newGatedLLM()
	s.manages()
	held := &heldRecorder{}
	s.manager.records = held
	return chat, held
}

// answered runs one command to completion and returns the response it was recorded as and
// the Chat messages it wrote: the person's, then the agent's.
func (s *SessionSuite) answered(running *Session, commandID, text string) (string, persistent.CommandReceipt) {
	receipt, responseID, err := running.RespondCommand(s.ctx, commandID, text, "", nil, options.LLM{})
	s.Require().NoError(err)
	s.Require().NotEmpty(responseID)
	s.gated.answers(1)
	s.eventually(func() bool {
		done, err := running.Command(commandID)
		return err == nil && done.State == "completed"
	}, "the command should finish")
	s.Require().NoError(running.FlushRecords(s.ctx))
	return responseID, receipt
}

// rewindable is what the store would read back for the turns held recorded.
func rewindable(held *heldRecorder, responses ...string) *recordedTurns {
	held.mu.Lock()
	defer held.mu.Unlock()
	recorded := &recordedTurns{chatMessages: map[string]string{}}
	for _, id := range responses {
		recorded.exchanges = append(recorded.exchanges, store.Exchange{ResponseID: id, Said: id, Answer: "Hello."})
		recorded.chatMessages[id] = held.answered[id]
	}
	return recorded
}

func (s *SessionSuite) TestAPersistentSessionRecordsTheChatMessageOfEachAnswer() {
	_, held := s.keptInChat()
	running := s.commands()

	first, receipt := s.answered(running, "command-a", "First question")

	held.mu.Lock()
	defer held.mu.Unlock()
	s.Equal(receipt.AssistantMessageID, held.answered[first], "a rewind to the response rewinds Chat to its answer")
}

func (s *SessionSuite) TestAPersistentSessionRewindsItsChannelToTheKeptResponse() {
	chat, held := s.keptInChat()
	running := s.commands()
	first, kept := s.answered(running, "command-a", "First question")
	_, gone := s.answered(running, "command-b", "Second question")
	second := held.responses[1].ID

	err := running.Rewind(s.ctx, rewindable(held, first, second), first)

	s.Require().NoError(err)
	s.True(chat.Rewound(gone.UserMessageID), "the later question is hidden in Chat")
	s.True(chat.Rewound(gone.AssistantMessageID), "the later answer is hidden in Chat")
	s.False(chat.Rewound(kept.UserMessageID))
	s.False(chat.Rewound(kept.AssistantMessageID), "the kept response itself stays")
	var ids []string
	for _, message := range s.stored(running) {
		ids = append(ids, message.ID)
	}
	s.Equal([]string{kept.UserMessageID, kept.AssistantMessageID}, ids,
		"reading the conversation back, as reopening it does, does not bring the rewound turn back")
}

func (s *SessionSuite) TestARewoundCommandIdStartsANewTurnRatherThanReplayingAHiddenReply() {
	_, held := s.keptInChat()
	running := s.commands()
	first, _ := s.answered(running, "command-a", "First question")
	s.answered(running, "command-b", "Second question")
	second := held.responses[1].ID
	s.Require().NoError(running.Rewind(s.ctx, rewindable(held, first, second), first))

	retry, _, err := running.RespondCommand(s.ctx, "command-b", "Second question", "", nil, options.LLM{})

	s.Require().NoError(err)
	s.False(retry.Duplicate, "the reply the id carried is hidden, so the retry is asked again")
	s.gated.answers(1)
}

func (s *SessionSuite) TestRewindingToTheNewestResponseHidesNothing() {
	chat, held := s.keptInChat()
	running := s.commands()
	first, kept := s.answered(running, "command-a", "First question")

	err := running.Rewind(s.ctx, rewindable(held, first), first)

	s.Require().NoError(err)
	s.False(chat.Rewound(kept.AssistantMessageID))
	for _, request := range chat.Requests(chat.Key) {
		s.NotContains(request.Path, "/rewind", "there is nothing after the newest answer to hide")
	}
}

func (s *SessionSuite) TestARewindChatHasTurnedOffIsRefusedAsSuch() {
	chat, held := s.keptInChat()
	running := s.commands()
	first, _ := s.answered(running, "command-a", "First question")
	s.answered(running, "command-b", "Second question")
	second := held.responses[1].ID
	chat.FailWith(http.MethodPost, "/rewind", http.StatusForbidden)

	err := running.Rewind(s.ctx, rewindable(held, first, second), first)

	s.ErrorIs(err, persistent.ErrRewindOff)
}

func (s *SessionSuite) TestAResponseWithNoChatMessageCannotBeRewoundTo() {
	_, held := s.keptInChat()
	running := s.commands()
	first, _ := s.answered(running, "command-a", "First question")
	s.answered(running, "command-b", "Second question")
	second := held.responses[1].ID
	recorded := rewindable(held, first, second)
	recorded.chatMessages[first] = ""

	err := running.Rewind(s.ctx, recorded, first)

	s.ErrorIs(err, ErrCannotRewind, "a response recorded before its Chat message was kept has nothing to go back to")
}
