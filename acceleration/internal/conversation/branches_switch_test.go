package conversation_test

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// turn is the response id answering its own question after parent.
func turn(id, parent string) conversation.Turn {
	return conversation.Turn{ResponseID: id, ParentResponseID: parent, QuestionResponseID: id}
}

// twoTurns is a conversation of two answered questions.
func (s *BranchesSuite) twoTurns() (first, second conversation.CommandReceipt) {
	first = s.answered(turn("resp-1", ""), "ask-1", "Is it raining?", "No.")
	second = s.answered(turn("resp-2", "resp-1"), "ask-2", "And tomorrow?", "Rain.")
	return first, second
}

// visibleTexts are the texts of the messages a read of the channel returns, oldest first.
func (s *BranchesSuite) visibleTexts() []string {
	visible := map[string]bool{}
	for _, id := range s.chat.Visible(s.session) {
		visible[id] = true
	}
	var texts []string
	for _, stored := range s.chat.Stored(s.session) {
		if visible[stored["id"].(string)] {
			texts = append(texts, stored["text"].(string))
		}
	}
	return texts
}

// stored is message id as Chat holds it.
func (s *BranchesSuite) stored(id string) map[string]any {
	for _, stored := range s.chat.Stored(s.session) {
		if stored["id"] == id {
			return stored
		}
	}
	s.FailNow("no such message in Chat", id)
	return nil
}

// regenerate is the regenerate of second's reply, under switch id.
func regenerate(id string, second conversation.CommandReceipt) conversation.RewindTarget {
	return conversation.RewindTarget{SwitchID: id, Kind: store.SwitchRegenerate, TargetResponseID: "resp-2", MessageID: second.UserMessageID}
}

// edit is the edit of second's question, under switch id.
func edit(id string, second conversation.CommandReceipt) conversation.RewindTarget {
	return conversation.RewindTarget{SwitchID: id, Kind: store.SwitchEdit, TargetResponseID: "resp-2", MessageID: second.UserMessageID, Inclusive: true}
}

// reopened is the conversation opened by another router on the same Chat and store, as after
// a restart, with what it told the session's hooks.
func (s *BranchesSuite) reopened(chosen []store.AgentBranchMessage) (*conversation.Conversation, []llm.Message, *[]string) {
	s.written()
	s.conv.Release()
	later := conversation.NewForChat(s.chat.Client)
	s.T().Cleanup(later.Close)
	later.SetBranches(s.branches)
	var mu sync.Mutex
	switched := &[]string{}
	later.SetSwitchHooks(conversation.SwitchHooks{
		Chosen: func(context.Context, store.AgentBranchSwitch) ([]store.AgentBranchMessage, error) { return chosen, nil },
		Switched: func(_ context.Context, sw store.AgentBranchSwitch) error {
			mu.Lock()
			defer mu.Unlock()
			*switched = append(*switched, sw.ID)
			return nil
		},
	})
	conv, history, _, err := later.OpenInApp(s.ctx, 0, "customer", "agent", s.cid, "", "", nil, "")
	s.Require().NoError(err)
	return conv, history, switched
}

func (s *BranchesSuite) TestARegenerateHidesTheReplyAndKeepsACopyOfIt() {
	first, second := s.twoTurns()

	out, err := s.conv.Rewind(s.ctx, regenerate("sw-1", second))

	s.Require().NoError(err)
	s.True(out.Rewound)
	s.Equal([]string{second.AssistantMessageID}, out.Hidden)
	s.Equal([]string{first.UserMessageID, first.AssistantMessageID, second.UserMessageID}, s.chat.Visible(s.session))
	copies, err := s.branches.BranchMessages(s.ctx, "customer", s.session, []string{"resp-2"})
	s.Require().NoError(err)
	s.Require().Len(copies, 1)
	s.Equal(second.AssistantMessageID, copies[0].OriginalMessageID)
	s.Equal("sw-1", copies[0].SwitchID)
	s.Equal("Rain.", copies[0].Message["text"])
	s.Equal(store.SwitchRewound, out.Switch.State)
	s.Equal([]llm.Message{{Role: llm.User, Content: "Is it raining?"}, {Role: llm.Assistant, Content: "No."}, {Role: llm.User, Content: "And tomorrow?"}},
		out.History, "the model is handed the conversation up to the question")

	again, err := s.conv.BeginRegenerate(conversation.Turn{ResponseID: "resp-3", ParentResponseID: "resp-1", QuestionResponseID: "resp-2",
		Reply: conversation.Place{Index: 2, Count: 2}}, "again-2", second.UserMessageID)
	s.Require().NoError(err)
	s.replied("again-2", "Sun, after all.")
	s.Require().NoError(s.conv.FinishSwitch(s.ctx, "sw-1"))

	s.Equal([]string{"Is it raining?", "No.", "And tomorrow?", "Sun, after all."}, s.visibleTexts())
	s.Equal(map[string]any{"response_id": "resp-3", "parent_response_id": "resp-1", "question_response_id": "resp-2",
		"of": "reply", "index": float64(2), "count": float64(2)}, s.custom(again.AssistantMessageID)[conversation.BranchField])
	row, _ := s.branches.row("sw-1")
	s.Equal(store.SwitchDone, row.State)
}

// A message no mark names, as a customer's backend writes one into the channel or a router
// from before branches wrote it, is kept with the turn it was written during, and comes back
// with its branch.
func (s *BranchesSuite) TestAnUnmarkedMessageIsKeptWithTheTurnItWasWrittenDuring() {
	_, second := s.twoTurns()
	s.written()
	id, text, user := "backend-note", "Your order shipped.", "backend"
	_, err := s.chat.Client.Chat().SendMessage(s.ctx, "agent", s.session, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{ID: &id, Text: &text, UserID: &user}})
	s.Require().NoError(err)

	_, err = s.conv.Rewind(s.ctx, regenerate("sw-1", second))

	s.Require().NoError(err)
	copies, err := s.branches.BranchMessages(s.ctx, "customer", s.session, []string{"resp-2"})
	s.Require().NoError(err)
	var kept []string
	for _, copied := range copies {
		kept = append(kept, copied.OriginalMessageID)
	}
	s.ElementsMatch([]string{second.AssistantMessageID, id}, kept)
}

func (s *BranchesSuite) TestAnEditHidesTheQuestionAndItsReply() {
	first, second := s.twoTurns()

	out, err := s.conv.Rewind(s.ctx, edit("sw-1", second))
	s.Require().NoError(err)
	s.answered(conversation.Turn{ResponseID: "resp-3", ParentResponseID: "resp-1", QuestionResponseID: "resp-3",
		Question: conversation.Place{Index: 2, Count: 2}}, "ask-2b", "And on Sunday?", "Sun.")

	s.Equal([]string{second.UserMessageID, second.AssistantMessageID}, out.Hidden)
	s.Equal([]llm.Message{{Role: llm.User, Content: "Is it raining?"}, {Role: llm.Assistant, Content: "No."}}, out.History)
	s.Equal([]string{"Is it raining?", "No.", "And on Sunday?", "Sun."}, s.visibleTexts())
	s.Equal([]string{second.UserMessageID, second.AssistantMessageID}, s.branches.copied(s.session))
	s.Contains(s.chat.Visible(s.session), first.AssistantMessageID)
}

// After an edit, the branch it replaced is sent again from its copies: the question under the
// request id its reply names, both carrying where they came from, and no push sent for them.
func (s *BranchesSuite) TestActivatingABranchSendsItAgainUnderNewIDs() {
	_, second := s.twoTurns()
	_, err := s.conv.Rewind(s.ctx, edit("sw-1", second))
	s.Require().NoError(err)
	edited := s.answered(turn("resp-3", "resp-1"), "ask-2b", "And on Sunday?", "Sun.")
	s.Require().NoError(s.conv.FinishSwitch(s.ctx, "sw-1"))
	chosen, err := s.branches.BranchMessages(s.ctx, "customer", s.session, []string{"resp-2"})
	s.Require().NoError(err)

	out, err := s.conv.Activate(s.ctx, conversation.RewindTarget{SwitchID: "sw-2", TargetResponseID: "resp-2",
		MessageID: edited.UserMessageID, Inclusive: true}, chosen)

	s.Require().NoError(err)
	question, reply := conversation.ResentID("sw-2", second.UserMessageID), conversation.ResentID("sw-2", second.AssistantMessageID)
	s.Equal([]string{"Is it raining?", "No.", "And tomorrow?", "Rain."}, s.visibleTexts())
	s.Equal([]string{edited.UserMessageID, edited.AssistantMessageID}, out.Hidden)
	s.Equal([]string{question, reply}, s.chat.Visible(s.session)[2:])

	asked := s.stored(question)
	s.Equal(true, asked["skip_push"])
	askedCustom := asked["custom"].(map[string]any)
	s.Equal(second.UserMessageID, askedCustom[conversation.ResentFromField])
	s.Equal(question, askedCustom["support_runtime"].(map[string]any)["command_id"], "a person's message is stored under its request id")
	s.Equal("agent", askedCustom["source"], "the router does not answer a message it sent again")
	s.Equal("resp-2", askedCustom[conversation.BranchField].(map[string]any)["response_id"])

	answered := s.stored(reply)
	s.Equal(true, answered["skip_push"])
	runtime := answered["custom"].(map[string]any)["support_runtime"].(map[string]any)
	s.Equal(question, runtime["question_id"])
	s.Equal(question, runtime["command_id"])
	s.Equal("turn-ask-2", runtime["turn_id"], "the reply keeps the turn its tool rounds are recorded under")
	s.Equal(second.AssistantMessageID, answered["custom"].(map[string]any)[conversation.ResentFromField])

	s.Equal([]llm.Message{{Role: llm.User, Content: "Is it raining?"}, {Role: llm.Assistant, Content: "No."},
		{Role: llm.User, Content: "And tomorrow?"}, {Role: llm.Assistant, Content: "Rain."}}, out.History)
	s.Equal([]string{reply, question}, []string{out.Resent[1].ID, out.Resent[0].ID})
	known, err := s.conv.Command(question)
	s.Require().NoError(err)
	s.Equal(reply, known.AssistantMessageID)
	copied := s.branches.copied(s.session)
	s.Contains(copied, edited.UserMessageID)
	s.Contains(copied, edited.AssistantMessageID)
}

func (s *BranchesSuite) TestABranchOfMoreThanAHundredMessagesIsRefusedBeforeAnythingIsWritten() {
	_, second := s.twoTurns()
	before := s.chat.Visible(s.session)
	chosen := make([]store.AgentBranchMessage, conversation.MaxResent+1)
	for i := range chosen {
		chosen[i] = store.AgentBranchMessage{SessionID: s.session, CustomerID: "customer", OriginalMessageID: fmt.Sprintf("m-%d", i),
			AuthorID: "agent", Message: map[string]any{"text": "again"}}
	}

	_, err := s.conv.Activate(s.ctx, conversation.RewindTarget{SwitchID: "sw-1", MessageID: second.UserMessageID, Inclusive: true}, chosen)

	s.ErrorIs(err, conversation.ErrBranchTooLong)
	s.Equal(before, s.chat.Visible(s.session))
	s.Empty(s.chat.HiddenRanges(s.session))
	_, journaled := s.branches.row("sw-1")
	s.False(journaled)
}

// A message written between the read of the tail and the rewind, by another device, is copied
// too and the rewind tried once more.
func (s *BranchesSuite) TestARewindConflictIsRetriedOnceWithTheTailReadAgain() {
	_, second := s.twoTurns()
	waiting, release := s.chat.Hold("/rewind")
	type result struct {
		out conversation.RewindOutcome
		err error
	}
	done := make(chan result, 1)
	go func() {
		out, err := s.conv.Rewind(s.ctx, regenerate("sw-1", second))
		done <- result{out, err}
	}()
	<-waiting
	late, text, user := "late-"+uuid.NewString(), "one more thing", "support-operator"
	_, err := s.chat.Client.Chat().SendMessage(context.Background(), "agent", s.session, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{ID: &late, Text: &text, UserID: &user}})
	s.Require().NoError(err)
	release()
	got := <-done

	s.Require().NoError(got.err)
	s.Equal([]string{second.AssistantMessageID, late}, got.out.Hidden)
	s.Equal([]string{"Is it raining?", "No.", "And tomorrow?"}, s.visibleTexts())
	s.Equal([]string{second.AssistantMessageID, late}, s.branches.copied(s.session))
}

// Chat keeps 100 separate hidden stretches per channel: one more is refused as the branch
// limit, the switch is forgotten with its copies, and the conversation goes on.
func (s *BranchesSuite) TestTooManyRewindsIsTheBranchLimitAndTheConversationGoesOn() {
	s.chat.CapRewinds(1)
	_, second := s.twoTurns()
	_, err := s.conv.Rewind(s.ctx, regenerate("sw-1", second))
	s.Require().NoError(err)
	_, err = s.conv.BeginRegenerate(turn("resp-3", "resp-1"), "again-2", second.UserMessageID)
	s.Require().NoError(err)
	s.replied("again-2", "Sun.")
	third := s.answered(turn("resp-4", "resp-3"), "ask-3", "And Monday?", "Clouds.")

	_, err = s.conv.Rewind(s.ctx, conversation.RewindTarget{SwitchID: "sw-2", Kind: store.SwitchRegenerate, TargetResponseID: "resp-4", MessageID: third.UserMessageID})

	s.ErrorIs(err, conversation.ErrBranchLimit)
	var limit *conversation.TooManyRewindsError
	s.Require().ErrorAs(err, &limit)
	s.Equal(1, limit.Limit)
	_, journaled := s.branches.row("sw-2")
	s.False(journaled, "a refused switch is forgotten")
	s.NotContains(s.branches.copied(s.session), third.AssistantMessageID, "its copies are of messages still visible")
	s.answered(turn("resp-5", "resp-4"), "ask-4", "Thanks", "Any time.")
	s.Equal([]string{"Is it raining?", "No.", "And tomorrow?", "Sun.", "And Monday?", "Clouds.", "Thanks", "Any time."}, s.visibleTexts())
}

func (s *BranchesSuite) TestAnAppWithoutRewindIsRefusedAsBranchesUnavailable() {
	_, second := s.twoTurns()
	s.chat.DisableRewind()

	_, err := s.conv.Rewind(s.ctx, regenerate("sw-1", second))

	s.ErrorIs(err, conversation.ErrBranchUnavailable)
	s.ErrorIs(err, conversation.ErrRewindUnavailable)
	_, journaled := s.branches.row("sw-1")
	s.False(journaled)
	s.Empty(s.branches.copied(s.session))
}

// Regenerating a reply that never landed, the newest message being its question, hides
// nothing and journals nothing: the question is answered again.
func (s *BranchesSuite) TestRegeneratingAReplyThatNeverLandedHidesNothing() {
	_, second := s.twoTurns()

	out, err := s.conv.Rewind(s.ctx, conversation.RewindTarget{SwitchID: "sw-1", Kind: store.SwitchRegenerate, MessageID: second.AssistantMessageID})

	s.Require().NoError(err)
	s.False(out.Rewound)
	s.Empty(s.chat.HiddenRanges(s.session))
	_, journaled := s.branches.row("sw-1")
	s.False(journaled)
	s.Len(out.History, 4)
}

// A rewind while a reply streams finishes it as interrupted before copying it, and leaves no
// write of it queued: the next reply is written.
func (s *BranchesSuite) TestARewindDuringAStreamingReplyLeavesTheQueueEmptyAndMoving() {
	first, second := s.twoTurns()
	streaming, err := s.conv.BeginCommandWithTurn(turn("resp-3", "resp-2"), "ask-3", "And Monday?", "", nil)
	s.Require().NoError(err)
	s.conv.BindTurn("ask-3", "turn-ask-3")
	s.conv.Observe(agent.Responding{TurnID: "turn-ask-3"})
	s.conv.Observe(agent.ResponseDelta{TurnID: "turn-ask-3", Text: "Monday will be"})
	s.conv.Observe(agent.ToolStarted{TurnID: "turn-ask-3", ID: "call-1", Tool: "search", StartedAt: time.Now()})

	_, err = s.conv.Rewind(s.ctx, regenerate("sw-1", second))
	s.Require().NoError(err)
	// The agent's late events about the hidden reply change nothing.
	s.conv.Observe(agent.ResponseDelta{TurnID: "turn-ask-3", Text: " cloudy."})
	s.conv.Interrupt()

	s.written()
	s.Equal([]string{first.UserMessageID, first.AssistantMessageID, second.UserMessageID}, s.chat.Visible(s.session))
	copies, err := s.branches.BranchMessages(s.ctx, "customer", s.session, []string{"resp-3"})
	s.Require().NoError(err)
	s.Require().Len(copies, 2)
	s.Equal(streaming.AssistantMessageID, copies[1].OriginalMessageID)
	runtime := copies[1].Message["custom"].(map[string]any)["support_runtime"].(map[string]any)
	s.Equal("interrupted", runtime["state"])
	_, err = s.conv.BeginRegenerate(turn("resp-4", "resp-1"), "again-2", second.UserMessageID)
	s.Require().NoError(err)
	s.replied("again-2", "Sun.")
	s.Equal([]string{"Is it raining?", "No.", "And tomorrow?", "Sun."}, s.visibleTexts())
}

// Writes Chat will not take now are a refusal the caller can retry, with nothing rewound;
// once Chat takes them again the queue drains.
func (s *BranchesSuite) TestAQueueThatCannotBeWrittenRefusesTheRewindWithoutWedging() {
	_, second := s.twoTurns()
	streaming, err := s.conv.BeginCommand("ask-3", "And Monday?", "")
	s.Require().NoError(err)
	s.written()
	s.chat.Fail("PUT", "/"+streaming.AssistantMessageID)

	_, err = s.conv.Rewind(s.ctx, regenerate("sw-1", second))

	s.ErrorIs(err, conversation.ErrConversationUnwritable)
	s.Empty(s.chat.HiddenRanges(s.session))
	_, journaled := s.branches.row("sw-1")
	s.False(journaled)
	s.chat.Heal("PUT", "/"+streaming.AssistantMessageID)
	s.written()
	s.Equal("interrupted", s.custom(streaming.AssistantMessageID)["support_runtime"].(map[string]any)["state"])
}

// A switch journaled whose rewind may or may not have reached Chat is never forgotten: a
// rewind that timed out can still be committed after the channel is read, and a forgotten
// switch would leave what it hides with no copy.
func (s *BranchesSuite) TestASwitchWhoseRewindIsNotKnownToHaveLandedIsKeptOnReopen() {
	_, second := s.twoTurns()
	// A router that copied the reply and journaled the switch, then lost track of Chat's answer.
	s.Require().NoError(s.branches.StartBranchSwitch(s.ctx, &store.AgentBranchSwitch{ID: "sw-1", SessionID: s.session, CustomerID: "customer",
		Kind: store.SwitchRegenerate, TargetResponseID: "resp-2", ForkMessageID: second.UserMessageID}))
	s.Require().NoError(s.branches.SaveBranchMessages(s.ctx, []store.AgentBranchMessage{{SessionID: s.session, CustomerID: "customer",
		OriginalMessageID: second.AssistantMessageID, ResponseID: "resp-2", SwitchID: "sw-1", AuthorID: "agent", Message: map[string]any{"text": "Rain."}}}))

	s.reopened(nil)

	_, journaled := s.branches.row("sw-1")
	s.True(journaled)
	s.Equal([]string{second.AssistantMessageID}, s.branches.copied(s.session))
}

// A switch whose rewind Chat did not answer is settled without forgetting it.
func (s *BranchesSuite) TestASwitchWhoseRewindTimedOutIsNotForgottenWhenSettled() {
	_, second := s.twoTurns()
	s.chat.FailWith("POST", "/rewind", 504)
	_, err := s.conv.Rewind(s.ctx, regenerate("sw-1", second))
	s.Require().Error(err)

	_, _, _ = s.conv.SettleSwitch(s.ctx, regenerate("sw-1", second), nil)

	_, journaled := s.branches.row("sw-1")
	s.True(journaled)
	s.Equal([]string{second.AssistantMessageID}, s.branches.copied(s.session))
}

func (s *BranchesSuite) TestASwitchStoppedAfterChatRewoundIsFinishedOnReopen() {
	_, second := s.twoTurns()
	s.branches.failAdvanceTo = store.SwitchRewound
	_, err := s.conv.Rewind(s.ctx, regenerate("sw-1", second))
	s.Require().ErrorIs(err, errStopped)

	_, history, switched := s.reopened(nil)

	s.Equal([]llm.Message{{Role: llm.User, Content: "Is it raining?"}, {Role: llm.Assistant, Content: "No."}, {Role: llm.User, Content: "And tomorrow?"}}, history)
	s.Equal([]string{"sw-1"}, *switched)
	row, _ := s.branches.row("sw-1")
	s.Equal(store.SwitchDone, row.State)
}

// An edit hides the message it rewinds to, which Chat then refuses as a read's cursor: whether
// its rewind landed is read from the newest messages back.
func (s *BranchesSuite) TestAnEditStoppedAfterChatRewoundIsFinishedOnReopen() {
	_, second := s.twoTurns()
	s.branches.failAdvanceTo = store.SwitchRewound
	_, err := s.conv.Rewind(s.ctx, edit("sw-1", second))
	s.Require().ErrorIs(err, errStopped)

	_, history, switched := s.reopened(nil)

	s.Equal([]llm.Message{{Role: llm.User, Content: "Is it raining?"}, {Role: llm.Assistant, Content: "No."}}, history)
	s.Equal([]string{"sw-1"}, *switched)
	row, _ := s.branches.row("sw-1")
	s.Equal(store.SwitchDone, row.State)
}

// Whether a switch landed is read back past more than a page of messages written after it.
func (s *BranchesSuite) TestAnEditStoppedAfterChatRewoundIsFinishedOnReopenAfterAPageOfMessages() {
	_, second := s.twoTurns()
	s.branches.failAdvanceTo = store.SwitchRewound
	_, err := s.conv.Rewind(s.ctx, edit("sw-1", second))
	s.Require().ErrorIs(err, errStopped)
	s.branches.failAdvanceTo = ""
	s.chat.PageNewestFirst()
	user := "employee-1"
	for i := range 101 {
		text := fmt.Sprintf("note %d", i)
		_, err := s.chat.Client.Chat().SendMessage(s.ctx, "agent", s.session, &getstream.SendMessageRequest{
			Message: getstream.MessageRequest{Text: &text, UserID: &user}})
		s.Require().NoError(err)
	}

	_, _, switched := s.reopened(nil)

	s.Equal([]string{"sw-1"}, *switched)
	row, _ := s.branches.row("sw-1")
	s.Equal(store.SwitchDone, row.State)
}

// A switch retried after its rewind landed is finished, and Chat refusing a read on the way is
// the conversation having changed, not a failure of the router.
func (s *BranchesSuite) TestAnActivationRetriedWhileChatRefusesTheReadIsAConflict() {
	s.branches.failCountAt = 2
	_, chosen, err := s.activateAfterEdit()
	s.Require().ErrorIs(err, errStopped)
	s.branches.failCountAt = 0
	edited := s.chat.Visible(s.session)[2]
	s.chat.FailWith("POST", "/query", 400)

	_, err = s.conv.Activate(s.ctx, conversation.RewindTarget{SwitchID: "sw-2", TargetResponseID: "resp-2",
		MessageID: edited, Inclusive: true}, chosen)

	s.Require().ErrorIs(err, conversation.ErrBranchConflict)
}

func (s *BranchesSuite) TestASwitchRewoundButNotFinishedIsFinishedOnReopen() {
	_, second := s.twoTurns()
	_, err := s.conv.Rewind(s.ctx, regenerate("sw-1", second))
	s.Require().NoError(err)

	_, history, switched := s.reopened(nil)

	s.Len(history, 3)
	s.Equal([]string{"sw-1"}, *switched)
	row, _ := s.branches.row("sw-1")
	s.Equal(store.SwitchDone, row.State)
}

// activateAfterEdit is the conversation edited at its second question, with the switch that
// brings the original second turn back under sw-2 begun, and the copies it sends again.
func (s *BranchesSuite) activateAfterEdit() (conversation.CommandReceipt, []store.AgentBranchMessage, error) {
	_, second := s.twoTurns()
	_, err := s.conv.Rewind(s.ctx, edit("sw-1", second))
	s.Require().NoError(err)
	edited := s.answered(turn("resp-3", "resp-1"), "ask-2b", "And on Sunday?", "Sun.")
	s.Require().NoError(s.conv.FinishSwitch(s.ctx, "sw-1"))
	chosen, err := s.branches.BranchMessages(s.ctx, "customer", s.session, []string{"resp-2"})
	s.Require().NoError(err)
	_, err = s.conv.Activate(s.ctx, conversation.RewindTarget{SwitchID: "sw-2", TargetResponseID: "resp-2",
		MessageID: edited.UserMessageID, Inclusive: true}, chosen)
	return second, chosen, err
}

func (s *BranchesSuite) TestAnActivationStoppedBeforeSendingIsSentOnReopen() {
	s.branches.failAdvanceTo = store.SwitchResending
	second, chosen, err := s.activateAfterEdit()
	s.Require().ErrorIs(err, errStopped)
	s.Equal([]string{"Is it raining?", "No."}, s.visibleTexts())

	_, history, switched := s.reopened(chosen)

	s.Equal([]string{"Is it raining?", "No.", "And tomorrow?", "Rain."}, s.visibleTexts())
	s.Equal([]string{conversation.ResentID("sw-2", second.UserMessageID), conversation.ResentID("sw-2", second.AssistantMessageID)},
		s.chat.Visible(s.session)[2:])
	s.Len(history, 4)
	s.Equal([]string{"sw-2"}, *switched)
	row, _ := s.branches.row("sw-2")
	s.Equal(store.SwitchDone, row.State)
	s.Equal(2, row.Resent)
}

// A router that sent a message again and stopped before counting it finds it in Chat on
// reopen: no message is sent twice.
func (s *BranchesSuite) TestAnActivationStoppedMidwayIsFinishedWithoutSendingAnythingTwice() {
	s.branches.failCountAt = 2
	second, chosen, err := s.activateAfterEdit()
	s.Require().ErrorIs(err, errStopped)
	row, _ := s.branches.row("sw-2")
	s.Equal(store.SwitchResending, row.State)
	s.Equal(1, row.Resent)

	_, history, switched := s.reopened(chosen)

	s.Equal([]string{"Is it raining?", "No.", "And tomorrow?", "Rain."}, s.visibleTexts())
	s.Equal([]string{conversation.ResentID("sw-2", second.UserMessageID), conversation.ResentID("sw-2", second.AssistantMessageID)},
		s.chat.Visible(s.session)[2:])
	s.Len(history, 4)
	s.Equal([]string{"sw-2"}, *switched)
	row, _ = s.branches.row("sw-2")
	s.Equal(store.SwitchDone, row.State)
	s.Equal(2, row.Resent)
}

// A conversation that is not a session's own private one cannot have branches.
func (s *BranchesSuite) TestOnlyASessionsOwnPrivateConversationHasBranches() {
	creator := "agent"
	made := func(id string, custom map[string]any, members ...string) {
		input := &getstream.ChannelInput{CreatedByID: &creator, Custom: custom}
		for _, member := range members {
			input.Members = append(input.Members, getstream.ChannelMemberRequest{UserID: member})
		}
		_, err := s.chat.Client.Chat().GetOrCreateChannel(context.Background(), "agent", id, &getstream.GetOrCreateChannelRequest{Data: input})
		s.Require().NoError(err)
		person, text := "anya", "hello"
		_, err = s.chat.Client.Chat().SendMessage(context.Background(), "agent", id, &getstream.SendMessageRequest{
			Message: getstream.MessageRequest{Text: &text, UserID: &person}})
		s.Require().NoError(err)
	}
	refusal := func(ctx context.Context, agentID, cid, caller string) string {
		conv, _, _, err := s.service.OpenInApp(ctx, 0, "customer", agentID, cid, caller, "", nil, "")
		s.Require().NoError(err, cid)
		_, err = conv.Rewind(ctx, conversation.RewindTarget{SwitchID: "sw-1", Kind: store.SwitchRegenerate, MessageID: "any"})
		var unsupported *conversation.BranchesUnsupportedError
		s.Require().ErrorAs(err, &unsupported, cid)
		s.ErrorIs(err, conversation.ErrBranchesUnsupported)
		return unsupported.Reason
	}

	thread := conversation.ThreadChannelPrefix + uuid.NewString()
	made(thread, map[string]any{conversation.CustomerField: "customer", "support_agent_id": thread})
	s.Equal(conversation.UnsupportedThread, refusal(conversation.RouterOpensThread(s.ctx, "agent:"+thread), "", "agent:"+thread, ""))

	dispatched := "chat-" + uuid.NewString()
	made(dispatched, map[string]any{"agent_config_id": "config"})
	s.Equal(conversation.UnsupportedDispatched, refusal(conversation.RouterOpensDispatched(s.ctx, "agent:"+dispatched), dispatched, "agent:"+dispatched, ""))

	unstamped := "plain-" + uuid.NewString()
	made(unstamped, map[string]any{conversation.CustomerField: "customer", "support_agent_id": "agent"})
	s.Equal(conversation.UnsupportedChannel, refusal(s.ctx, "agent", "agent:"+unstamped, ""))

	shared := "shared-" + uuid.NewString()
	made(shared, map[string]any{conversation.CustomerField: "customer", "support_agent_id": "agent", "support_owner_id": "anya",
		"support_access": "members", conversation.TriggerField: conversation.SessionCommandTrigger}, "anya")
	s.Equal(conversation.UnsupportedShared, refusal(s.ctx, "agent", "agent:"+shared, "anya"))
}

func (s *BranchesSuite) TestAConversationOpenedWithoutASessionCannotSwitch() {
	_, second := s.twoTurns()
	s.conv.Release()
	reopened, _, _, err := s.service.OpenInApp(context.Background(), 0, "customer", "agent", s.cid, "", "", nil, "")
	s.Require().NoError(err)

	_, err = reopened.Rewind(context.Background(), regenerate("sw-1", second))

	s.Require().Error(err)
	s.True(strings.Contains(err.Error(), "without a session"), err.Error())
	s.Empty(s.chat.HiddenRanges(s.session))
}

// A login finished after its reply was rewound away would write that reply again, which Chat
// refuses for a hidden message and which would hold every write after it: it is not queued.
func (s *BranchesSuite) TestALateWriteOfARewoundReplyIsNeverQueued() {
	s.conv.Release()
	s.session = uuid.NewString()
	s.cid = "agent:" + s.session
	s.ctx = conversation.WithSession(context.Background(), s.session)
	conv, _, _, err := s.service.CreateInApp(s.ctx, 0, "customer", "agent", s.cid, "employee", "", nil, "")
	s.Require().NoError(err)
	s.conv = conv
	s.answered(turn("resp-1", ""), "ask-1", "Is it raining?", "No.")
	second, err := s.conv.BeginCommandWithTurn(turn("resp-2", "resp-1"), "ask-2", "Post it in Slack", "", nil)
	s.Require().NoError(err)
	s.Require().True(s.conv.AskToConnect("employee", conversation.ConnectorAuthorization{Name: "slack", ConnectorID: "slack",
		ConnectionID: "connection-1", AuthorizationID: "attempt-1", Title: "Connect Slack",
		LaunchURL: "https://router.example/v1/agents/connectors/oauth/launch/attempt-1", ExpiresAt: time.Now().Add(time.Hour)}))
	s.replied("ask-2", "Connect Slack first.")
	_, err = s.conv.Rewind(s.ctx, regenerate("sw-1", second))
	s.Require().NoError(err)

	s.True(s.conv.ConnectorConnected("attempt-1"))
	_, err = s.conv.BeginRegenerate(turn("resp-3", "resp-1"), "again-2", second.UserMessageID)
	s.Require().NoError(err)
	s.replied("again-2", "Posted.")

	s.Equal([]string{"Is it raining?", "No.", "Post it in Slack", "Posted."}, s.visibleTexts())
}

// A request whose messages an edit hid, retried after another router reopened the conversation,
// is answered as the duplicate it is, as the router that hid it answers it: its question cannot
// be written again under the same id, so a new command would leave a reply with no question.
func (s *BranchesSuite) TestARequestAnEditHidRetriedAfterARestartIsADuplicate() {
	_, second := s.twoTurns()
	_, err := s.conv.Rewind(s.ctx, edit("sw-1", second))
	s.Require().NoError(err)
	s.Require().NoError(s.conv.FinishSwitch(s.ctx, "sw-1"))
	conv, _, _ := s.reopened(nil)
	before := s.chat.Visible(s.session)

	retry, err := conv.BeginCommand(second.CommandID, "And tomorrow?", "")

	s.Require().NoError(err)
	s.True(retry.Duplicate)
	s.Equal(second.UserMessageID, retry.UserMessageID)
	s.written()
	s.Equal(before, s.chat.Visible(s.session), "nothing is written: no reply without its question")
}

// An edit retried while the rewind its first try sent lands keeps the copies of what that
// rewind hid: the retry's own rewind conflicts, and the channel shows the question gone.
func (s *BranchesSuite) TestAnEditRetriedWhileItsFirstRewindLandsKeepsItsCopies() {
	_, second := s.twoTurns()
	s.chat.FailWith(http.MethodPost, "/rewind", http.StatusGatewayTimeout)
	_, err := s.conv.Rewind(s.ctx, edit("sw-1", second))
	s.Require().Error(err)
	s.chat.Heal(http.MethodPost, "/rewind")
	waiting, release := s.chat.Hold("/rewind")
	retried := make(chan retriedRewind, 1)
	go func() {
		out, err := s.conv.Rewind(s.ctx, edit("sw-1", second))
		retried <- retriedRewind{out, err}
	}()
	<-waiting
	// The first try's rewind lands after all, before the retry's reaches Chat.
	_, err = conversation.NewRewinder(s.chat.Client).Rewind(s.ctx, "agent", s.session, second.UserMessageID, true, second.AssistantMessageID, "agent")
	s.Require().NoError(err)
	release()

	out := <-retried
	s.Require().NoError(out.err)
	s.True(out.Rewound)
	s.ElementsMatch([]string{second.UserMessageID, second.AssistantMessageID}, out.Hidden, "what the earlier rewind hid")
	copies, err := s.branches.BranchMessages(s.ctx, "customer", s.session, []string{"resp-2"})
	s.Require().NoError(err)
	s.Len(copies, 2, "the hidden question and reply are still copied")
	_, journaled := s.branches.row("sw-1")
	s.True(journaled)
	s.NotContains(s.chat.Visible(s.session), second.UserMessageID)
}

// retriedRewind is what a rewind run in the background answered.
type retriedRewind struct {
	conversation.RewindOutcome
	err error
}

// An edit retried while its first rewind lands, whose check of the channel then fails, is left
// for the next open with its copies: whether the earlier rewind landed is not known.
func (s *BranchesSuite) TestAnEditRetriedWhoseCheckOfTheChannelFailsKeepsItsCopies() {
	_, second := s.twoTurns()
	s.chat.FailWith(http.MethodPost, "/rewind", http.StatusGatewayTimeout)
	_, err := s.conv.Rewind(s.ctx, edit("sw-1", second))
	s.Require().Error(err)
	s.chat.Heal(http.MethodPost, "/rewind")
	waiting, release := s.chat.Hold("/rewind")
	retried := make(chan error, 1)
	go func() {
		_, err := s.conv.Rewind(s.ctx, edit("sw-1", second))
		retried <- err
	}()
	<-waiting
	_, err = conversation.NewRewinder(s.chat.Client).Rewind(s.ctx, "agent", s.session, second.UserMessageID, true, second.AssistantMessageID, "agent")
	s.Require().NoError(err)
	// Chat cannot be read once the retry's rewind is refused.
	s.chat.Fail(http.MethodPost, "/query")
	release()

	s.Require().Error(<-retried)
	s.chat.Heal(http.MethodPost, "/query")
	copies, err := s.branches.BranchMessages(s.ctx, "customer", s.session, []string{"resp-2"})
	s.Require().NoError(err)
	s.Len(copies, 2, "the copies of what Chat hid are kept")
	_, journaled := s.branches.row("sw-1")
	s.True(journaled)
}
