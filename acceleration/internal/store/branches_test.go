//go:build integration

package store

import (
	"time"
)

// branchAnswered starts a finished response of session "one" on into, after parent and answering
// question's question: empty question is a question of its own.
func (s *StoreSuite) branchAnswered(into *Store, id, parent, question string, at time.Time) {
	s.Require().NoError(into.StartResponse(s.ctx, &AgentResponse{
		ID: id, SessionID: "one", CustomerID: "app", Said: id, CreatedAt: at,
		ParentResponseID: parent, QuestionResponseID: question,
	}))
	s.Require().NoError(into.FinishResponse(s.ctx, id, ResponseCompleted, "", at.Add(time.Second)))
}

// conversation is the ids of session "one"'s responses still in its conversation, oldest first.
func (s *StoreSuite) conversation() []string {
	responses, err := s.store.SessionResponses(s.ctx, "app", "one", 0, nil)
	s.Require().NoError(err)
	ids := make([]string, 0, len(responses))
	for _, response := range responses {
		ids = append(ids, response.ID)
	}
	return ids
}

func responseIDs(responses []AgentResponse) []string {
	found := make([]string, 0, len(responses))
	for _, response := range responses {
		found = append(found, response.ID)
	}
	return found
}

// branchedAfterFirst is session "one" with a first question, then a question asked after it
// answered twice, and an edit of that question answered twice:
//
//	first ─┬─ asked, asked-again          (question asked)
//	       └─ edited, edited-again        (question edited)
func (s *StoreSuite) branchedAfterFirst() {
	s.opened("one", "app", s.base, nil)
	s.branchAnswered(s.store, "first", "", "", s.base)
	s.branchAnswered(s.store, "asked", "first", "", s.base.Add(time.Minute))
	s.branchAnswered(s.store, "asked-again", "first", "asked", s.base.Add(2*time.Minute))
	s.branchAnswered(s.store, "edited", "first", "", s.base.Add(3*time.Minute))
	s.branchAnswered(s.store, "edited-again", "first", "edited", s.base.Add(4*time.Minute))
}

func (s *StoreSuite) TestARegenerationIsAnotherReplyToTheSameQuestion() {
	s.branchedAfterFirst()

	replies, err := s.store.Siblings(s.ctx, "app", "one", "asked-again", SiblingsOfReply)
	s.Require().NoError(err)
	s.Equal([]string{"asked", "asked-again"}, responseIDs(replies))
	index, count := SiblingPosition(replies, replies[1], SiblingsOfReply)
	s.Equal([2]int{2, 2}, [2]int{index, count})

	replies, err = s.store.Siblings(s.ctx, "app", "one", "edited", SiblingsOfReply)
	s.Require().NoError(err)
	s.Equal([]string{"edited", "edited-again"}, responseIDs(replies))
}

func (s *StoreSuite) TestAnEditIsAnotherQuestionHoweverOftenEachWasAnswered() {
	s.branchedAfterFirst()

	questions, err := s.store.Siblings(s.ctx, "app", "one", "edited-again", SiblingsOfQuestion)
	s.Require().NoError(err)
	s.Equal([]string{"asked", "edited"}, responseIDs(questions))

	regenerated, err := s.store.sessionResponse(s.ctx, "app", "one", "asked-again")
	s.Require().NoError(err)
	index, count := SiblingPosition(questions, regenerated, SiblingsOfQuestion)
	s.Equal([2]int{1, 2}, [2]int{index, count}, "a regeneration stands where its question does")
	edit, err := s.store.sessionResponse(s.ctx, "app", "one", "edited-again")
	s.Require().NoError(err)
	index, count = SiblingPosition(questions, edit, SiblingsOfQuestion)
	s.Equal([2]int{2, 2}, [2]int{index, count})
}

func (s *StoreSuite) TestTheFirstQuestionOfASessionHasAlternativesToo() {
	s.opened("one", "app", s.base, nil)
	// A row from before branches were kept has no question, and is no alternative to anything.
	_, err := s.store.shard.NewInsert().Model(&AgentResponse{
		ID: "legacy", SessionID: "one", CustomerID: "app", Status: ResponseCompleted, CreatedAt: s.base,
	}).Exec(s.ctx)
	s.Require().NoError(err)
	s.branchAnswered(s.store, "opening", "", "", s.base.Add(time.Minute))
	s.branchAnswered(s.store, "reworded", "", "", s.base.Add(2*time.Minute))

	questions, err := s.store.Siblings(s.ctx, "app", "one", "reworded", SiblingsOfQuestion)
	s.Require().NoError(err)
	s.Equal([]string{"opening", "reworded"}, responseIDs(questions))

	legacy, err := s.store.Siblings(s.ctx, "app", "one", "legacy", SiblingsOfQuestion)
	s.Require().NoError(err)
	s.Equal([]string{"legacy"}, responseIDs(legacy))
}

func (s *StoreSuite) TestSiblingsOfSomethingElseAreRefused() {
	s.branchedAfterFirst()

	_, err := s.store.Siblings(s.ctx, "app", "one", "asked", "thread")
	s.ErrorContains(err, "siblings of")
}

func (s *StoreSuite) TestTheTailOfABranchEndsAtItsNewestResponseWhereverItIs() {
	// fork ─┬─ old ── old-later      (old-later answered last)
	//       └─ newer
	s.opened("one", "app", s.base, nil)
	s.branchAnswered(s.store, "fork", "", "", s.base)
	s.branchAnswered(s.store, "old", "fork", "", s.base.Add(time.Minute))
	s.branchAnswered(s.store, "newer", "fork", "old", s.base.Add(2*time.Minute))
	s.branchAnswered(s.store, "old-later", "old", "", s.base.Add(3*time.Minute))

	tail, err := s.store.BranchTail(s.ctx, "app", "one", "fork")
	s.Require().NoError(err)
	s.Equal([]string{"fork", "old", "old-later"}, responseIDs(tail))

	tail, err = s.store.BranchTail(s.ctx, "app", "one", "newer")
	s.Require().NoError(err)
	s.Equal([]string{"newer"}, responseIDs(tail))
}

func (s *StoreSuite) TestMovingThePathRewindsTheOldBranchAndBringsBackTheNew() {
	// first ── second ── third        (rewound by a regenerate of second)
	//      └── second-again           (the conversation)
	s.opened("one", "app", s.base, nil)
	s.branchAnswered(s.store, "first", "", "", s.base)
	s.branchAnswered(s.store, "second", "first", "", s.base.Add(time.Minute))
	s.branchAnswered(s.store, "third", "second", "", s.base.Add(2*time.Minute))
	s.Require().NoError(s.store.RewindResponses(s.ctx, "app", "one", "first", s.base.Add(3*time.Minute)))
	s.branchAnswered(s.store, "second-again", "first", "second", s.base.Add(4*time.Minute))
	s.Require().Equal([]string{"first", "second-again"}, s.conversation())

	s.Require().NoError(s.store.MoveActivePath(s.ctx, "app", "one", "third", s.base.Add(5*time.Minute)))
	s.Equal([]string{"first", "second", "third"}, s.conversation())

	s.Require().NoError(s.store.MoveActivePath(s.ctx, "app", "one", "third", s.base.Add(6*time.Minute)))
	s.Equal([]string{"first", "second", "third"}, s.conversation(), "moving to the same path again changes nothing")

	s.Require().NoError(s.store.MoveActivePath(s.ctx, "app", "one", "second-again", s.base.Add(7*time.Minute)))
	s.Equal([]string{"first", "second-again"}, s.conversation())
}

func (s *StoreSuite) TestMovingToAResponseInTheConversationCutsItShortThere() {
	s.opened("one", "app", s.base, nil)
	s.branchAnswered(s.store, "first", "", "", s.base)
	s.branchAnswered(s.store, "second", "first", "", s.base.Add(time.Minute))

	s.Require().NoError(s.store.MoveActivePath(s.ctx, "app", "one", "first", s.base.Add(2*time.Minute)))

	s.Equal([]string{"first"}, s.conversation())
}

func (s *StoreSuite) TestMovingThePathKeepsTurnsFromBeforeBranchesWereKept() {
	s.opened("one", "app", s.base, nil)
	_, err := s.store.shard.NewInsert().Model(&AgentResponse{
		ID: "legacy", SessionID: "one", CustomerID: "app", Status: ResponseCompleted, CreatedAt: s.base,
	}).Exec(s.ctx)
	s.Require().NoError(err)
	s.branchAnswered(s.store, "first", "legacy", "", s.base.Add(time.Minute))
	s.branchAnswered(s.store, "second", "first", "", s.base.Add(2*time.Minute))
	s.Require().NoError(s.store.RewindResponses(s.ctx, "app", "one", "legacy", s.base.Add(3*time.Minute)))
	s.branchAnswered(s.store, "first-again", "legacy", "first", s.base.Add(4*time.Minute))

	s.Require().NoError(s.store.MoveActivePath(s.ctx, "app", "one", "second", s.base.Add(5*time.Minute)))

	s.Equal([]string{"legacy", "first", "second"}, s.conversation())
}

func (s *StoreSuite) TestMovingToAResponseTheSessionDoesNotHaveFails() {
	s.opened("one", "app", s.base, nil)

	err := s.store.MoveActivePath(s.ctx, "app", "one", "nowhere", s.base)
	s.ErrorIs(err, ErrUnknownResponse)
}

func (s *ShardedStoreSuite) TestMovingThePathMovesItInEveryShard() {
	// first and second were answered here; the session was carried on in the other shard,
	// where second was regenerated and answered on.
	s.opened("one", "app", s.base, nil)
	s.branchAnswered(s.store, "first", "", "", s.base)
	s.branchAnswered(s.store, "second", "first", "", s.base.Add(time.Minute))
	s.Require().NoError(s.other.SaveSession(s.ctx, &AgentSession{ID: "one", CustomerID: "app", AgentName: "docs", UserID: "jlahey", CallerKind: "user", CreatedAt: s.base}))
	s.Require().NoError(s.other.RewindResponses(s.ctx, "app", "one", "first", s.base.Add(2*time.Minute)))
	s.branchAnswered(s.other, "second-again", "first", "second", s.base.Add(3*time.Minute))
	s.branchAnswered(s.other, "third-again", "second-again", "", s.base.Add(4*time.Minute))
	s.Require().Equal([]string{"first", "second-again", "third-again"}, s.conversation())

	s.Require().NoError(s.store.MoveActivePath(s.ctx, "app", "one", "second", s.base.Add(5*time.Minute)))
	s.Equal([]string{"first", "second"}, s.conversation())

	tail, err := s.other.BranchTail(s.ctx, "app", "one", "second-again")
	s.Require().NoError(err)
	s.Equal([]string{"second-again", "third-again"}, responseIDs(tail))
	s.Require().NoError(s.other.MoveActivePath(s.ctx, "app", "one", "third-again", s.base.Add(6*time.Minute)))
	s.Equal([]string{"first", "second-again", "third-again"}, s.conversation())

	questions, err := s.store.Siblings(s.ctx, "app", "one", "second-again", SiblingsOfReply)
	s.Require().NoError(err)
	s.Equal([]string{"second", "second-again"}, responseIDs(questions))
}

// hidden is a copy of a message of session "one" that switch hid.
func hidden(id, response, switchID string, at time.Time, text string) AgentBranchMessage {
	return AgentBranchMessage{
		SessionID: "one", OriginalMessageID: id, CustomerID: "app", ResponseID: response,
		SwitchID: switchID, ChatCreatedAt: at, AuthorID: "jlahey", Message: map[string]any{"text": text},
	}
}

func (s *StoreSuite) TestACopiedMessageIsListedByTheResponseItBelongsTo() {
	s.opened("one", "app", s.base, nil)
	s.Require().NoError(s.store.SaveBranchMessages(s.ctx, []AgentBranchMessage{
		hidden("m2", "second", "switch", s.base.Add(time.Minute), "answer"),
		hidden("m1", "second", "switch", s.base, "question"),
		hidden("m3", "third", "switch", s.base.Add(2*time.Minute), "later"),
	}))

	found, err := s.store.BranchMessages(s.ctx, "app", "one", []string{"second"})
	s.Require().NoError(err)
	s.Require().Len(found, 2)
	s.Equal("m1", found[0].OriginalMessageID)
	s.Equal("m2", found[1].OriginalMessageID)
	s.Equal("answer", found[1].Message["text"])

	all, err := s.store.BranchMessages(s.ctx, "app", "one", nil)
	s.Require().NoError(err)
	s.Len(all, 3)
}

func (s *StoreSuite) TestCopyingAMessageAgainKeepsWhatItSaysNow() {
	s.opened("one", "app", s.base, nil)
	s.Require().NoError(s.store.SaveBranchMessages(s.ctx, []AgentBranchMessage{hidden("m1", "second", "switch", s.base, "partial")}))
	s.Require().NoError(s.store.SaveBranchMessages(s.ctx, []AgentBranchMessage{hidden("m1", "second", "switch", s.base, "partial and finished")}))

	found, err := s.store.BranchMessages(s.ctx, "app", "one", nil)
	s.Require().NoError(err)
	s.Require().Len(found, 1)
	s.Equal("partial and finished", found[0].Message["text"])
}

func (s *StoreSuite) TestCopiesAreDeletedForTheirSessionAlone() {
	s.opened("one", "app", s.base, nil)
	s.opened("two", "app", s.base, nil)
	other := hidden("m2", "second", "switch", s.base, "kept")
	other.SessionID = "two"
	s.Require().NoError(s.store.SaveBranchMessages(s.ctx, []AgentBranchMessage{hidden("m1", "second", "switch", s.base, "gone"), other}))

	deleted, err := s.store.DeleteBranchMessages(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.EqualValues(1, deleted)

	left, err := s.store.BranchMessages(s.ctx, "app", "two", nil)
	s.Require().NoError(err)
	s.Len(left, 1)
}

func (s *StoreSuite) TestForgettingAUserDeletesTheCopiesOfTheirSessionsAlone() {
	s.opened("one", "app", s.base, nil)
	s.opened("two", "app", s.base, func(session *AgentSession) { session.UserID = "rlafleur" })
	theirs := hidden("m2", "second", "switch", s.base, "kept")
	theirs.SessionID = "two"
	s.Require().NoError(s.store.SaveBranchMessages(s.ctx, []AgentBranchMessage{hidden("m1", "second", "switch", s.base, "gone"), theirs}))

	deleted, err := s.store.DeleteUserBranchMessages(s.ctx, "app", "jlahey")
	s.Require().NoError(err)
	s.EqualValues(1, deleted)

	mine, err := s.store.BranchMessages(s.ctx, "app", "one", nil)
	s.Require().NoError(err)
	s.Empty(mine)
	left, err := s.store.BranchMessages(s.ctx, "app", "two", nil)
	s.Require().NoError(err)
	s.Len(left, 1)
}

// switched records a switch of session "one" on into, hiding from from at at.
func (s *StoreSuite) switched(into *Store, id string, from, at time.Time, state string) {
	s.Require().NoError(into.StartBranchSwitch(s.ctx, &AgentBranchSwitch{
		ID: id, SessionID: "one", CustomerID: "app", Kind: SwitchRegenerate, TargetResponseID: "second",
		ForkMessageID: "m-" + id, HiddenFrom: from, State: state, CreatedAt: at,
	}))
}

func (s *StoreSuite) TestASwitchOnlyMovesForward() {
	s.opened("one", "app", s.base, nil)
	s.switched(s.store, "switch", s.base, s.base.Add(time.Minute), "")

	s.Require().NoError(s.store.AdvanceBranchSwitch(s.ctx, "app", "switch", SwitchResending))
	s.Require().NoError(s.store.AdvanceBranchSwitch(s.ctx, "app", "switch", SwitchRewound))

	unfinished, err := s.store.UnfinishedBranchSwitches(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Require().Len(unfinished, 1)
	s.Equal(SwitchResending, unfinished[0].State, "a late router finishing an earlier step does not move it back")

	s.Require().NoError(s.store.AdvanceBranchSwitch(s.ctx, "app", "switch", SwitchDone))
	unfinished, err = s.store.UnfinishedBranchSwitches(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Empty(unfinished)
}

func (s *StoreSuite) TestStartingASwitchAgainKeepsHowFarItGot() {
	s.opened("one", "app", s.base, nil)
	s.switched(s.store, "switch", s.base, s.base.Add(time.Minute), "")
	s.Require().NoError(s.store.AdvanceBranchSwitch(s.ctx, "app", "switch", SwitchRewound))

	s.switched(s.store, "switch", s.base, s.base.Add(time.Minute), "")

	unfinished, err := s.store.UnfinishedBranchSwitches(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Require().Len(unfinished, 1)
	s.Equal(SwitchRewound, unfinished[0].State)
}

func (s *StoreSuite) TestAdvancingToAStateThereIsNotIsRefused() {
	s.opened("one", "app", s.base, nil)
	s.switched(s.store, "switch", s.base, s.base.Add(time.Minute), "")

	s.ErrorContains(s.store.AdvanceBranchSwitch(s.ctx, "app", "switch", "abandoned"), "unknown state")
}

func (s *StoreSuite) TestEachMessageSentAgainIsCounted() {
	s.opened("one", "app", s.base, nil)
	s.switched(s.store, "switch", s.base, s.base.Add(time.Minute), "")

	first, err := s.store.CountBranchResent(s.ctx, "app", "switch")
	s.Require().NoError(err)
	second, err := s.store.CountBranchResent(s.ctx, "app", "switch")
	s.Require().NoError(err)

	s.Equal([2]int{1, 2}, [2]int{first, second})
}

func (s *StoreSuite) TestAnAbandonedSwitchTakesOnlyItsOwnCopiesWithIt() {
	s.opened("one", "app", s.base, nil)
	s.switched(s.store, "earlier", s.base, s.base.Add(time.Minute), SwitchDone)
	s.switched(s.store, "refused", s.base.Add(2*time.Minute), s.base.Add(3*time.Minute), "")
	s.Require().NoError(s.store.SaveBranchMessages(s.ctx, []AgentBranchMessage{
		hidden("m1", "second", "earlier", s.base, "hidden"),
		hidden("m2", "third", "refused", s.base.Add(2*time.Minute), "still visible"),
	}))

	s.Require().NoError(s.store.AbandonBranchSwitch(s.ctx, "app", "one", "refused"))

	left, err := s.store.BranchMessages(s.ctx, "app", "one", nil)
	s.Require().NoError(err)
	s.Require().Len(left, 1)
	s.Equal("m1", left[0].OriginalMessageID)
	unfinished, err := s.store.UnfinishedBranchSwitches(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Empty(unfinished)
	count, err := s.store.MergedRangeCount(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Equal(1, count)
}

func (s *StoreSuite) TestUnfinishedSwitchesAreOldestFirst() {
	s.opened("one", "app", s.base, nil)
	s.switched(s.store, "later", s.base, s.base.Add(2*time.Minute), "")
	s.switched(s.store, "earlier", s.base, s.base.Add(time.Minute), SwitchResending)
	s.switched(s.store, "finished", s.base, s.base.Add(30*time.Second), SwitchDone)

	unfinished, err := s.store.UnfinishedBranchSwitches(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Require().Len(unfinished, 2)
	s.Equal("earlier", unfinished[0].ID)
	s.Equal("later", unfinished[1].ID)
}

func (s *StoreSuite) TestTheRangesRewindsHidAreCountedMerged() {
	s.opened("one", "app", s.base, nil)
	// Two rewinds whose ranges overlap, one that starts where the previous one ended, one
	// apart from the rest, and one that has not rewound the channel yet.
	s.switched(s.store, "a", s.base, s.base.Add(10*time.Minute), SwitchDone)
	s.switched(s.store, "b", s.base.Add(5*time.Minute), s.base.Add(15*time.Minute), SwitchDone)
	s.switched(s.store, "c", s.base.Add(15*time.Minute), s.base.Add(20*time.Minute), SwitchRewound)
	s.switched(s.store, "d", s.base.Add(time.Hour), s.base.Add(2*time.Hour), SwitchResending)
	s.switched(s.store, "e", s.base.Add(3*time.Hour), s.base.Add(4*time.Hour), SwitchSnapshotted)

	count, err := s.store.MergedRangeCount(s.ctx, "app", "one")
	s.Require().NoError(err)

	s.Equal(2, count)
}

func (s *ShardedStoreSuite) TestTheRangesOfEveryShardAreCountedTogether() {
	s.opened("one", "app", s.base, nil)
	s.Require().NoError(s.other.SaveSession(s.ctx, &AgentSession{ID: "one", CustomerID: "app", AgentName: "docs", UserID: "jlahey", CallerKind: "user", CreatedAt: s.base}))
	s.switched(s.store, "here", s.base, s.base.Add(10*time.Minute), SwitchDone)
	s.switched(s.other, "there", s.base.Add(5*time.Minute), s.base.Add(15*time.Minute), SwitchDone)
	s.switched(s.other, "apart", s.base.Add(time.Hour), s.base.Add(2*time.Hour), SwitchDone)

	count, err := s.store.MergedRangeCount(s.ctx, "app", "one")
	s.Require().NoError(err)

	s.Equal(2, count)
}

func (s *StoreSuite) TestDeletingASessionTakesItsBranchesWithIt() {
	s.branchedAfterFirst()
	s.Require().NoError(s.store.SaveBranchMessages(s.ctx, []AgentBranchMessage{hidden("m1", "asked", "switch", s.base, "gone")}))
	s.switched(s.store, "switch", s.base, s.base.Add(time.Minute), SwitchRewound)

	s.Require().NoError(s.store.DeleteSession(s.ctx, "app", "one"))

	messages, err := s.store.BranchMessages(s.ctx, "app", "one", nil)
	s.Require().NoError(err)
	s.Empty(messages)
	unfinished, err := s.store.UnfinishedBranchSwitches(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Empty(unfinished)
	count, err := s.store.MergedRangeCount(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Zero(count)
}

func (s *ShardedStoreSuite) TestDeletingASessionTakesItsBranchesWithItInEveryShard() {
	s.opened("one", "app", s.base, nil)
	s.Require().NoError(s.other.SaveSession(s.ctx, &AgentSession{ID: "one", CustomerID: "app", AgentName: "docs", UserID: "jlahey", CallerKind: "user", CreatedAt: s.base}))
	s.Require().NoError(s.store.SaveBranchMessages(s.ctx, []AgentBranchMessage{hidden("here", "second", "switch", s.base, "gone")}))
	s.Require().NoError(s.other.SaveBranchMessages(s.ctx, []AgentBranchMessage{hidden("there", "second", "switch", s.base, "gone")}))
	s.switched(s.store, "here", s.base, s.base.Add(time.Minute), SwitchRewound)
	s.switched(s.other, "there", s.base, s.base.Add(time.Minute), SwitchRewound)
	messages, err := s.store.BranchMessages(s.ctx, "app", "one", nil)
	s.Require().NoError(err)
	s.Require().Len(messages, 2)

	s.Require().NoError(s.store.DeleteSession(s.ctx, "app", "one"))

	messages, err = s.store.BranchMessages(s.ctx, "app", "one", nil)
	s.Require().NoError(err)
	s.Empty(messages)
	unfinished, err := s.store.UnfinishedBranchSwitches(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Empty(unfinished)
}

func (s *StoreSuite) TestBranchesAreOnlyTheirOwnCustomers() {
	s.branchedAfterFirst()
	s.Require().NoError(s.store.SaveBranchMessages(s.ctx, []AgentBranchMessage{hidden("m1", "asked", "switch", s.base, "mine")}))
	s.switched(s.store, "switch", s.base, s.base.Add(time.Minute), SwitchRewound)
	s.Require().NoError(s.store.RewindResponses(s.ctx, "app", "one", "first", s.base.Add(5*time.Minute)))

	_, err := s.store.Siblings(s.ctx, "somebody-else", "one", "asked", SiblingsOfReply)
	s.ErrorIs(err, ErrUnknownResponse)
	_, err = s.store.BranchTail(s.ctx, "somebody-else", "one", "first")
	s.ErrorIs(err, ErrUnknownResponse)
	s.ErrorIs(s.store.MoveActivePath(s.ctx, "somebody-else", "one", "asked", s.base.Add(6*time.Minute)), ErrUnknownResponse)
	s.Equal([]string{"first"}, s.conversation(), "another customer moves nothing")

	messages, err := s.store.BranchMessages(s.ctx, "somebody-else", "one", nil)
	s.Require().NoError(err)
	s.Empty(messages)
	deleted, err := s.store.DeleteBranchMessages(s.ctx, "somebody-else", "one")
	s.Require().NoError(err)
	s.Zero(deleted)
	deleted, err = s.store.DeleteUserBranchMessages(s.ctx, "somebody-else", "jlahey")
	s.Require().NoError(err)
	s.Zero(deleted)
	stolen := hidden("m1", "asked", "switch", s.base, "theirs")
	stolen.CustomerID = "somebody-else"
	s.Require().NoError(s.store.SaveBranchMessages(s.ctx, []AgentBranchMessage{stolen}))
	messages, err = s.store.BranchMessages(s.ctx, "app", "one", nil)
	s.Require().NoError(err)
	s.Require().Len(messages, 1)
	s.Equal("mine", messages[0].Message["text"], "another customer cannot overwrite a copy")

	s.Require().NoError(s.store.AdvanceBranchSwitch(s.ctx, "somebody-else", "switch", SwitchDone))
	_, err = s.store.CountBranchResent(s.ctx, "somebody-else", "switch")
	s.ErrorContains(err, "no switch")
	unfinished, err := s.store.UnfinishedBranchSwitches(s.ctx, "somebody-else", "one")
	s.Require().NoError(err)
	s.Empty(unfinished)
	count, err := s.store.MergedRangeCount(s.ctx, "somebody-else", "one")
	s.Require().NoError(err)
	s.Zero(count)

	unfinished, err = s.store.UnfinishedBranchSwitches(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Require().Len(unfinished, 1)
	s.Equal(SwitchRewound, unfinished[0].State)
	s.Zero(unfinished[0].Resent)
	s.Equal([]string{"first"}, s.conversation())
}
