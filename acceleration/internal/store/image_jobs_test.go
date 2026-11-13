//go:build integration

package store

import (
	"encoding/json"
	"time"
)

// queuedImageJob queues a picture in a conversation of its own and returns it as queued.
func (s *StoreSuite) queuedImageJob() ImageJob {
	job := ImageJob{
		ID: "job-" + time.Now().Format("150405.000000000"), CustomerID: "acme", CID: "agent:a", AgentUser: "illustrator",
		ToolCallID: "draw-1", Target: "image-fast", Attribution: json.RawMessage(`{}`), Prompt: "a field",
	}
	queued, err := s.store.QueueImageJob(s.ctx, &job)
	s.Require().NoError(err)
	s.Require().True(queued)
	return job
}

func (s *StoreSuite) TestASlowClaimCannotFinishAnImageJobAnotherClaimTook() {
	// A router that ran past its lease and the one that claimed the job after it would both
	// post, and both record, without the claim being part of every write.
	s.queuedImageJob()
	start := time.Now()
	first, err := s.store.ClaimImageJobs(s.ctx, start, 1, start.Add(time.Millisecond))
	s.Require().NoError(err)
	s.Require().Len(first, 1)
	second, err := s.store.ClaimImageJobs(s.ctx, start.Add(time.Second), 1, start.Add(time.Minute))
	s.Require().NoError(err)
	s.Require().Len(second, 1, "due again once the first lease ran out")
	s.Equal(2, second[0].Attempts)

	finished, err := s.store.FinishImageJob(s.ctx, first[0], ImageJobDone, "", time.Now())
	s.Require().NoError(err)
	s.False(finished, "the first claim no longer holds it")
	saved, err := s.store.SaveImageJobPicture(s.ctx, first[0], json.RawMessage(`{"url":"x"}`))
	s.Require().NoError(err)
	s.False(saved)

	finished, err = s.store.FinishImageJob(s.ctx, second[0], ImageJobDone, "", time.Now())
	s.Require().NoError(err)
	s.True(finished)
}

func (s *StoreSuite) TestAnImageJobUnderItsLeaseIsNotClaimedAgain() {
	s.queuedImageJob()
	now := time.Now()
	claimed, err := s.store.ClaimImageJobs(s.ctx, now, 1, now.Add(time.Minute))
	s.Require().NoError(err)
	s.Require().Len(claimed, 1)

	again, err := s.store.ClaimImageJobs(s.ctx, now.Add(time.Second), 1, now.Add(2*time.Minute))
	s.Require().NoError(err)
	s.Empty(again)
}
