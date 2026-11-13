package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// States of a picture an agent was asked to draw (ImageJob.State).
const (
	ImageJobQueued = "queued"
	ImageJobDone   = "done"
	ImageJobFailed = "failed"
)

// imageJobDoneKeep is how long a posted picture's row is kept, and imageJobFailedKeep a failed
// one's, for someone to find why. They hold the prompt, so not longer: the chat_inbound keeps.
const (
	imageJobDoneKeep   = chatInboundDoneKeep
	imageJobFailedKeep = chatInboundFailedKeep
)

// ImageJob is one picture an agent was asked to draw (20261230090000_image_jobs.sql): what to
// draw, who pays, where it is posted and how far it got.
type ImageJob struct {
	bun.BaseModel `bun:"table:image_jobs,alias:ij"`

	// ID is the job, and names the message it posts.
	ID         string `bun:"id,pk"`
	CustomerID string `bun:"customer_id,notnull"`
	// StreamApp, CID and AgentUser are the conversation: its app, channel and the agent's
	// Stream user, who posts.
	StreamApp int64  `bun:"stream_app,notnull"`
	CID       string `bun:"cid,notnull"`
	AgentUser string `bun:"agent_user,notnull"`
	// SessionID and ToolCallID are the call that asked.
	SessionID  string `bun:"session_id,notnull"`
	ToolCallID string `bun:"tool_call_id,notnull"`
	// AgentID, CallID, Region, Target, Tags and Attribution are what the draw is routed and
	// billed under. Attribution is a routing.Attribution, kept as JSON.
	AgentID     string            `bun:"agent_id,notnull"`
	CallID      string            `bun:"call_id,notnull"`
	Region      string            `bun:"region,notnull"`
	Target      string            `bun:"target,notnull"`
	Tags        map[string]string `bun:"tags,type:jsonb,nullzero"`
	Attribution json.RawMessage   `bun:"attribution,type:jsonb,notnull"`
	Prompt      string            `bun:"prompt,notnull"`
	AspectRatio string            `bun:"aspect_ratio,notnull"`
	State       string            `bun:"state,notnull"`
	// Attempts is how many times the job was claimed, the current claim included: the claim a
	// write holds the job by.
	Attempts int `bun:"attempts,notnull"`
	// Draws is how many draws were started that a provider took; Posts how many posts failed.
	Draws int `bun:"draws,notnull"`
	Posts int `bun:"posts,notnull"`
	// NextAttemptAt is when the job is due; on a claimed row, the end of the claim's lease.
	NextAttemptAt time.Time `bun:"next_attempt_at,notnull"`
	// Drawn and DrawnType are the picture as drawn, until it is uploaded; Picture is it uploaded,
	// a sandbox.Attachment as JSON. Both nil until it is drawn.
	Drawn     []byte          `bun:"drawn,nullzero"`
	DrawnType string          `bun:"drawn_type,notnull"`
	Picture   json.RawMessage `bun:"picture,type:jsonb,nullzero"`
	// Message is what the job posts, once decided; Failed is a message saying why there is no
	// picture.
	Message    string     `bun:"message,notnull"`
	Failed     bool       `bun:"failed,notnull"`
	Outcome    string     `bun:"outcome,notnull"`
	CreatedAt  time.Time  `bun:"created_at,notnull"`
	FinishedAt *time.Time `bun:"finished_at"`
}

// QueueImageJob writes a job to be drawn now, and reports whether it was new: a call of a
// session that already queued one queues nothing.
func (s *Store) QueueImageJob(ctx context.Context, job *ImageJob) (bool, error) {
	if job.ID == "" || job.CID == "" || job.SessionID == "" || job.ToolCallID == "" || job.AgentUser == "" {
		return false, stack.Wrap(errors.New("store: an image job needs an id, a channel, a session, a call and an agent"))
	}
	job.State = ImageJobQueued
	if job.CreatedAt.IsZero() {
		job.CreatedAt = time.Now().UTC()
	}
	if job.NextAttemptAt.IsZero() {
		job.NextAttemptAt = job.CreatedAt
	}
	result, err := s.db.NewInsert().Model(job).On("CONFLICT (session_id, tool_call_id) DO NOTHING").Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: queue image job: %w", err))
	}
	inserted, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: queue image job: %w", err))
	}
	return inserted == 1, nil
}

// ClaimImageJobs takes up to limit queued jobs due by now, moves each one's next attempt to
// leaseUntil and counts the claim. decided takes the jobs whose message is decided, which only
// post; otherwise the ones still to be drawn. A job whose earlier claim's lease ran out, such as
// one a router stopped in the middle of, is due again, as claimChatInboundQuery's rows are.
func (s *Store) ClaimImageJobs(ctx context.Context, now time.Time, limit int, leaseUntil time.Time, decided bool) ([]ImageJob, error) {
	var claimed []ImageJob
	if limit < 1 {
		return claimed, nil
	}
	err := s.db.NewRaw(`
WITH due AS (
    SELECT ij.id FROM image_jobs AS ij
    WHERE ij.state = 'queued' AND ij.next_attempt_at <= ? AND (ij.message <> '') = ?
    ORDER BY ij.next_attempt_at, ij.id
    LIMIT ?
    FOR UPDATE OF ij SKIP LOCKED
)
UPDATE image_jobs AS ij
SET next_attempt_at = ?, attempts = ij.attempts + 1
FROM due
WHERE ij.id = due.id
RETURNING ij.*`, now.UTC(), decided, limit, leaseUntil.UTC()).Scan(ctx, &claimed)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: claim image jobs: %w", err))
	}
	return claimed, nil
}

// NextImageJobAt is when the next queued job is due, and false when none is queued.
func (s *Store) NextImageJobAt(ctx context.Context) (time.Time, bool, error) {
	var next time.Time
	err := s.db.NewSelect().Model((*ImageJob)(nil)).ColumnExpr("next_attempt_at").
		Where("state = ?", ImageJobQueued).OrderExpr("next_attempt_at").Limit(1).Scan(ctx, &next)
	if errors.Is(err, sql.ErrNoRows) {
		return time.Time{}, false, nil
	}
	if err != nil {
		return time.Time{}, false, stack.Wrap(fmt.Errorf("store: next image job: %w", err))
	}
	return next, true, nil
}

// StartImageJobDraw counts a draw the claim is about to start, before it starts: a router that
// stops in the middle of it leaves it counted. It reports false when the claim no longer holds
// the job.
func (s *Store) StartImageJobDraw(ctx context.Context, job ImageJob) (bool, error) {
	return s.updateImageJob(ctx, job, "start image job draw", func(update *bun.UpdateQuery) *bun.UpdateQuery {
		return update.Set("draws = draws + 1")
	})
}

// ReleaseImageJob gives the job back for another claim now, without the draw the claim started
// when undraw: a draw no provider took, or one this router cut short as it stopped.
func (s *Store) ReleaseImageJob(ctx context.Context, job ImageJob, undraw bool, next time.Time) (bool, error) {
	return s.updateImageJob(ctx, job, "release image job", func(update *bun.UpdateQuery) *bun.UpdateQuery {
		if undraw {
			update = update.Set("draws = greatest(draws - 1, 0)")
		}
		return update.Set("next_attempt_at = ?", next.UTC())
	})
}

// KeepImageJobDrawing keeps the picture a claim drew, and the message that posts it, so the job
// is never drawn again. It reports false when the claim no longer holds the job.
func (s *Store) KeepImageJobDrawing(ctx context.Context, job ImageJob, drawn []byte, mediaType, message string) (bool, error) {
	return s.updateImageJob(ctx, job, "keep image job drawing", func(update *bun.UpdateQuery) *bun.UpdateQuery {
		return update.Set("drawn = ?", drawn).Set("drawn_type = ?", mediaType).Set("message = ?", message).Set("failed = false")
	})
}

// DecideImageJobFailure decides that the job posts message, why there is no picture, and lets
// go of any picture it held. It reports false when the claim no longer holds the job.
func (s *Store) DecideImageJobFailure(ctx context.Context, job ImageJob, message string) (bool, error) {
	return s.updateImageJob(ctx, job, "decide image job failure", func(update *bun.UpdateQuery) *bun.UpdateQuery {
		return update.Set("message = ?", message).Set("failed = true").Set("drawn = NULL").Set("picture = NULL")
	})
}

// SaveImageJobPicture keeps the uploaded picture and lets go of the drawn one. It reports false
// when the claim no longer holds the job.
func (s *Store) SaveImageJobPicture(ctx context.Context, job ImageJob, picture json.RawMessage) (bool, error) {
	return s.updateImageJob(ctx, job, "save image job picture", func(update *bun.UpdateQuery) *bun.UpdateQuery {
		return update.Set("picture = ?", picture).Set("drawn = NULL")
	})
}

// RetryImageJobPost counts a post that failed and makes the job due again at next. It reports
// false when the claim no longer holds the job.
func (s *Store) RetryImageJobPost(ctx context.Context, job ImageJob, next time.Time) (bool, error) {
	return s.updateImageJob(ctx, job, "retry image job post", func(update *bun.UpdateQuery) *bun.UpdateQuery {
		return update.Set("posts = posts + 1").Set("next_attempt_at = ?", next.UTC())
	})
}

// FinishImageJob records that a claim is done with the job: posted (done), or failed with
// outcome. It reports false when the claim no longer holds the job.
func (s *Store) FinishImageJob(ctx context.Context, job ImageJob, state, outcome string, at time.Time) (bool, error) {
	if state != ImageJobDone && state != ImageJobFailed {
		return false, stack.Wrap(fmt.Errorf("store: an image job finishes done or failed, not %q", state))
	}
	return s.updateImageJob(ctx, job, "finish image job", func(update *bun.UpdateQuery) *bun.UpdateQuery {
		return update.Set("state = ?", state).Set("outcome = ?", outcome).Set("finished_at = ?", at.UTC()).Set("drawn = NULL")
	})
}

// updateImageJob changes a queued job only while the claim job came from still holds it: a
// slow claim and the one that took the job after its lease ran out never both write it.
func (s *Store) updateImageJob(ctx context.Context, job ImageJob, doing string,
	set func(*bun.UpdateQuery) *bun.UpdateQuery) (bool, error) {
	result, err := set(s.db.NewUpdate().Model((*ImageJob)(nil))).
		Where("id = ?", job.ID).Where("state = ?", ImageJobQueued).Where("attempts = ?", job.Attempts).
		Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: %s: %w", doing, err))
	}
	changed, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: %s: %w", doing, err))
	}
	return changed == 1, nil
}

// PruneImageJobs deletes the jobs past their keep as of now, and returns how many.
func (s *Store) PruneImageJobs(ctx context.Context, now time.Time) (int64, error) {
	result, err := s.db.NewDelete().Model((*ImageJob)(nil)).
		Where("(state = ? AND finished_at < ?) OR (state = ? AND finished_at < ?)",
			ImageJobDone, now.Add(-imageJobDoneKeep).UTC(), ImageJobFailed, now.Add(-imageJobFailedKeep).UTC()).
		Exec(ctx)
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: prune image jobs: %w", err))
	}
	return result.RowsAffected()
}
