package store

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// The money limits an organization is first refused at (LimitNotice.Kind): its credit, Stream's
// spend hard limit for the day or the month, and one of its own spend limits.
const (
	LimitNoCredit       = "no_credit"
	LimitSpendHardLimit = "spend_hard_limit"
	LimitSpendLimit     = "spend_limit"
)

// LimitNotice is the first time an organization was refused at a kind of money limit, or would
// have been in shadow (Enforced false). One is written per organization and kind, ever.
type LimitNotice struct {
	bun.BaseModel `bun:"table:limit_notices,alias:ln"`

	OrganizationID string     `bun:"organization_id,pk"`
	Kind           string     `bun:"kind,pk"`
	AppID          string     `bun:"app_id,notnull"`
	Tier           string     `bun:"tier,notnull"`
	Enforced       bool       `bun:"enforced,notnull"`
	Message        string     `bun:"message,notnull"`
	CreatedAt      time.Time  `bun:"created_at,nullzero,notnull,default:current_timestamp"`
	ClaimedAt      time.Time  `bun:"claimed_at,nullzero,notnull,default:current_timestamp"`
	PostedAt       *time.Time `bun:"posted_at"`
}

// NoteLimit writes the notice unless one of its organization and kind already is. written says
// this call wrote it, and so holds the claim to post it.
func (s *Store) NoteLimit(ctx context.Context, notice LimitNotice) (written bool, err error) {
	if notice.OrganizationID == "" {
		return false, stack.Wrap(errors.New("store: a limit notice needs an organization"))
	}
	switch notice.Kind {
	case LimitNoCredit, LimitSpendHardLimit, LimitSpendLimit:
	default:
		return false, stack.Wrap(fmt.Errorf("store: a limit notice is %s, %s or %s, not %q",
			LimitNoCredit, LimitSpendHardLimit, LimitSpendLimit, notice.Kind))
	}
	notice.CreatedAt, notice.ClaimedAt, notice.PostedAt = time.Time{}, time.Time{}, nil
	result, err := s.db.NewInsert().Model(&notice).On("CONFLICT DO NOTHING").Exec(ctx)
	if err != nil {
		return false, stack.Wrap(err)
	}
	affected, _ := result.RowsAffected()
	return affected > 0, nil
}

// ClaimUnpostedLimits claims, and returns, up to limit notices never posted whose last claim is
// older than stale, so a router that wrote one and stopped, or failed to post it, leaves it for
// another to post.
func (s *Store) ClaimUnpostedLimits(ctx context.Context, stale time.Duration, limit int) ([]LimitNotice, error) {
	var notices []LimitNotice
	due := s.db.NewSelect().Model((*LimitNotice)(nil)).Column("organization_id", "kind").
		Where("posted_at IS NULL").
		Where("claimed_at < now() - ? * interval '1 millisecond'", stale.Milliseconds()).
		OrderExpr("claimed_at").Limit(limit).For("UPDATE SKIP LOCKED")
	err := s.db.NewUpdate().Model((*LimitNotice)(nil)).
		Set("claimed_at = now()").
		Where("(ln.organization_id, ln.kind) IN (?)", due).
		Returning("*").
		Scan(ctx, &notices)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	return notices, nil
}

// MarkLimitPosted records that the organization's notice of the kind was posted.
func (s *Store) MarkLimitPosted(ctx context.Context, organization, kind string) error {
	_, err := s.db.NewUpdate().Model((*LimitNotice)(nil)).
		Set("posted_at = now()").
		Where("organization_id = ?", organization).
		Where("kind = ?", kind).
		Exec(ctx)
	return stack.Wrap(err)
}
