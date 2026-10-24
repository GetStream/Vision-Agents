package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// StreamSource is the built-in source of the events the router itself sends a customer, such
// as a warning that its organization's credit is running out. Its destinations are connector
// event destinations whose connector id is StreamSource, which need no provider app.
const StreamSource = "stream"

// The warnings an organization is sent as its credit runs out (BalanceNotice.Kind).
const (
	NoticeBalanceLow       = "balance_low"
	NoticeBalanceExhausted = "balance_exhausted"
)

// BalanceNotice is one warning sent that an organization's credit is running out. One is
// written per organization, kind and credit granted, so the first router to write it is the
// one that sends it, and a top-up or a grant re-arms it.
type BalanceNotice struct {
	bun.BaseModel `bun:"table:balance_notices,alias:bn"`

	OrganizationID  string    `bun:"organization_id,pk"`
	Kind            string    `bun:"kind,pk"`
	GrantedMicros   int64     `bun:"granted_micros,pk"`
	BalanceMicros   int64     `bun:"balance_micros,notnull"`
	ThresholdMicros int64     `bun:"threshold_micros,notnull"`
	CreatedAt       time.Time `bun:"created_at,nullzero,notnull,default:current_timestamp"`
}

// NoteBalance writes the notice and, when this call is the one that wrote it, queues one
// delivery of it under the delivery's id and headers for every StreamSource destination of
// every app the organization was seen with, each with the body body makes for its app: all in
// one transaction, so a notice written is a notice queued. sent says this call wrote it, and
// queued is how many deliveries it queued.
func (s *Store) NoteBalance(ctx context.Context, notice BalanceNotice, delivery EventDelivery, body func(appID string) ([]byte, error)) (sent bool, queued int, err error) {
	if notice.OrganizationID == "" || delivery.ID == "" {
		return false, 0, stack.Wrap(errors.New("store: a balance notice needs an organization and a delivery id"))
	}
	if notice.Kind != NoticeBalanceLow && notice.Kind != NoticeBalanceExhausted {
		return false, 0, stack.Wrap(fmt.Errorf("store: a balance notice is %s or %s, not %q", NoticeBalanceLow, NoticeBalanceExhausted, notice.Kind))
	}
	headers, err := json.Marshal(delivery.Headers)
	if err != nil {
		return false, 0, stack.Wrap(err)
	}
	err = s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		written, err := tx.NewInsert().Model(&notice).On("CONFLICT DO NOTHING").Exec(ctx)
		if err != nil {
			return err
		}
		if affected, _ := written.RowsAffected(); affected == 0 {
			return nil
		}
		sent = true
		var destinations []struct {
			ID         string `bun:"id"`
			CustomerID string `bun:"customer_id"`
		}
		err = tx.NewSelect().TableExpr("connector_event_destinations AS ced").
			ColumnExpr("ced.id, ced.customer_id").
			Join("JOIN app_organizations AS ao ON ao.app_id = ced.customer_id").
			Where("ao.organization_id = ?", notice.OrganizationID).
			Where("ced.connector_id = ?", StreamSource).
			Scan(ctx, &destinations)
		if err != nil {
			return err
		}
		for _, destination := range destinations {
			payload, err := body(destination.CustomerID)
			if err != nil {
				return err
			}
			if _, err := tx.NewRaw(`
INSERT INTO connector_event_deliveries (destination_id, id, headers, body, attempts, next_attempt_at)
VALUES (?, ?, ?::jsonb, ?, 0, ?)
ON CONFLICT (destination_id, id) DO NOTHING`,
				destination.ID, delivery.ID, string(headers), payload, delivery.NextAttemptAt.UTC()).Exec(ctx); err != nil {
				return err
			}
			queued++
		}
		return nil
	})
	if err != nil {
		return false, 0, stack.Wrap(fmt.Errorf("store: note balance: %w", err))
	}
	return sent, queued, nil
}
