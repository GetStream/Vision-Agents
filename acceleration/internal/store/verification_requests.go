package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// VerificationRequest is what an organization told us about itself to be verified. An
// organization has one, replaced whenever it sends again.
type VerificationRequest struct {
	bun.BaseModel `bun:"table:verification_requests,alias:vr"`

	OrganizationID string    `bun:"organization_id,pk"`
	LegalName      string    `bun:"legal_name,notnull"`
	Country        string    `bun:"country,notnull"`
	Website        string    `bun:"website,notnull"`
	TaxID          string    `bun:"tax_id,notnull"`
	UseCase        string    `bun:"use_case,notnull"`
	CallsPerMonth  string    `bun:"calls_per_month,notnull"`
	Email          string    `bun:"email,notnull"`
	SubmittedBy    string    `bun:"submitted_by,notnull"`
	SubmittedAt    time.Time `bun:"submitted_at,notnull"`
}

// VerificationPosition is where a page of verification requests ended: the next page starts
// after it.
type VerificationPosition struct {
	SubmittedAt    time.Time `json:"s"`
	OrganizationID string    `json:"o"`
}

// VerificationRequestEntry is a request with the tier Stream last sent for its organization,
// empty when it never sent one.
type VerificationRequestEntry struct {
	VerificationRequest `bun:",extend"`

	Tier string `bun:"tier"`
}

// SaveVerificationRequest stores an organization's request, replacing the one it had.
func (s *Store) SaveVerificationRequest(ctx context.Context, request VerificationRequest) (VerificationRequest, error) {
	request.SubmittedAt = request.SubmittedAt.UTC()
	_, err := s.db.NewInsert().Model(&request).
		On("CONFLICT (organization_id) DO UPDATE").
		Set("legal_name = EXCLUDED.legal_name").
		Set("country = EXCLUDED.country").
		Set("website = EXCLUDED.website").
		Set("tax_id = EXCLUDED.tax_id").
		Set("use_case = EXCLUDED.use_case").
		Set("calls_per_month = EXCLUDED.calls_per_month").
		Set("email = EXCLUDED.email").
		Set("submitted_by = EXCLUDED.submitted_by").
		Set("submitted_at = EXCLUDED.submitted_at").
		Returning("*").
		Exec(ctx)
	if err != nil {
		return VerificationRequest{}, stack.Wrap(fmt.Errorf("store: save verification request: %w", err))
	}
	return request, nil
}

// VerificationRequestOf returns an organization's request, or nothing for one that never sent it.
func (s *Store) VerificationRequestOf(ctx context.Context, organizationID string) (*VerificationRequestEntry, error) {
	var entry VerificationRequestEntry
	err := s.verificationEntries(&entry).Where("vr.organization_id = ?", organizationID).Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: verification request of: %w", err))
	}
	return &entry, nil
}

// ListVerificationRequests returns requests newest first, after the position a page ended at.
// It returns one more than limit so the caller can tell whether another page follows.
func (s *Store) ListVerificationRequests(ctx context.Context, limit int, after *VerificationPosition) ([]VerificationRequestEntry, error) {
	var entries []VerificationRequestEntry
	query := s.verificationEntries(&entries).
		OrderExpr("vr.submitted_at DESC, vr.organization_id DESC").
		Limit(limit + 1)
	if after != nil {
		query = query.Where("(vr.submitted_at, vr.organization_id) < (?, ?)", after.SubmittedAt.UTC(), after.OrganizationID)
	}
	if err := query.Scan(ctx); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: list verification requests: %w", err))
	}
	return entries, nil
}

func (s *Store) verificationEntries(into any) *bun.SelectQuery {
	return s.db.NewSelect().Model(into).
		ColumnExpr("vr.*").
		ColumnExpr("COALESCE(ot.tier, '') AS tier").
		Join("LEFT JOIN organization_tiers AS ot ON ot.organization_id = vr.organization_id")
}
