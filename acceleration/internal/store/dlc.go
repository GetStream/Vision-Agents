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

// ErrNoBusinessProfile is an app that has not said who it is yet.
var ErrNoBusinessProfile = errors.New("store: this app has no business profile")

// ErrUnknownUseCase is a use case id the app does not hold.
var ErrUnknownUseCase = errors.New("store: there is no such use case")

// ErrUseCaseMoved is a use case somebody else moved between reading and writing it.
var ErrUseCaseMoved = errors.New("store: the use case changed status meanwhile")

// ErrUnknownOptOut is an opt-out id the app does not hold, or one already revoked.
var ErrUnknownOptOut = errors.New("store: there is no such opt-out")

// How many use cases, reviews or opt-outs a page holds.
const (
	defaultDLCLimit = 50
	maxDLCLimit     = 200
)

// DLCLimit is the page size the use case, review and opt-out lists use for the limit asked
// for. They return one row more than this, so a caller can tell the page is not the last.
func DLCLimit(asked int) int { return clampLimit(asked, defaultDLCLimit, maxDLCLimit) }

// BusinessProfile returns who an app said it is.
func (s *Store) BusinessProfile(ctx context.Context, customerID string) (BusinessProfile, error) {
	if customerID == "" {
		return BusinessProfile{}, stack.Wrap(ErrNoBusinessProfile)
	}
	var profile BusinessProfile
	err := s.db.NewSelect().Model(&profile).Where("customer_id = ?", customerID).Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return BusinessProfile{}, stack.Wrap(ErrNoBusinessProfile)
	}
	if err != nil {
		return BusinessProfile{}, stack.Wrap(fmt.Errorf("store: business profile: %w", err))
	}
	return profile, nil
}

// SaveBusinessProfile writes an app's profile, replacing what it said before. The brand the
// vendor registered stays: it is the vendor's to change, through SetBrand.
func (s *Store) SaveBusinessProfile(ctx context.Context, profile *BusinessProfile) error {
	if profile.CustomerID == "" {
		return stack.Wrap(errors.New("store: a customer is required"))
	}
	now := time.Now().UTC().Truncate(time.Microsecond)
	profile.CreatedAt, profile.UpdatedAt = now, now
	if profile.VerificationDocuments == nil {
		profile.VerificationDocuments = []string{}
	}
	_, err := s.db.NewInsert().Model(profile).
		On("CONFLICT (customer_id) DO UPDATE").
		Set("legal_business_name = EXCLUDED.legal_business_name").
		Set("brand_name = EXCLUDED.brand_name").
		Set("legal_entity_type = EXCLUDED.legal_entity_type").
		Set("organization_type = EXCLUDED.organization_type").
		Set("business_registration_country = EXCLUDED.business_registration_country").
		Set("tax_id = EXCLUDED.tax_id").
		Set("tax_id_issuing_country = EXCLUDED.tax_id_issuing_country").
		Set("registered_address = EXCLUDED.registered_address").
		Set("website_url = EXCLUDED.website_url").
		Set("industry = EXCLUDED.industry").
		Set("authorized_contact_first_name = EXCLUDED.authorized_contact_first_name").
		Set("authorized_contact_last_name = EXCLUDED.authorized_contact_last_name").
		Set("authorized_contact_title = EXCLUDED.authorized_contact_title").
		Set("authorized_contact_email = EXCLUDED.authorized_contact_email").
		Set("authorized_contact_phone = EXCLUDED.authorized_contact_phone").
		Set("privacy_policy_url = EXCLUDED.privacy_policy_url").
		Set("terms_and_conditions_url = EXCLUDED.terms_and_conditions_url").
		Set("stock_symbol = EXCLUDED.stock_symbol").
		Set("stock_exchange = EXCLUDED.stock_exchange").
		Set("business_verification_documents = EXCLUDED.business_verification_documents").
		Set("updated_at = EXCLUDED.updated_at").
		Returning("*").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: save business profile: %w", err))
	}
	return nil
}

// SetBrand records the brand a vendor registered an app as, and where it stands.
func (s *Store) SetBrand(ctx context.Context, customerID, brandID, status string) error {
	_, err := s.db.NewUpdate().Model((*BusinessProfile)(nil)).
		Set("vendor_brand_id = ?", brandID).
		Set("brand_status = ?", status).
		Set("updated_at = ?", time.Now().UTC()).
		Where("customer_id = ?", customerID).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: set brand: %w", err))
	}
	return nil
}

// UseCases pages through an app's use cases, newest first.
func (s *Store) UseCases(ctx context.Context, customerID string, limit int, after *CreatedPosition) ([]UseCase, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: a customer is required"))
	}
	useCases := []UseCase{}
	query := s.db.NewSelect().Model(&useCases).Where("customer_id = ?", customerID)
	if after != nil {
		query = query.Where("(created_at, id) < (?, ?)", after.At, after.ID)
	}
	err := query.Order("created_at DESC", "id DESC").Limit(DLCLimit(limit) + 1).Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: use cases: %w", err))
	}
	return useCases, nil
}

// UseCase returns one use case an app holds.
func (s *Store) UseCase(ctx context.Context, customerID, id string) (UseCase, error) {
	if customerID == "" || id == "" {
		return UseCase{}, stack.Wrap(ErrUnknownUseCase)
	}
	return s.useCase(ctx, s.db.NewSelect().Where("customer_id = ?", customerID).Where("id = ?", id))
}

// UseCaseForNumber returns the use case a number sends as: the one it is assigned to, or
// the app's default. A number the app does not hold sends as the default too.
func (s *Store) UseCaseForNumber(ctx context.Context, customerID, e164 string) (UseCase, error) {
	if customerID == "" {
		return UseCase{}, ErrUnknownUseCase
	}
	assigned := s.db.NewSelect().Model((*PhoneNumber)(nil)).
		Column("dlc_use_case_id").
		Where("customer_id = ?", customerID).
		Where("e164 = ?", e164).
		Where("released_at IS NULL")
	return s.useCase(ctx, s.db.NewSelect().
		Where("customer_id = ?", customerID).
		WhereGroup(" AND ", func(q *bun.SelectQuery) *bun.SelectQuery {
			return q.Where("id IN (?)", assigned).WhereOr("is_default")
		}).
		OrderExpr("is_default"))
}

// UseCaseByID returns a use case whoever holds it, for Stream's own review.
func (s *Store) UseCaseByID(ctx context.Context, id string) (UseCase, error) {
	if id == "" {
		return UseCase{}, stack.Wrap(ErrUnknownUseCase)
	}
	return s.useCase(ctx, s.db.NewSelect().Where("id = ?", id))
}

// UseCaseByCampaign returns the use case a vendor knows by a campaign id.
func (s *Store) UseCaseByCampaign(ctx context.Context, vendor, campaignID string) (UseCase, error) {
	if vendor == "" || campaignID == "" {
		return UseCase{}, ErrUnknownUseCase
	}
	return s.useCase(ctx, s.db.NewSelect().Where("vendor = ?", vendor).Where("vendor_campaign_id = ?", campaignID))
}

func (s *Store) useCase(ctx context.Context, query *bun.SelectQuery) (UseCase, error) {
	var useCase UseCase
	err := query.Model(&useCase).Limit(1).Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return UseCase{}, stack.Wrap(ErrUnknownUseCase)
	}
	if err != nil {
		return UseCase{}, stack.Wrap(fmt.Errorf("store: use case: %w", err))
	}
	return useCase, nil
}

// UseCasesInStatus pages through every app's use cases in one status, longest waiting
// first, which is the order Stream reviews them in.
func (s *Store) UseCasesInStatus(ctx context.Context, status string, limit int, after *CreatedPosition) ([]UseCase, error) {
	useCases := []UseCase{}
	query := s.db.NewSelect().Model(&useCases).Where("status = ?", status)
	if after != nil {
		query = query.Where("(updated_at, id) > (?, ?)", after.At, after.ID)
	}
	err := query.Order("updated_at", "id").Limit(DLCLimit(limit) + 1).Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: use cases in status: %w", err))
	}
	return useCases, nil
}

// HasUseCaseIn reports whether an app holds a use case in any of statuses.
func (s *Store) HasUseCaseIn(ctx context.Context, customerID string, statuses ...string) (bool, error) {
	found, err := s.db.NewSelect().Model((*UseCase)(nil)).
		Where("customer_id = ?", customerID).
		Where("status IN (?)", bun.In(statuses)).
		Exists(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: has use case: %w", err))
	}
	return found, nil
}

// CreateUseCase records a new use case. An app's first one is its default, and a new
// default takes over from the old one.
func (s *Store) CreateUseCase(ctx context.Context, useCase *UseCase) error {
	if useCase.CustomerID == "" || useCase.Name == "" || useCase.Status == "" {
		return stack.Wrap(errors.New("store: a customer, a name and a status are required"))
	}
	now := time.Now().UTC().Truncate(time.Microsecond)
	useCase.ID = newID()
	useCase.CreatedAt, useCase.UpdatedAt = now, now
	if useCase.MessageSamples == nil {
		useCase.MessageSamples = []string{}
	}
	return stack.Wrap(s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		held, err := tx.NewSelect().Model((*UseCase)(nil)).Where("customer_id = ?", useCase.CustomerID).Exists(ctx)
		if err != nil {
			return fmt.Errorf("store: create use case: %w", err)
		}
		useCase.IsDefault = useCase.IsDefault || !held
		if useCase.IsDefault {
			if err := clearDefaultUseCase(ctx, tx, useCase.CustomerID, ""); err != nil {
				return err
			}
		}
		if _, err := tx.NewInsert().Model(useCase).Exec(ctx); err != nil {
			return fmt.Errorf("store: create use case: %w", err)
		}
		return nil
	}))
}

// UpdateUseCase replaces what an app wrote on a use case. Its status, vendor fields and
// review timestamps are left alone: those move through MoveUseCase.
func (s *Store) UpdateUseCase(ctx context.Context, useCase *UseCase) error {
	useCase.UpdatedAt = time.Now().UTC().Truncate(time.Microsecond)
	if useCase.MessageSamples == nil {
		useCase.MessageSamples = []string{}
	}
	return stack.Wrap(s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if useCase.IsDefault {
			if err := clearDefaultUseCase(ctx, tx, useCase.CustomerID, useCase.ID); err != nil {
				return err
			}
		}
		result, err := tx.NewUpdate().Model(useCase).
			Column("name", "is_default", "use_case_type", "description", "message_flow", "message_samples",
				"help_message", "opt_out_message", "opt_in_message", "embedded_links", "embedded_phone",
				"age_gated", "direct_lending", "channels", "updated_at").
			Where("customer_id = ?", useCase.CustomerID).
			Where("id = ?", useCase.ID).
			Where("status = ?", useCase.Status).
			Exec(ctx)
		if err != nil {
			return fmt.Errorf("store: update use case: %w", err)
		}
		if affected, _ := result.RowsAffected(); affected == 0 {
			return ErrUseCaseMoved
		}
		return nil
	}))
}

func clearDefaultUseCase(ctx context.Context, tx bun.Tx, customerID, keep string) error {
	_, err := tx.NewUpdate().Model((*UseCase)(nil)).
		Set("is_default = FALSE").
		Where("customer_id = ?", customerID).
		Where("is_default").
		Where("id <> ?", keep).
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: clear default use case: %w", err)
	}
	return nil
}

// DeleteUseCase removes a use case, its review log and its number assignments.
func (s *Store) DeleteUseCase(ctx context.Context, customerID, id string) error {
	return stack.Wrap(s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		result, err := tx.NewDelete().Model((*UseCase)(nil)).
			Where("customer_id = ?", customerID).
			Where("id = ?", id).
			Exec(ctx)
		if err != nil {
			return fmt.Errorf("store: delete use case: %w", err)
		}
		if affected, _ := result.RowsAffected(); affected == 0 {
			return ErrUnknownUseCase
		}
		_, err = tx.NewUpdate().Model((*PhoneNumber)(nil)).
			Set("dlc_use_case_id = NULL").
			Where("customer_id = ?", customerID).
			Where("dlc_use_case_id = ?", id).
			Exec(ctx)
		if err != nil {
			return fmt.Errorf("store: delete use case: %w", err)
		}
		return nil
	}))
}

// MoveUseCase writes a use case's new status, vendor fields and timestamps, and the log row
// saying who moved it, as one. from is the status it was read in: a use case that moved
// since answers ErrUseCaseMoved and nothing is written.
func (s *Store) MoveUseCase(ctx context.Context, useCase *UseCase, from string, log *ReviewLog) error {
	now := time.Now().UTC().Truncate(time.Microsecond)
	useCase.UpdatedAt = now
	log.ID = newID()
	log.UseCaseID, log.CustomerID = useCase.ID, useCase.CustomerID
	log.FromStatus, log.ToStatus = from, useCase.Status
	log.CreatedAt = now
	return stack.Wrap(s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		result, err := tx.NewUpdate().Model(useCase).
			Column("status", "vendor", "vendor_campaign_id", "vendor_status", "submitted_at", "approved_at", "updated_at").
			Where("id = ?", useCase.ID).
			Where("status = ?", from).
			Exec(ctx)
		if err != nil {
			return fmt.Errorf("store: move use case: %w", err)
		}
		if affected, _ := result.RowsAffected(); affected == 0 {
			return ErrUseCaseMoved
		}
		if _, err := tx.NewInsert().Model(log).Exec(ctx); err != nil {
			return fmt.Errorf("store: move use case: %w", err)
		}
		return nil
	}))
}

// ReviewLogs pages through what happened to a use case, oldest first.
func (s *Store) ReviewLogs(ctx context.Context, customerID, useCaseID string, limit int, after *CreatedPosition) ([]ReviewLog, error) {
	logs := []ReviewLog{}
	query := s.db.NewSelect().Model(&logs).
		Where("customer_id = ?", customerID).
		Where("use_case_id = ?", useCaseID)
	if after != nil {
		query = query.Where("(created_at, id) > (?, ?)", after.At, after.ID)
	}
	err := query.Order("created_at", "id").Limit(DLCLimit(limit) + 1).Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: review logs: %w", err))
	}
	return logs, nil
}

// AssignNumbers makes numbers exactly the ones that send as a use case. A number named
// leaves whatever use case it sent as before; one no longer named goes back to the default.
func (s *Store) AssignNumbers(ctx context.Context, customerID, useCaseID string, numbers []string) error {
	return stack.Wrap(s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		_, err := tx.NewUpdate().Model((*PhoneNumber)(nil)).
			Set("dlc_use_case_id = NULL").
			Where("customer_id = ?", customerID).
			Where("dlc_use_case_id = ?", useCaseID).
			Exec(ctx)
		if err != nil {
			return fmt.Errorf("store: assign numbers: %w", err)
		}
		if len(numbers) == 0 {
			return nil
		}
		result, err := tx.NewUpdate().Model((*PhoneNumber)(nil)).
			Set("dlc_use_case_id = ?", useCaseID).
			Where("customer_id = ?", customerID).
			Where("e164 IN (?)", bun.In(numbers)).
			Where("released_at IS NULL").
			Exec(ctx)
		if err != nil {
			return fmt.Errorf("store: assign numbers: %w", err)
		}
		if affected, _ := result.RowsAffected(); int(affected) != len(numbers) {
			return fmt.Errorf("store: assign numbers: not every number is one %s holds", customerID)
		}
		return nil
	}))
}

// AssignedNumbers returns the numbers assigned to a use case by name, in order.
func (s *Store) AssignedNumbers(ctx context.Context, customerID, useCaseID string) ([]string, error) {
	numbers := []string{}
	err := s.db.NewSelect().Model((*PhoneNumber)(nil)).
		Column("e164").
		Where("customer_id = ?", customerID).
		Where("dlc_use_case_id = ?", useCaseID).
		Where("released_at IS NULL").
		Order("e164").
		Scan(ctx, &numbers)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: assigned numbers: %w", err))
	}
	return numbers, nil
}

// UseCaseNumbers returns the numbers that send as a use case: those assigned to it, and for
// the default those assigned to none.
func (s *Store) UseCaseNumbers(ctx context.Context, useCase UseCase) ([]PhoneNumber, error) {
	numbers := []PhoneNumber{}
	query := s.db.NewSelect().Model(&numbers).
		Where("customer_id = ?", useCase.CustomerID).
		Where("released_at IS NULL")
	if useCase.IsDefault {
		query = query.WhereGroup(" AND ", func(q *bun.SelectQuery) *bun.SelectQuery {
			return q.Where("dlc_use_case_id = ?", useCase.ID).WhereOr("dlc_use_case_id IS NULL")
		})
	} else {
		query = query.Where("dlc_use_case_id = ?", useCase.ID)
	}
	if err := query.Order("e164").Scan(ctx); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: use case numbers: %w", err))
	}
	return numbers, nil
}

// OptOut records that somebody asked not to be reached. Asking twice keeps the first.
func (s *Store) OptOut(ctx context.Context, optOut *OptOut) error {
	if optOut.CustomerID == "" || optOut.Recipient == "" || optOut.Channel == "" {
		return stack.Wrap(errors.New("store: a customer, a recipient and a channel are required"))
	}
	optOut.ID = newID()
	optOut.CreatedAt = time.Now().UTC().Truncate(time.Microsecond)
	_, err := s.db.NewInsert().Model(optOut).
		On("CONFLICT (customer_id, recipient, channel) WHERE revoked_at IS NULL DO NOTHING").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: opt out: %w", err))
	}
	err = s.db.NewSelect().Model(optOut).
		Where("customer_id = ?", optOut.CustomerID).
		Where("recipient = ?", optOut.Recipient).
		Where("channel = ?", optOut.Channel).
		Where("revoked_at IS NULL").
		Scan(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: opt out: %w", err))
	}
	return nil
}

// RevokeOptOut lifts one opt-out, keeping the row.
func (s *Store) RevokeOptOut(ctx context.Context, customerID, id string) error {
	result, err := s.db.NewUpdate().Model((*OptOut)(nil)).
		Set("revoked_at = ?", time.Now().UTC()).
		Where("customer_id = ?", customerID).
		Where("id = ?", id).
		Where("revoked_at IS NULL").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: revoke opt-out: %w", err))
	}
	if affected, _ := result.RowsAffected(); affected == 0 {
		return stack.Wrap(ErrUnknownOptOut)
	}
	return nil
}

// RevokeOptOuts lifts every opt-out a recipient has on a channel and on all of them, as
// texting START does.
func (s *Store) RevokeOptOuts(ctx context.Context, customerID, recipient, channel string) error {
	_, err := s.db.NewUpdate().Model((*OptOut)(nil)).
		Set("revoked_at = ?", time.Now().UTC()).
		Where("customer_id = ?", customerID).
		Where("recipient = ?", recipient).
		Where("channel IN (?)", bun.In([]string{channel, OptOutAll})).
		Where("revoked_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: revoke opt-outs: %w", err)
	}
	return nil
}

// OptedOut reports whether a recipient asked not to be reached on a channel, or on any.
func (s *Store) OptedOut(ctx context.Context, customerID, recipient, channel string) (bool, error) {
	found, err := s.db.NewSelect().Model((*OptOut)(nil)).
		Where("customer_id = ?", customerID).
		Where("recipient = ?", recipient).
		Where("channel IN (?)", bun.In([]string{channel, OptOutAll})).
		Where("revoked_at IS NULL").
		Exists(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: opted out: %w", err))
	}
	return found, nil
}

// OptOuts pages through an app's live opt-outs, newest first.
func (s *Store) OptOuts(ctx context.Context, customerID string, limit int, after *CreatedPosition) ([]OptOut, error) {
	optOuts := []OptOut{}
	query := s.db.NewSelect().Model(&optOuts).
		Where("customer_id = ?", customerID).
		Where("revoked_at IS NULL")
	if after != nil {
		query = query.Where("(created_at, id) < (?, ?)", after.At, after.ID)
	}
	err := query.Order("created_at DESC", "id DESC").Limit(DLCLimit(limit) + 1).Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: opt-outs: %w", err))
	}
	return optOuts, nil
}

// SandboxRecipients returns the numbers an app may reach while sandboxed, in order.
func (s *Store) SandboxRecipients(ctx context.Context, customerID string) ([]string, error) {
	recipients := []string{}
	err := s.db.NewSelect().Model((*SandboxRecipient)(nil)).
		Column("recipient").
		Where("customer_id = ?", customerID).
		Order("recipient").
		Scan(ctx, &recipients)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: sandbox recipients: %w", err))
	}
	return recipients, nil
}

// SetSandboxRecipients replaces the numbers an app may reach while sandboxed.
func (s *Store) SetSandboxRecipients(ctx context.Context, customerID string, recipients []string) error {
	now := time.Now().UTC()
	return stack.Wrap(s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		_, err := tx.NewDelete().Model((*SandboxRecipient)(nil)).Where("customer_id = ?", customerID).Exec(ctx)
		if err != nil {
			return fmt.Errorf("store: set sandbox recipients: %w", err)
		}
		if len(recipients) == 0 {
			return nil
		}
		rows := make([]SandboxRecipient, len(recipients))
		for i, recipient := range recipients {
			rows[i] = SandboxRecipient{CustomerID: customerID, Recipient: recipient, CreatedAt: now}
		}
		if _, err := tx.NewInsert().Model(&rows).On("CONFLICT DO NOTHING").Exec(ctx); err != nil {
			return fmt.Errorf("store: set sandbox recipients: %w", err)
		}
		return nil
	}))
}
