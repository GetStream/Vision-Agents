package ledger

import (
	"cmp"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/google/uuid"
	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// Bounds of what a management write stores.
const (
	// MaxReasonLength is the longest a suspension's or a restore's reason may be, in
	// characters.
	MaxReasonLength = 500
	// MaxIdempotencyKeyLength is the longest an idempotency key may be, in characters.
	MaxIdempotencyKeyLength = 255
	// maxReferenceLength and maxNoteLength bound a credit entry's free text, in characters.
	maxReferenceLength = 255
	maxNoteLength      = 1000
)

// The actions manage_audit records.
const (
	ActionSeedTier   = "seed_tier"
	ActionSetTier    = "set_tier"
	ActionDeleteTier = "delete_tier"
	ActionSetLimits  = "set_limits"
	ActionSuspend    = "suspend"
	ActionRestore    = "restore"
	ActionAddCredit  = "add_credit"
)

var (
	// ErrDefaultTier is a delete of the default tier, which answers for every tier string
	// with no row and so cannot go.
	ErrDefaultTier = errors.New("ledger: the default tier cannot be deleted")
	// ErrIdempotencyKeyReused is an account write made under a key that already holds a
	// request of another operation or organization, or with another body (for a credit,
	// another kind or amount): a writer's bug, since a retry sends the same request.
	ErrIdempotencyKeyReused = errors.New("ledger: the idempotency key holds another request")
)

// InvalidError is a write refused for what it holds, with what is wrong.
type InvalidError struct {
	Problem string
}

func (e *InvalidError) Error() string { return "ledger: " + e.Problem }

// invalid is the InvalidError of problem.
func invalid(format string, args ...any) error {
	return &InvalidError{Problem: fmt.Sprintf(format, args...)}
}

// Actor is who a management write is made by: an id, and the name of the person behind it
// when the caller gave one.
type Actor struct {
	ID   string
	Name string
}

// Tier is one tier string's row.
type Tier struct {
	bun.BaseModel `bun:"table:tiers,alias:t"`

	Name      string         `bun:"name,pk" json:"name"`
	FailMode  usage.FailMode `bun:"fail_mode,notnull" json:"fail_mode"`
	Limits    usage.Limits   `bun:"limits,type:jsonb,notnull" json:"limits"`
	UpdatedBy string         `bun:"updated_by,notnull" json:"updated_by"`
	UpdatedAt time.Time      `bun:"updated_at,nullzero,notnull,default:current_timestamp" json:"updated_at"`
}

// Contract is the tier as the contract has it.
func (t Tier) Contract() usage.Tier {
	return usage.Tier{Name: t.Name, FailMode: t.FailMode, Limits: t.Limits}
}

// Account is one organization's credit, spend, status and own settings. An organization with
// no row is active with nothing granted or spent.
type Account struct {
	bun.BaseModel `bun:"table:accounts,alias:a"`

	OrganizationID string              `bun:"organization_id,pk" json:"organization_id"`
	Status         usage.AccountStatus `bun:"status,nullzero,notnull,default:'active'" json:"status"`
	StatusReason   string              `bun:"status_reason,notnull" json:"status_reason"`
	LimitOverrides usage.Limits        `bun:"limit_overrides,type:jsonb,notnull" json:"limit_overrides"`
	GrantedMicros  int64               `bun:"granted_micros,notnull" json:"granted_micros"`
	SpentMicros    int64               `bun:"spent_micros,notnull" json:"spent_micros"`
	CreatedAt      time.Time           `bun:"created_at,nullzero,notnull,default:current_timestamp" json:"created_at"`
	UpdatedAt      time.Time           `bun:"updated_at,nullzero,notnull,default:current_timestamp" json:"updated_at"`
}

// virtualAccount is the account of an organization with no row.
func virtualAccount(organization string) Account {
	return Account{OrganizationID: organization, Status: usage.AccountActive}
}

// CreditEntry is one change to an organization's credit.
type CreditEntry struct {
	bun.BaseModel `bun:"table:credit_entries,alias:ce"`

	ID             int64            `bun:"id,pk,autoincrement"`
	OrganizationID string           `bun:"organization_id,notnull"`
	Kind           usage.CreditKind `bun:"kind,notnull"`
	AmountMicros   int64            `bun:"amount_micros,notnull"`
	Reference      string           `bun:"reference,notnull"`
	IdempotencyKey string           `bun:"idempotency_key,notnull"`
	Note           string           `bun:"note,notnull"`
	Actor          string           `bun:"actor,notnull"`
	ActorName      string           `bun:"actor_name,notnull"`
	CreatedAt      time.Time        `bun:"created_at,nullzero,notnull,default:current_timestamp"`
}

// Contract is the entry as the contract has it.
func (e CreditEntry) Contract() usage.CreditEntry {
	return usage.CreditEntry{
		ID: e.ID, OrganizationID: e.OrganizationID, Kind: e.Kind, AmountMicros: e.AmountMicros,
		Reference: e.Reference, IdempotencyKey: e.IdempotencyKey, Note: e.Note,
		Actor: e.Actor, ActorName: e.ActorName, CreatedAt: e.CreatedAt.UTC(),
	}
}

// ManageAudit is one management write: who made it, on which organization (none for a
// tier), and the thing it changed before and after.
type ManageAudit struct {
	bun.BaseModel `bun:"table:manage_audit,alias:ma"`

	ID             int64           `bun:"id,pk,autoincrement"`
	Actor          string          `bun:"actor,notnull"`
	ActorName      string          `bun:"actor_name,notnull"`
	OrganizationID string          `bun:"organization_id,notnull"`
	Action         string          `bun:"action,notnull"`
	Reason         string          `bun:"reason,notnull"`
	Before         json.RawMessage `bun:"before,type:jsonb,notnull"`
	After          json.RawMessage `bun:"after,type:jsonb,notnull"`
	CreatedAt      time.Time       `bun:"created_at,nullzero,notnull,default:current_timestamp"`
}

// ManageRequest is an account write made under an idempotency key: a credit, a suspension, a
// restore or an override. Its operation is the action the write audits.
type ManageRequest struct {
	bun.BaseModel `bun:"table:manage_requests,alias:mr"`

	IdempotencyKey string `bun:"idempotency_key,pk"`
	Operation      string `bun:"operation,notnull"`
	OrganizationID string `bun:"organization_id,notnull"`
	// RequestSHA256 is the hex SHA-256 of what the request asks for, as newRequest has it.
	RequestSHA256 string    `bun:"request_sha256,notnull"`
	CreatedAt     time.Time `bun:"created_at,nullzero,notnull,default:current_timestamp"`
}

// newRequest is the request of operation on organization under key, asking for body, whose
// JSON encoding is its digest.
func newRequest(key, operation, organization string, body any) (ManageRequest, error) {
	if err := checkKey(key); err != nil {
		return ManageRequest{}, err
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		return ManageRequest{}, fmt.Errorf("ledger: %s: %w", operation, err)
	}
	digest := sha256.Sum256(encoded)
	return ManageRequest{IdempotencyKey: key, Operation: operation, OrganizationID: organization, RequestSHA256: hex.EncodeToString(digest[:])}, nil
}

// checkKey refuses an idempotency key that is not 1 to 255 characters of UTF-8 without a
// control character. Postgres stores invalid UTF-8 replaced, so two different keys would be
// one, and the second request answered as a replay of the first.
func checkKey(key string) error {
	if n := utf8.RuneCountInString(key); n == 0 || n > MaxIdempotencyKeyLength {
		return invalid("an idempotency key is required: 1 to %d characters", MaxIdempotencyKeyLength)
	}
	if !utf8.ValidString(key) || strings.ContainsFunc(key, unicode.IsControl) {
		return invalid("the idempotency key is not UTF-8 or holds a control character")
	}
	return nil
}

// claim records the request under its key in tx and reports whether it is the key's first. A
// key holding a request of another operation, organization or body is ErrIdempotencyKeyReused.
// Two writers of one key wait on each other here, and the second finds the first's request
// once it commits.
func claim(ctx context.Context, tx bun.Tx, request ManageRequest) (bool, error) {
	var inserted []ManageRequest
	if err := tx.NewInsert().Model(&request).On("CONFLICT (idempotency_key) DO NOTHING").
		Returning("*").Scan(ctx, &inserted); err != nil {
		return false, fmt.Errorf("ledger: %s: %w", request.Operation, err)
	}
	if len(inserted) > 0 {
		return true, nil
	}
	var existing ManageRequest
	if err := tx.NewSelect().Model(&existing).Where("idempotency_key = ?", request.IdempotencyKey).Scan(ctx); err != nil {
		return false, fmt.Errorf("ledger: %s: %w", request.Operation, err)
	}
	if existing.Operation != request.Operation || existing.OrganizationID != request.OrganizationID ||
		existing.RequestSHA256 != request.RequestSHA256 {
		return false, ErrIdempotencyKeyReused
	}
	return false, nil
}

// audit records one management write in the transaction that makes it. A side that does
// not exist, a tier before it was created, is nil.
func audit(ctx context.Context, db bun.IDB, actor Actor, organization, action, reason string, before, after any) error {
	snapshot := func(v any) (json.RawMessage, error) {
		if v == nil {
			return json.RawMessage(`{}`), nil
		}
		return json.Marshal(v)
	}
	row := ManageAudit{Actor: actor.ID, ActorName: actor.Name, OrganizationID: organization, Action: action, Reason: reason}
	var err error
	if row.Before, err = snapshot(before); err != nil {
		return fmt.Errorf("ledger: audit %s: %w", action, err)
	}
	if row.After, err = snapshot(after); err != nil {
		return fmt.Errorf("ledger: audit %s: %w", action, err)
	}
	if _, err := db.NewInsert().Model(&row).Exec(ctx); err != nil {
		return fmt.Errorf("ledger: audit %s: %w", action, err)
	}
	return nil
}

// Tiers is every tier row, by name.
func (l *Ledger) Tiers(ctx context.Context) ([]usage.Tier, error) {
	var rows []Tier
	if err := l.db.NewSelect().Model(&rows).Order("name").Scan(ctx); err != nil {
		return nil, fmt.Errorf("ledger: tiers: %w", err)
	}
	tiers := make([]usage.Tier, 0, len(rows))
	for _, row := range rows {
		tiers = append(tiers, row.Contract())
	}
	return tiers, nil
}

// Tier is the row of the tier name, and true; or, when name has none, the default tier's row
// (the built-in default while there is none) and false. It reads the table every time, so a
// tier written shows at once.
func (l *Ledger) Tier(ctx context.Context, name string) (usage.Tier, bool, error) {
	tiers, err := tiersNamed(ctx, l.db, []string{name})
	if err != nil {
		return usage.Tier{}, false, err
	}
	tier, found := tiers.resolve(name)
	return tier, found, nil
}

// tierSet is some tier rows by name, the default tier's among them when it has one.
type tierSet map[string]usage.Tier

// tiersNamed reads the rows of names and of the default tier.
func tiersNamed(ctx context.Context, db bun.IDB, names []string) (tierSet, error) {
	var rows []Tier
	if err := db.NewSelect().Model(&rows).Where("name IN (?)", bun.In(append(slices.Clone(names), usage.DefaultTier))).Scan(ctx); err != nil {
		return nil, fmt.Errorf("ledger: tiers: %w", err)
	}
	tiers := make(tierSet, len(rows))
	for _, row := range rows {
		tiers[row.Name] = row.Contract()
	}
	return tiers, nil
}

// resolve is the tier name's row and true, or the default tier and false.
func (t tierSet) resolve(name string) (usage.Tier, bool) {
	if tier, ok := t[name]; ok {
		return tier, true
	}
	if tier, ok := t[usage.DefaultTier]; ok {
		return tier, false
	}
	return usage.BuiltInDefaultTier(), false
}

// SetTier writes the tier, a new one or over the one there.
func (l *Ledger) SetTier(ctx context.Context, tier usage.Tier, actor Actor) (usage.Tier, error) {
	if problem := tier.Problem(); problem != "" {
		return usage.Tier{}, invalid("%s", problem)
	}
	row := Tier{Name: tier.Name, FailMode: tier.FailMode, Limits: tier.Limits.Compact(), UpdatedBy: actor.ID}
	err := l.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if err := lockTiers(ctx, tx); err != nil {
			return err
		}
		var before []Tier
		if err := tx.NewSelect().Model(&before).Where("name = ?", tier.Name).Scan(ctx); err != nil {
			return fmt.Errorf("ledger: set tier: %w", err)
		}
		if err := tx.NewInsert().Model(&row).On("CONFLICT (name) DO UPDATE").
			Set("fail_mode = EXCLUDED.fail_mode").Set("limits = EXCLUDED.limits").
			Set("updated_by = EXCLUDED.updated_by").Set("updated_at = now()").
			Returning("*").Scan(ctx); err != nil {
			return fmt.Errorf("ledger: set tier: %w", err)
		}
		var previous any
		if len(before) > 0 {
			previous = before[0]
		}
		return audit(ctx, tx, actor, "", ActionSetTier, "", previous, row)
	})
	if err != nil {
		return usage.Tier{}, err
	}
	return row.Contract(), nil
}

// DeleteTier deletes the tier name's row, and reports whether there was one; its
// organizations fall back to the default tier. The default tier cannot be deleted.
func (l *Ledger) DeleteTier(ctx context.Context, name string, actor Actor) (bool, error) {
	if name == usage.DefaultTier {
		return false, ErrDefaultTier
	}
	var deleted bool
	err := l.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if err := lockTiers(ctx, tx); err != nil {
			return err
		}
		var rows []Tier
		if err := tx.NewDelete().Model(&rows).Where("name = ?", name).Returning("*").Scan(ctx); err != nil {
			return fmt.Errorf("ledger: delete tier: %w", err)
		}
		if len(rows) == 0 {
			return nil
		}
		deleted = true
		return audit(ctx, tx, actor, "", ActionDeleteTier, "", rows[0], nil)
	})
	return deleted, err
}

// lockTiers holds every other tier write, the start's seeding included, until the
// transaction ends: two writers creating one tier would otherwise both read no row before
// and audit a create, the second over the first's.
func lockTiers(ctx context.Context, tx bun.Tx) error {
	if _, err := tx.ExecContext(ctx, "SELECT pg_advisory_xact_lock(hashtext('usage tiers'))"); err != nil {
		return fmt.Errorf("ledger: lock tiers: %w", err)
	}
	return nil
}

// newestEvent orders an organization's events newest first, the usage id breaking a tie, so
// its newest event, and so its tier, is the same on every read.
const newestEvent = "occurred_at DESC, usage_id DESC"

// TierOverrides are the accounts, after the organization id after and at most limit of
// them, that override any setting and whose newest event carried tier name, by organization
// id; and whether there are more. For the default tier they include the organizations whose
// newest event carried a tier with no row, which run on the default's settings. An
// organization with overrides and no event is listed under no tier.
func (l *Ledger) TierOverrides(ctx context.Context, name, after string, limit int) ([]Account, bool, error) {
	var rows []Account
	query := l.db.NewSelect().Model(&rows).
		Join(`CROSS JOIN LATERAL (SELECT e.organization_tier FROM usage_events AS e
			WHERE e.organization_id = a.organization_id ORDER BY e.` + newestEvent + ` LIMIT 1) AS newest`).
		Where("a.limit_overrides <> '{}'::jsonb")
	if name == usage.DefaultTier {
		query = query.Where("(newest.organization_tier = ? OR NOT EXISTS (SELECT 1 FROM tiers AS t WHERE t.name = newest.organization_tier))", name)
	} else {
		query = query.Where("newest.organization_tier = ?", name)
	}
	if after != "" {
		query = query.Where("a.organization_id > ?", after)
	}
	if err := query.Order("a.organization_id").Limit(limit + 1).Scan(ctx); err != nil {
		return nil, false, fmt.Errorf("ledger: tier overrides: %w", err)
	}
	more := len(rows) > limit
	return rows[:min(len(rows), limit)], more, nil
}

// Account is the organization's account row, and whether it has one: without, it is
// virtual, active with nothing granted or spent.
func (l *Ledger) Account(ctx context.Context, organization string) (Account, bool, error) {
	account := Account{}
	err := l.db.NewSelect().Model(&account).Where("organization_id = ?", organization).Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return virtualAccount(organization), false, nil
	}
	if err != nil {
		return Account{}, false, fmt.Errorf("ledger: account: %w", err)
	}
	return account, true, nil
}

// LastTier is the tier the organization's newest event carried, and whether it has any.
func (l *Ledger) LastTier(ctx context.Context, organization string) (string, bool, error) {
	var tier string
	err := l.db.NewSelect().Model((*UsageEvent)(nil)).Column("organization_tier").
		Where("organization_id = ?", organization).OrderExpr(newestEvent).Limit(1).Scan(ctx, &tier)
	if errors.Is(err, sql.ErrNoRows) {
		return "", false, nil
	}
	if err != nil {
		return "", false, fmt.Errorf("ledger: last tier: %w", err)
	}
	return tier, true, nil
}

// changeAccount locks the request's account, writing it first when it has none, and claims
// the request's key. The key's first request hands the account to change, which updates it
// and returns the row after, and audits the change; a replay changes nothing and answers the
// account as it is, with false.
func (l *Ledger) changeAccount(ctx context.Context, request ManageRequest, actor Actor, reason string,
	change func(ctx context.Context, tx bun.Tx, before Account) (Account, error),
) (Account, bool, error) {
	var after Account
	applied := false
	err := l.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		before, err := lockAccount(ctx, tx, request.OrganizationID)
		if err != nil {
			return err
		}
		first, err := claim(ctx, tx, request)
		if err != nil {
			return err
		}
		if !first {
			after = before
			return nil
		}
		// A credit written before manage_requests holds its key in credit_entries alone.
		credited, err := tx.NewSelect().Model((*CreditEntry)(nil)).Where("idempotency_key = ?", request.IdempotencyKey).Exists(ctx)
		if err != nil {
			return fmt.Errorf("ledger: %s: %w", request.Operation, err)
		}
		if credited {
			return ErrIdempotencyKeyReused
		}
		if after, err = change(ctx, tx, before); err != nil {
			return err
		}
		applied = true
		return audit(ctx, tx, actor, request.OrganizationID, request.Operation, reason, before, after)
	})
	if err != nil {
		return Account{}, false, err
	}
	return after, applied, nil
}

// lockAccount writes the organization's account when it has none and locks it until the
// transaction ends, so every change to it is made on the row as it is.
func lockAccount(ctx context.Context, tx bun.Tx, organization string) (Account, error) {
	if _, err := tx.NewInsert().Model(&Account{OrganizationID: organization}).Column("organization_id").
		On("CONFLICT (organization_id) DO NOTHING").Exec(ctx); err != nil {
		return Account{}, fmt.Errorf("ledger: account: %w", err)
	}
	account := Account{}
	if err := tx.NewSelect().Model(&account).Where("organization_id = ?", organization).For("UPDATE").Scan(ctx); err != nil {
		return Account{}, fmt.Errorf("ledger: account: %w", err)
	}
	return account, nil
}

// SetAccountLimits replaces the organization's own settings with overrides: a setting left
// out goes back to its tier's, and none at all clears them. It is made once per key: a
// replay with the same overrides changes nothing and answers the account as it is, with
// false.
func (l *Ledger) SetAccountLimits(ctx context.Context, organization string, overrides usage.Limits, key string, actor Actor) (Account, bool, error) {
	if problem := overrides.Problem(); problem != "" {
		return Account{}, false, invalid("%s", problem)
	}
	overrides = overrides.Compact()
	request, err := newRequest(key, ActionSetLimits, organization, overrides)
	if err != nil {
		return Account{}, false, err
	}
	return l.changeAccount(ctx, request, actor, "", func(ctx context.Context, tx bun.Tx, before Account) (Account, error) {
		after := before
		after.LimitOverrides = overrides
		if err := tx.NewUpdate().Model(&after).Column("limit_overrides").Set("updated_at = now()").
			WherePK().Returning("*").Scan(ctx); err != nil {
			return Account{}, fmt.Errorf("ledger: set account limits: %w", err)
		}
		return after, nil
	})
}

// SetAccountStatus suspends or restores the organization, for reason, which the audit keeps.
// Suspending sets the account's status reason, a suspended one's included; restoring clears
// it, and restoring an active account changes nothing but is audited all the same. It is
// made once per key: a replay with the same reason changes nothing, is not audited, and
// answers the account as it is, with false.
func (l *Ledger) SetAccountStatus(ctx context.Context, organization string, status usage.AccountStatus, reason, key string, actor Actor) (Account, bool, error) {
	if n := utf8.RuneCountInString(reason); n == 0 || n > MaxReasonLength {
		return Account{}, false, invalid("a reason is required: 1 to %d characters", MaxReasonLength)
	}
	if !storable(reason) {
		return Account{}, false, invalid("reason is not UTF-8 or holds a NUL character")
	}
	action, statusReason := ActionSuspend, reason
	switch status {
	case usage.AccountSuspended:
	case usage.AccountActive:
		action, statusReason = ActionRestore, ""
	default:
		return Account{}, false, invalid("status %q is not active or suspended", status)
	}
	request, err := newRequest(key, action, organization, statusRequest{Reason: reason})
	if err != nil {
		return Account{}, false, err
	}
	return l.changeAccount(ctx, request, actor, reason, func(ctx context.Context, tx bun.Tx, before Account) (Account, error) {
		after := before
		after.Status, after.StatusReason = status, statusReason
		if err := tx.NewUpdate().Model(&after).Column("status", "status_reason").Set("updated_at = now()").
			WherePK().Returning("*").Scan(ctx); err != nil {
			return Account{}, fmt.Errorf("ledger: set account status: %w", err)
		}
		return after, nil
	})
}

// statusRequest is what a suspension or a restore asks for.
type statusRequest struct {
	Reason string `json:"reason"`
}

// creditRequest is what a credit asks for, as a replay is compared: an entry's reference
// and note are not.
type creditRequest struct {
	Kind         usage.CreditKind `json:"kind"`
	AmountMicros int64            `json:"amount_micros"`
}

// AddCredit writes the entry and adds its amount to the organization's credit, once per
// idempotency key: an entry already written under the key is answered as it is, with false,
// and changes nothing. Only the sign and the key are checked: a suspended account takes
// credit too. The entry's ID and CreatedAt are the service's.
func (l *Ledger) AddCredit(ctx context.Context, entry CreditEntry) (CreditEntry, bool, error) {
	if problem := entry.Kind.AmountProblem(entry.AmountMicros); problem != "" {
		return CreditEntry{}, false, invalid("%s", problem)
	}
	if entry.AmountMicros > usage.MaxCostMicros || entry.AmountMicros < -usage.MaxCostMicros {
		return CreditEntry{}, false, invalid("amount_micros is at most %d either way", int64(usage.MaxCostMicros))
	}
	request, err := newRequest(entry.IdempotencyKey, ActionAddCredit, entry.OrganizationID,
		creditRequest{Kind: entry.Kind, AmountMicros: entry.AmountMicros})
	if err != nil {
		return CreditEntry{}, false, err
	}
	if utf8.RuneCountInString(entry.Reference) > maxReferenceLength || utf8.RuneCountInString(entry.Note) > maxNoteLength {
		return CreditEntry{}, false, invalid("reference is at most %d characters and note at most %d", maxReferenceLength, maxNoteLength)
	}
	if !storable(entry.Reference) || !storable(entry.Note) {
		return CreditEntry{}, false, invalid("reference and note are UTF-8 without a NUL character")
	}
	entry.ID, entry.CreatedAt = 0, time.Time{}
	written := false
	err = l.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		before, err := lockAccount(ctx, tx, entry.OrganizationID)
		if err != nil {
			return err
		}
		first, err := claim(ctx, tx, request)
		if err != nil {
			return err
		}
		if !first {
			if err := tx.NewSelect().Model(&entry).Where("idempotency_key = ?", entry.IdempotencyKey).Scan(ctx); err != nil {
				return fmt.Errorf("ledger: add credit: %w", err)
			}
			return nil
		}
		// A credit written before manage_requests holds its key in credit_entries alone: it
		// is answered as a replay when it is the same entry.
		var inserted []CreditEntry
		if err := tx.NewInsert().Model(&entry).On("CONFLICT (idempotency_key) DO NOTHING").
			Returning("*").Scan(ctx, &inserted); err != nil {
			return fmt.Errorf("ledger: add credit: %w", err)
		}
		if len(inserted) == 0 {
			var existing CreditEntry
			if err := tx.NewSelect().Model(&existing).Where("idempotency_key = ?", entry.IdempotencyKey).Scan(ctx); err != nil {
				return fmt.Errorf("ledger: add credit: %w", err)
			}
			if existing.OrganizationID != entry.OrganizationID || existing.Kind != entry.Kind || existing.AmountMicros != entry.AmountMicros {
				return ErrIdempotencyKeyReused
			}
			entry = existing
			return nil
		}
		entry, written = inserted[0], true
		after := before
		if err := tx.NewUpdate().Model(&after).Set("granted_micros = a.granted_micros + ?", entry.AmountMicros).
			Set("updated_at = now()").WherePK().Returning("*").Scan(ctx); err != nil {
			return fmt.Errorf("ledger: add credit: %w", err)
		}
		return audit(ctx, tx, Actor{ID: entry.Actor, Name: entry.ActorName}, entry.OrganizationID, ActionAddCredit, "",
			creditAudit{Account: before}, creditAudit{Account: after, Entry: new(entry.Contract())})
	})
	if err != nil {
		return CreditEntry{}, false, err
	}
	return entry, written, nil
}

// creditAudit is what an add_credit audit row keeps: the account before, and after it the
// account and the entry written.
type creditAudit struct {
	Account Account            `json:"account"`
	Entry   *usage.CreditEntry `json:"entry,omitempty"`
}

// storable reports whether text is what Postgres stores as given: UTF-8 without a NUL, which
// it would refuse or the driver strip.
func storable(text string) bool {
	return utf8.ValidString(text) && !strings.ContainsRune(text, 0)
}

// CreditPosition is where a page of credit entries ends: entries are newest first.
type CreditPosition struct {
	CreatedAt time.Time `json:"c"`
	ID        int64     `json:"id"`
}

// CreditEntries are the organization's credit entries after the position, newest first, at
// most limit of them; and whether there are more.
func (l *Ledger) CreditEntries(ctx context.Context, organization string, after *CreditPosition, limit int) ([]CreditEntry, bool, error) {
	var rows []CreditEntry
	query := l.db.NewSelect().Model(&rows).Where("organization_id = ?", organization)
	if after != nil {
		query = query.Where("(created_at, id) < (?, ?)", after.CreatedAt, after.ID)
	}
	if err := query.Order("created_at DESC", "id DESC").Limit(limit + 1).Scan(ctx); err != nil {
		return nil, false, fmt.Errorf("ledger: credit entries: %w", err)
	}
	more := len(rows) > limit
	return rows[:min(len(rows), limit)], more, nil
}

// Standings are the organizations' standings, in the order asked, at the ledger's now: one
// read of their accounts, one of the tiers they name, and one of their organization
// windows of the current day and month. Nothing is written for an organization with no
// account.
func (l *Ledger) Standings(ctx context.Context, organizations []usage.StandingOrganization) ([]usage.Standing, error) {
	if len(organizations) == 0 {
		return []usage.Standing{}, nil
	}
	now := l.now()
	ids := make([]string, 0, len(organizations))
	names := make([]string, 0, len(organizations))
	wanted := make([]UsageWindow, 0, 2*len(organizations))
	for _, organization := range organizations {
		ids = append(ids, organization.OrganizationID)
		names = append(names, organization.OrganizationTier)
		ref := usage.SubjectRef{OrganizationID: organization.OrganizationID, Subject: usage.SubjectOrganization}
		wanted = append(wanted, windowOf(ref, usage.Daily, now), windowOf(ref, usage.Monthly, now))
	}
	var rows []Account
	if err := l.db.NewSelect().Model(&rows).Where("organization_id IN (?)", bun.In(ids)).Scan(ctx); err != nil {
		return nil, fmt.Errorf("ledger: standings: %w", err)
	}
	accounts := make(map[string]Account, len(rows))
	for _, row := range rows {
		accounts[row.OrganizationID] = row
	}
	tiers, err := tiersNamed(ctx, l.db, names)
	if err != nil {
		return nil, err
	}
	found, err := storedWindows(ctx, l.db, wanted)
	if err != nil {
		return nil, err
	}
	stored := make(map[windowKey]UsageWindow, len(found))
	for _, window := range found {
		stored[window.key()] = window
	}
	spend := func(window UsageWindow) usage.WindowSpend {
		if row, ok := stored[window.key()]; ok {
			return row.Spend()
		}
		return window.Spend()
	}
	standings := make([]usage.Standing, 0, len(organizations))
	for i, organization := range organizations {
		account, ok := accounts[organization.OrganizationID]
		if !ok {
			account = virtualAccount(organization.OrganizationID)
		}
		tier, own := tiers.resolve(organization.OrganizationTier)
		standings = append(standings, usage.Standing{
			OrganizationID:   organization.OrganizationID,
			OrganizationTier: organization.OrganizationTier,
			Tier:             tier,
			FallbackTier:     !own,
			Limits:           tier.Limits.Over(account.LimitOverrides),
			Overrides:        account.LimitOverrides,
			Status:           account.Status,
			StatusReason:     account.StatusReason,
			GrantedMicros:    account.GrantedMicros,
			SpentMicros:      account.SpentMicros,
			BalanceMicros:    account.GrantedMicros - account.SpentMicros,
			SpendWindows:     usage.SpendWindows{Daily: spend(wanted[2*i]), Monthly: spend(wanted[2*i+1])},
		})
	}
	return standings, nil
}

// debit adds the billable cost of events, the ones a batch just recorded, to their
// organizations' spend, in one statement taking the accounts' locks in organization order,
// as the windows' are, so two batches cannot deadlock. An event is debited only by the call
// that inserts it: one recorded by a binary that did not debit stays out of spent_micros
// until it is rebuilt as the billable SUM(cost_micros) of the organization's events.
func debit(ctx context.Context, tx bun.Tx, events []usage.Event) error {
	spent := map[string]int64{}
	for _, event := range events {
		if event.Billable && event.CostMicros > 0 {
			spent[event.OrganizationID] += event.CostMicros
		}
	}
	if len(spent) == 0 {
		return nil
	}
	rows := make([]Account, 0, len(spent))
	for organization, micros := range spent {
		rows = append(rows, Account{OrganizationID: organization, SpentMicros: micros})
	}
	slices.SortFunc(rows, func(a, b Account) int { return cmp.Compare(a.OrganizationID, b.OrganizationID) })
	if _, err := tx.NewInsert().Model(&rows).Column("organization_id", "spent_micros").
		On("CONFLICT (organization_id) DO UPDATE").
		Set("spent_micros = a.spent_micros + EXCLUDED.spent_micros").
		Set("updated_at = now()").Exec(ctx); err != nil {
		return fmt.Errorf("ledger: debit: %w", err)
	}
	return nil
}

// UsageFilter picks an organization's events for a usage listing. An empty field filters
// nothing; From is inclusive and To exclusive.
type UsageFilter struct {
	OrganizationID string
	From, To       time.Time
	Meter          string
	AppID          string
	APIKey         string
	UserID         string
	Category       string
	After          *UsagePosition
	Limit          int
}

// UsagePosition is where a page of events ends: events are newest first.
type UsagePosition struct {
	OccurredAt time.Time `json:"o"`
	UsageID    uuid.UUID `json:"id"`
}

// Usage is the events the filter picks, newest first, at most its limit; and whether there
// are more.
func (l *Ledger) Usage(ctx context.Context, filter UsageFilter) ([]UsageEvent, bool, error) {
	var rows []UsageEvent
	query := l.db.NewSelect().Model(&rows).Where("organization_id = ?", filter.OrganizationID)
	if !filter.From.IsZero() {
		query = query.Where("occurred_at >= ?", filter.From)
	}
	if !filter.To.IsZero() {
		query = query.Where("occurred_at < ?", filter.To)
	}
	for _, equal := range [][2]string{{"meter", filter.Meter}, {"app_id", filter.AppID}, {"api_key", filter.APIKey}, {"user_id", filter.UserID}} {
		if equal[1] != "" {
			query = query.Where("? = ?", bun.Ident(equal[0]), equal[1])
		}
	}
	if filter.Category != "" {
		// The second condition is the categories index's own predicate, so it can be used.
		query = query.Where("usage_categories @> ARRAY[?]::text[]", filter.Category).Where("usage_categories <> '{}'")
	}
	if after := filter.After; after != nil {
		query = query.Where("(occurred_at, usage_id) < (?, ?)", after.OccurredAt, after.UsageID)
	}
	if err := query.Order("occurred_at DESC", "usage_id DESC").Limit(filter.Limit + 1).Scan(ctx); err != nil {
		return nil, false, fmt.Errorf("ledger: usage: %w", err)
	}
	more := len(rows) > filter.Limit
	return rows[:min(len(rows), filter.Limit)], more, nil
}
