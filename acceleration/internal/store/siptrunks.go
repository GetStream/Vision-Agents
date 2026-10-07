package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"
	"github.com/uptrace/bun/driver/pgdriver"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// Postgres error codes this file tells apart.
const (
	foreignKeyViolation = "23503"
	uniqueViolation     = "23505"
)

// ErrNoSIPTrunk says the customer has no trunk with that id: none was made, it is another
// customer's, or it was deleted. One error for all three, so a caller learns nothing about
// anybody else's trunks.
var ErrNoSIPTrunk = errors.New("store: no such sip trunk")

// ErrSIPTrunkInUse says a trunk still has numbers on it, which have to be released first.
var ErrSIPTrunkInUse = errors.New("store: the sip trunk still has numbers on it")

// ErrNumberHeld says the customer already holds the number, bought or on a trunk.
var ErrNumberHeld = errors.New("store: the customer already holds this number")

// ErrNoNumber is what Number's error matches when the customer holds no such number, as
// opposed to the lookup failing.
var ErrNoNumber = errors.New("store: no such number")

// numberNotHeld keeps Number's message as it was while matching ErrNoNumber.
type numberNotHeld struct{ message string }

func (e numberNotHeld) Error() string        { return e.message }
func (e numberNotHeld) Is(target error) bool { return target == ErrNoNumber }

// SIPTrunk is a customer's own SIP trunk. The password is sealed by the caller before it
// reaches the store, with the customer and the trunk id as associated data, which is why
// the caller chooses the id.
type SIPTrunk struct {
	bun.BaseModel `bun:"table:sip_trunks,alias:st"`

	ID         string `bun:"id,pk"`
	CustomerID string `bun:"customer_id,notnull"`
	Name       string `bun:"name,notnull"`
	// Host is a bare hostname: no sip: and no port.
	Host      string `bun:"host,notnull"`
	Port      int    `bun:"port,notnull"`
	Transport string `bun:"transport,notnull"`
	Username  string `bun:"username,notnull"`
	// PasswordSealed is empty, at version 0, for a trunk whose password has to be set
	// again, which is how one arrives from a data move.
	PasswordSealed     []byte `bun:"password_sealed,notnull"`
	PasswordKEKVersion int    `bun:"password_kek_version,notnull"`
	// LateOffer says the trunk accepts an INVITE without SDP.
	LateOffer bool      `bun:"late_offer,notnull"`
	Codecs    []string  `bun:"codecs,array"`
	CreatedAt time.Time `bun:"created_at,notnull"`
	UpdatedAt time.Time `bun:"updated_at,notnull"`
}

// HasPassword says whether a password is stored that a key can open. A data move leaves
// version 0, and sealed bytes may survive it when the password was set again before.
func (t SIPTrunk) HasPassword() bool {
	return len(t.PasswordSealed) > 0 && t.PasswordKEKVersion > 0
}

// CreateSIPTrunk stores a new trunk.
func (s *Store) CreateSIPTrunk(ctx context.Context, trunk *SIPTrunk) error {
	if trunk.ID == "" || trunk.CustomerID == "" {
		return stack.Wrap(errors.New("store: a sip trunk needs an id and a customer"))
	}
	now := time.Now().UTC()
	trunk.CreatedAt, trunk.UpdatedAt = now, now
	if trunk.Codecs == nil {
		trunk.Codecs = []string{}
	}
	if _, err := s.db.NewInsert().Model(trunk).Exec(ctx); err != nil {
		return stack.Wrap(fmt.Errorf("store: create sip trunk: %w", err))
	}
	return nil
}

// SIPTrunks returns a customer's trunks, newest first.
func (s *Store) SIPTrunks(ctx context.Context, customerID string) ([]SIPTrunk, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}
	var trunks []SIPTrunk
	err := s.db.NewSelect().Model(&trunks).
		Where("customer_id = ?", customerID).
		Order("created_at DESC").
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: sip trunks: %w", err))
	}
	return trunks, nil
}

// SIPTrunk returns one of a customer's trunks.
func (s *Store) SIPTrunk(ctx context.Context, customerID, id string) (SIPTrunk, error) {
	var trunk SIPTrunk
	err := s.db.NewSelect().Model(&trunk).
		Where("customer_id = ?", customerID).
		Where("id = ?", id).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return SIPTrunk{}, stack.Wrap(ErrNoSIPTrunk)
	}
	if err != nil {
		return SIPTrunk{}, stack.Wrap(fmt.Errorf("store: sip trunk: %w", err))
	}
	return trunk, nil
}

// UpdateSIPTrunk writes every field of a trunk the customer holds.
func (s *Store) UpdateSIPTrunk(ctx context.Context, trunk *SIPTrunk) error {
	trunk.UpdatedAt = time.Now().UTC()
	if trunk.Codecs == nil {
		trunk.Codecs = []string{}
	}
	result, err := s.db.NewUpdate().Model(trunk).
		Column("name", "host", "port", "transport", "username", "password_sealed",
			"password_kek_version", "late_offer", "codecs", "updated_at").
		Where("customer_id = ?", trunk.CustomerID).
		Where("id = ?", trunk.ID).
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: update sip trunk: %w", err))
	}
	return oneRow(result, ErrNoSIPTrunk)
}

// DeleteSIPTrunk removes a trunk with no numbers on it. The foreign key from phone_numbers
// is what refuses one that has some, so a number added at the same moment cannot be left
// pointing at nothing.
func (s *Store) DeleteSIPTrunk(ctx context.Context, customerID, id string) error {
	result, err := s.db.NewDelete().Model((*SIPTrunk)(nil)).
		Where("customer_id = ?", customerID).
		Where("id = ?", id).
		Exec(ctx)
	if isViolation(err, foreignKeyViolation) {
		return stack.Wrap(ErrSIPTrunkInUse)
	}
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete sip trunk: %w", err))
	}
	return oneRow(result, ErrNoSIPTrunk)
}

// AddTrunkNumber records a number the customer says is on one of their own trunks.
//
// A customer looking a number up gets one row, so holding it twice, once bought and once
// on a trunk, would make which one a call goes through an accident. That is refused here;
// the unique index only covers numbers on trunks.
func (s *Store) AddTrunkNumber(ctx context.Context, number *PhoneNumber) error {
	if number.SIPTrunkID == "" {
		return stack.Wrap(errors.New("store: a number on a trunk needs the trunk"))
	}
	if _, err := s.SIPTrunk(ctx, number.CustomerID, number.SIPTrunkID); err != nil {
		return err
	}
	_, err := s.Number(ctx, number.CustomerID, number.E164)
	if err == nil {
		return stack.Wrap(ErrNumberHeld)
	}
	if !errors.Is(err, ErrNoNumber) {
		return err
	}
	number.Vendor = "sip_trunk"
	err = s.RecordNumber(ctx, number)
	if isViolation(err, uniqueViolation) {
		return stack.Wrap(ErrNumberHeld)
	}
	// The trunk was deleted after it was looked up above.
	if isViolation(err, foreignKeyViolation) {
		return stack.Wrap(ErrNoSIPTrunk)
	}
	return err
}

// oneRow turns an update or delete that matched nothing into missing.
func oneRow(result sql.Result, missing error) error {
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: rows affected: %w", err))
	}
	if affected == 0 {
		return stack.Wrap(missing)
	}
	return nil
}

// isViolation reports whether err is Postgres refusing a write with the given code.
func isViolation(err error, code string) bool {
	var refused pgdriver.Error
	return errors.As(err, &refused) && refused.Field('C') == code
}
