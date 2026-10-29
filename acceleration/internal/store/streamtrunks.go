package store

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// StreamTrunk is a Stream SIP trunk a customer's numbers of one vendor share in one Stream
// app, with the credentials the vendor presents to it.
type StreamTrunk struct {
	bun.BaseModel `bun:"table:stream_trunks,alias:st"`

	// ID is Stream's id for the trunk.
	ID         string `bun:"id,pk"`
	CustomerID string `bun:"customer_id,notnull"`
	// StreamAppPK is the app the trunk is in. Zero, stored as NULL, is the deployment's own.
	StreamAppPK int64  `bun:"stream_app_pk,nullzero"`
	Vendor      string `bun:"vendor,notnull"`
	// Numbers are the numbers the trunk has at Stream.
	Numbers        []string `bun:"numbers,array,notnull"`
	BridgeURI      string   `bun:"bridge_uri,notnull"`
	BridgeUsername string   `bun:"bridge_username,notnull"`
	// BridgePasswordSealed is sealed once for the trunk, since every number on it shares it.
	BridgePasswordSealed     []byte    `bun:"bridge_password_sealed,notnull"`
	BridgePasswordKEKVersion int       `bun:"bridge_password_kek_version,notnull"`
	CreatedAt                time.Time `bun:"created_at,nullzero,notnull,default:current_timestamp"`
}

// TrunkGroup is the trunks numbers may share: one customer's, of one vendor, in one app.
type TrunkGroup struct {
	CustomerID string
	// StreamAppPK is the app, zero for the deployment's.
	StreamAppPK int64
	Vendor      string
}

// NumberBridge is the trunk a number's calls go to, as kept on the number.
type NumberBridge struct {
	URI            string
	Username       string
	PasswordSealed []byte
	KEKVersion     int
}

// TrunkGroupTx is one trunk group's rows, read and written in a transaction that holds the
// group's lock.
type TrunkGroupTx struct {
	// ctx lives only as long as the InTrunkGroup call that made the TrunkGroupTx.
	ctx   context.Context
	tx    bun.Tx
	group TrunkGroup
}

// trunkGroupLockSeed keeps the trunk group lock apart from other advisory locks hashed from
// the same text.
const trunkGroupLockSeed = 0

// InTrunkGroup runs fn in a transaction that holds a lock on the group, so two numbers
// joining at once do not both see room on a trunk and both open a new one.
func (s *Store) InTrunkGroup(ctx context.Context, group TrunkGroup, fn func(ctx context.Context, tx *TrunkGroupTx) error) error {
	if group.CustomerID == "" || group.Vendor == "" {
		return stack.Wrap(errors.New("store: a trunk group needs a customer and a vendor"))
	}
	key := "stream_trunks:" + group.CustomerID + ":" + strconv.FormatInt(group.StreamAppPK, 10) + ":" + group.Vendor
	return s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if _, err := tx.ExecContext(ctx, "SELECT pg_advisory_xact_lock(hashtextextended(?, ?))", key, trunkGroupLockSeed); err != nil {
			return stack.Wrap(fmt.Errorf("store: lock trunk group: %w", err))
		}
		return fn(ctx, &TrunkGroupTx{ctx: ctx, tx: tx, group: group})
	})
}

// Trunks are the group's trunks, oldest first.
func (t *TrunkGroupTx) Trunks() ([]StreamTrunk, error) {
	var trunks []StreamTrunk
	err := t.tx.NewSelect().Model(&trunks).
		ApplyQueryBuilder(t.where).
		OrderExpr("created_at, id").
		Scan(t.ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: trunks: %w", err))
	}
	return trunks, nil
}

// SaveTrunk stores a trunk in the group, or replaces what is stored for it. A trunk of
// another group is refused rather than taken over.
func (t *TrunkGroupTx) SaveTrunk(trunk StreamTrunk) error {
	trunk.CustomerID, trunk.StreamAppPK, trunk.Vendor = t.group.CustomerID, t.group.StreamAppPK, t.group.Vendor
	result, err := t.tx.NewInsert().Model(&trunk).
		On("CONFLICT (id) DO UPDATE").
		Set("numbers = EXCLUDED.numbers").
		Set("bridge_uri = EXCLUDED.bridge_uri").
		Set("bridge_username = EXCLUDED.bridge_username").
		Set("bridge_password_sealed = EXCLUDED.bridge_password_sealed").
		Set("bridge_password_kek_version = EXCLUDED.bridge_password_kek_version").
		Where("st.customer_id = EXCLUDED.customer_id").
		Where("st.vendor = EXCLUDED.vendor").
		Where("st.stream_app_pk IS NOT DISTINCT FROM EXCLUDED.stream_app_pk").
		Exec(t.ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: save trunk: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: save trunk: %w", err))
	}
	if affected == 0 {
		return stack.Wrap(fmt.Errorf("store: trunk %s is another group's", trunk.ID))
	}
	return nil
}

// DeleteTrunk forgets one of the group's trunks.
func (t *TrunkGroupTx) DeleteTrunk(id string) error {
	_, err := t.tx.NewDelete().Model((*StreamTrunk)(nil)).
		Where("id = ?", id).
		ApplyQueryBuilder(t.where).
		Exec(t.ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: delete trunk: %w", err))
	}
	return nil
}

// Number is one of the customer's numbers held, locked until the transaction ends. A line
// being made for it in another app's group, which holds a different lock, then waits to
// record itself until this transaction is done, and finds the number released if it was.
func (t *TrunkGroupTx) Number(e164 string) (PhoneNumber, error) {
	return heldNumber(t.ctx, t.tx.NewSelect().For("UPDATE"), t.group.CustomerID, e164)
}

// ReleaseNumber marks one of the customer's numbers as given back.
func (t *TrunkGroupTx) ReleaseNumber(e164 string, at time.Time) error {
	return releaseNumber(t.ctx, t.tx, t.group.CustomerID, e164, at)
}

// SetLine records the trunk and rule a number held in the group is reached through.
func (t *TrunkGroupTx) SetLine(e164, trunkID, routeID string, bridge NumberBridge) error {
	result, err := t.tx.NewUpdate().Model((*PhoneNumber)(nil)).
		Set("stream_trunk_id = ?", trunkID).
		Set("stream_route_id = ?", routeID).
		Set("stream_app_pk = ?", nullablePin(t.group.StreamAppPK)).
		Set("bridge_uri = ?", nullable(bridge.URI)).
		Set("bridge_username = ?", nullable(bridge.Username)).
		Set("bridge_password_sealed = ?", nullableBytes(bridge.PasswordSealed)).
		Set("bridge_password_kek_version = ?", nullableVersion(bridge.KEKVersion)).
		Where("customer_id = ?", t.group.CustomerID).
		Where("vendor = ?", t.group.Vendor).
		Where("e164 = ?", e164).
		Where("released_at IS NULL").
		Exec(t.ctx)
	return t.heldOne(result, err, e164, "set line")
}

// ClearLine takes a number held in the group off its trunk and rule.
func (t *TrunkGroupTx) ClearLine(e164 string) error {
	result, err := t.tx.NewUpdate().Model((*PhoneNumber)(nil)).
		Set("stream_trunk_id = NULL").
		Set("stream_route_id = NULL").
		Set("attached_at = NULL").
		Set("config_id = NULL").
		Set("bridge_uri = NULL").
		Set("bridge_username = NULL").
		Set("bridge_password_sealed = NULL").
		Set("bridge_password_kek_version = NULL").
		Where("customer_id = ?", t.group.CustomerID).
		Where("vendor = ?", t.group.Vendor).
		Where("e164 = ?", e164).
		Where("released_at IS NULL").
		Exec(t.ctx)
	return t.heldOne(result, err, e164, "clear line")
}

// Detach stops a number held in the group from ringing its agent, and keeps its trunk and rule.
func (t *TrunkGroupTx) Detach(e164 string) error {
	result, err := t.tx.NewUpdate().Model((*PhoneNumber)(nil)).
		Set("attached_at = NULL").
		Set("config_id = NULL").
		Where("customer_id = ?", t.group.CustomerID).
		Where("vendor = ?", t.group.Vendor).
		Where("e164 = ?", e164).
		Where("released_at IS NULL").
		Exec(t.ctx)
	return t.heldOne(result, err, e164, "detach")
}

// heldOne is an update's error, or one saying the number is not held in the group.
func (t *TrunkGroupTx) heldOne(result sql.Result, err error, e164, what string) error {
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: %s: %w", what, err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: %s: %w", what, err))
	}
	if affected == 0 {
		return stack.Wrap(fmt.Errorf("store: %s is not a %s number %s holds", e164, t.group.Vendor, t.group.CustomerID))
	}
	return nil
}

func (t *TrunkGroupTx) where(q bun.QueryBuilder) bun.QueryBuilder {
	q = q.Where("customer_id = ?", t.group.CustomerID).Where("vendor = ?", t.group.Vendor)
	if t.group.StreamAppPK == 0 {
		return q.Where("stream_app_pk IS NULL")
	}
	return q.Where("stream_app_pk = ?", t.group.StreamAppPK)
}
