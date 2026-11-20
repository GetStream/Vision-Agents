package store

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// chatChannelTurn is the lease on a channel's running turn (TakeChatChannelTurn).
type chatChannelTurn struct {
	bun.BaseModel `bun:"table:chat_channel_turns,alias:cht"`

	StreamAppPK int64     `bun:"stream_app_pk,pk"`
	CID         string    `bun:"cid,pk"`
	TurnHolder  string    `bun:"turn_holder,notnull"`
	TurnUntil   time.Time `bun:"turn_until,notnull"`
}

// TakeChatChannelTurn leases the one running turn of the channel cid of the Stream app app, a
// channel of a type an agent config answers in, to holder until until, when no turn holds it
// or the one that did ran out. False is a turn another holder has, as TakeChannelThreadTurn
// says of a thread channel. One statement, so two routers asking at once get one turn.
func (s *Store) TakeChatChannelTurn(ctx context.Context, app int64, cid, holder string, until time.Time) (bool, error) {
	if cid == "" || holder == "" {
		return false, stack.Wrap(errors.New("store: a channel and a holder are required"))
	}
	result, err := s.db.NewInsert().Model(&chatChannelTurn{StreamAppPK: app, CID: cid, TurnHolder: holder, TurnUntil: until.UTC()}).
		On("CONFLICT (stream_app_pk, cid) DO UPDATE").
		Set("turn_holder = EXCLUDED.turn_holder").
		Set("turn_until = EXCLUDED.turn_until").
		Where("cht.turn_until < now() OR cht.turn_holder = EXCLUDED.turn_holder").
		Exec(ctx)
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: take chat channel turn: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return false, stack.Wrap(fmt.Errorf("store: take chat channel turn: %w", err))
	}
	return affected == 1, nil
}

// ReleaseChatChannelTurn ends holder's lease on a channel's turn, so the next turn need not
// wait for it to run out. A lease another holder took since is left alone.
func (s *Store) ReleaseChatChannelTurn(ctx context.Context, app int64, cid, holder string) error {
	_, err := s.db.NewDelete().Model((*chatChannelTurn)(nil)).
		Where("stream_app_pk = ? AND cid = ? AND turn_holder = ?", app, cid, holder).
		Exec(ctx)
	return stack.Wrap(err)
}
