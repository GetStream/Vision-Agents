package store

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// TakeChatChannelTurn leases the one running turn of the channel cid of the Stream app app, a
// channel of a type an agent config answers in, to holder until until, when no turn holds it
// or the one that did ran out. False is a turn another holder has, as TakeChannelThreadTurn
// says of a thread channel. One statement, so two routers asking at once get one turn.
func (s *Store) TakeChatChannelTurn(ctx context.Context, app int64, cid, holder string, until time.Time) (bool, error) {
	if cid == "" || holder == "" {
		return false, stack.Wrap(errors.New("store: a channel and a holder are required"))
	}
	result, err := s.db.NewRaw(
		"INSERT INTO chat_channel_turns (stream_app_pk, cid, turn_holder, turn_until) VALUES (?, ?, ?, ?) "+
			"ON CONFLICT (stream_app_pk, cid) DO UPDATE "+
			"SET turn_holder = EXCLUDED.turn_holder, turn_until = EXCLUDED.turn_until "+
			"WHERE chat_channel_turns.turn_until < now() OR chat_channel_turns.turn_holder = EXCLUDED.turn_holder",
		app, cid, holder, until.UTC()).Exec(ctx)
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
	_, err := s.db.NewRaw("DELETE FROM chat_channel_turns WHERE stream_app_pk = ? AND cid = ? AND turn_holder = ?",
		app, cid, holder).Exec(ctx)
	return stack.Wrap(err)
}
