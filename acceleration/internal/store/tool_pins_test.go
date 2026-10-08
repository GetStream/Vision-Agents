//go:build integration

package store

import (
	"strings"
	"sync"
	"time"
)

// digestOf is a synthetic schema digest, 64 hex characters of c.
func digestOf(c string) string {
	return strings.Repeat(c, 64)
}

// pinRows is how many pins connection id has, under any grant.
func (s *StoreSuite) pinRows(id string) int {
	var rows int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT count(*) FROM connector_tool_pins WHERE connection_id = ?", id).Scan(&rows))
	return rows
}

func (s *StoreSuite) TestAToolIsPinnedOncePerGrant() {
	connection := s.connection("acme-app", userOwned("alice"))
	grant := time.Now().UTC().Truncate(time.Microsecond)
	first, err := s.store.PinConnectorTools(s.ctx, connection.ID, &grant, map[string]string{"search": digestOf("a")})
	s.Require().NoError(err)
	var pinnedAt time.Time
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT pinned_at FROM connector_tool_pins WHERE connection_id = ?", connection.ID).Scan(&pinnedAt))

	again, err := s.store.PinConnectorTools(s.ctx, connection.ID, &grant, map[string]string{"search": digestOf("b")})
	s.Require().NoError(err)

	s.Equal(map[string]string{"search": digestOf("a")}, first)
	s.Equal(map[string]string{"search": digestOf("a")}, again, "the first pin stands")
	var later time.Time
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT pinned_at FROM connector_tool_pins WHERE connection_id = ?", connection.ID).Scan(&later))
	s.True(pinnedAt.Equal(later), "the pin was not written again")
}

func (s *StoreSuite) TestAPinOfAnEarlierGrantIsNoneAndIsReplaced() {
	connection := s.connection("acme-app", userOwned("alice"))
	before := time.Now().UTC().Truncate(time.Microsecond)
	reconnected := before.Add(time.Minute)
	_, err := s.store.PinConnectorTools(s.ctx, connection.ID, &before, map[string]string{"search": digestOf("a")})
	s.Require().NoError(err)

	unpinned, err := s.store.ConnectorToolPins(s.ctx, connection.ID, &reconnected)
	s.Require().NoError(err)
	pinned, err := s.store.PinConnectorTools(s.ctx, connection.ID, &reconnected, map[string]string{"search": digestOf("b")})
	s.Require().NoError(err)
	earlier, err := s.store.ConnectorToolPins(s.ctx, connection.ID, &before)
	s.Require().NoError(err)

	s.Empty(unpinned, "a reconnect is a new trust event")
	s.Equal(map[string]string{"search": digestOf("b")}, pinned)
	s.Empty(earlier)
	s.Equal(1, s.pinRows(connection.ID))
}

// TestAPinOfALaterGrantIsKept: a router that read the connection before a reconnect another
// router has already pinned under does not take the pin back to the old grant.
func (s *StoreSuite) TestAPinOfALaterGrantIsKept() {
	connection := s.connection("acme-app", userOwned("alice"))
	before := time.Now().UTC().Truncate(time.Microsecond)
	reconnected := before.Add(time.Minute)
	_, err := s.store.PinConnectorTools(s.ctx, connection.ID, &reconnected, map[string]string{"search": digestOf("b")})
	s.Require().NoError(err)

	stale, err := s.store.PinConnectorTools(s.ctx, connection.ID, &before, map[string]string{"search": digestOf("a")})
	s.Require().NoError(err)
	current, err := s.store.ConnectorToolPins(s.ctx, connection.ID, &reconnected)
	s.Require().NoError(err)

	s.Empty(stale)
	s.Equal(map[string]string{"search": digestOf("b")}, current)
}

// TestAConnectionConnectedBeforeConnectedAtWasKeptIsPinnedToo: such a connection has no
// connected_at, and its pins hold until a consent gives it one.
func (s *StoreSuite) TestAConnectionConnectedBeforeConnectedAtWasKeptIsPinnedToo() {
	connection := s.connection("acme-app", userOwned("alice"))
	_, err := s.store.PinConnectorTools(s.ctx, connection.ID, nil, map[string]string{"search": digestOf("a")})
	s.Require().NoError(err)

	again, err := s.store.PinConnectorTools(s.ctx, connection.ID, nil, map[string]string{"search": digestOf("b")})
	s.Require().NoError(err)
	consented := time.Now().UTC().Truncate(time.Microsecond)
	after, err := s.store.ConnectorToolPins(s.ctx, connection.ID, &consented)
	s.Require().NoError(err)

	s.Equal(map[string]string{"search": digestOf("a")}, again)
	s.Empty(after)
}

func (s *StoreSuite) TestSessionsPinningAtOnceAllReadTheFirstPin() {
	connection := s.connection("acme-app", userOwned("alice"))
	grant := time.Now().UTC().Truncate(time.Microsecond)
	const sessions = 8
	read := make([]map[string]string, sessions)
	var wg sync.WaitGroup
	for i := range sessions {
		wg.Go(func() {
			pins, err := s.store.PinConnectorTools(s.ctx, connection.ID, &grant,
				map[string]string{"search": digestOf(string(rune('a' + i))), "fetch": digestOf(string(rune('a' + i)))})
			s.NoError(err)
			read[i] = pins
		})
	}
	wg.Wait()

	for i := range sessions {
		s.Equal(read[0], read[i], "session %d", i)
	}
	s.Len(read[0], 2)
	s.Equal(2, s.pinRows(connection.ID))
}

func (s *StoreSuite) TestDeletingAConnectionDropsItsToolPins() {
	forced := s.connection("acme-app", userOwned("alice"))
	unbound := s.connection("acme-app", userOwned("alice"))
	for _, id := range []string{forced.ID, unbound.ID} {
		_, err := s.store.PinConnectorTools(s.ctx, id, nil, map[string]string{"search": digestOf("a")})
		s.Require().NoError(err)
	}

	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", forced.ID))
	s.Require().NoError(s.store.DeleteUnboundConnectorConnection(s.ctx, "acme-app", unbound.ID))

	s.Zero(s.pinRows(forced.ID))
	s.Zero(s.pinRows(unbound.ID))
}

func (s *StoreSuite) TestARefusedDeleteKeepsTheToolPins() {
	connection := s.connection("acme-app", nil)
	s.bind("acme-app", fixedBinding(connection.ID))
	_, err := s.store.PinConnectorTools(s.ctx, connection.ID, nil, map[string]string{"search": digestOf("a")})
	s.Require().NoError(err)

	s.Require().ErrorIs(s.store.DeleteUnboundConnectorConnection(s.ctx, "acme-app", connection.ID), ErrConnectorConnectionBound)

	s.Equal(1, s.pinRows(connection.ID))
}

func (s *StoreSuite) TestAnOffboardedUsersConnectionsTakeTheirToolPinsWithThem() {
	connection := s.connection("acme-app", userOwned("alice"))
	_, err := s.store.PinConnectorTools(s.ctx, connection.ID, nil, map[string]string{"search": digestOf("a")})
	s.Require().NoError(err)

	_, err = s.store.DeleteUserConnectorConnections(s.ctx, "acme-app", "alice")
	s.Require().NoError(err)

	s.Zero(s.pinRows(connection.ID))
}
