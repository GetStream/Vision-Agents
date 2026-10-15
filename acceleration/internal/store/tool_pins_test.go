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
// TestConnectionsPinnedUnderTheSameGrantReadOnlyTheirOwnPins: connections connected before
// connected_at was kept all have a nil one, so the grant alone does not tell their pins apart.
func (s *StoreSuite) TestConnectionsPinnedUnderTheSameGrantReadOnlyTheirOwnPins() {
	shared := time.Now().UTC().Truncate(time.Microsecond)
	for name, grant := range map[string]*time.Time{"legacy": nil, "dated": &shared} {
		s.Run(name, func() {
			alice := s.connection("acme-app", userOwned("alice-"+name))
			bob := s.connection("acme-app", userOwned("bob-"+name))
			_, err := s.store.PinConnectorTools(s.ctx, alice.ID, grant, map[string]string{"search": digestOf("a")})
			s.Require().NoError(err)
			_, err = s.store.PinConnectorTools(s.ctx, bob.ID, grant, map[string]string{"search": digestOf("b")})
			s.Require().NoError(err)

			alices, err := s.store.ConnectorToolPins(s.ctx, alice.ID, grant)
			s.Require().NoError(err)
			bobs, err := s.store.ConnectorToolPins(s.ctx, bob.ID, grant)
			s.Require().NoError(err)

			s.Equal(map[string]string{"search": digestOf("a")}, alices)
			s.Equal(map[string]string{"search": digestOf("b")}, bobs)
		})
	}
}

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

// consented is connection with a grant that began now, as a consent leaves it, read back.
func (s *StoreSuite) consented(connection ConnectorConnection) ConnectorConnection {
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE connector_connections SET connected_at = ? WHERE id = ?",
		time.Now().UTC().Truncate(time.Microsecond), connection.ID)
	s.Require().NoError(err)
	read, err := s.store.ConnectorConnection(s.ctx, connection.CustomerID, connection.ID)
	s.Require().NoError(err)
	s.Require().NotNil(read.ConnectedAt)
	return read
}

// reapprovals are connection's tool_reapproved audit rows.
func (s *StoreSuite) reapprovals(connection ConnectorConnection) []ConnectorAuditEvent {
	events, err := s.store.ConnectorAuditEvents(s.ctx, connection.CustomerID, AuditFilter{ConnectionID: connection.ID})
	s.Require().NoError(err)
	var kept []ConnectorAuditEvent
	for _, event := range events {
		if event.Action == AuditToolReapproved {
			kept = append(kept, event)
		}
	}
	return kept
}

func (s *StoreSuite) TestAReapprovalMovesThePinsAndRecordsOneAuditRowNamingThem() {
	connection := s.consented(s.connection("acme-app", userOwned("alice")))
	_, err := s.store.PinConnectorTools(s.ctx, connection.ID, connection.ConnectedAt,
		map[string]string{"search": digestOf("a"), "fetch": digestOf("a")})
	s.Require().NoError(err)

	err = s.store.ReapproveConnectorTools(s.ctx, connection, "request-1", []ApprovedTool{
		{Name: "search", SchemaDigest: digestOf("b"), PreviousSchemaDigest: digestOf("a")},
		{Name: "fetch", SchemaDigest: digestOf("c"), PreviousSchemaDigest: digestOf("a")},
	})
	s.Require().NoError(err)

	pins, err := s.store.ConnectorToolPins(s.ctx, connection.ID, connection.ConnectedAt)
	s.Require().NoError(err)
	s.Equal(map[string]string{"search": digestOf("b"), "fetch": digestOf("c")}, pins)
	audited := s.reapprovals(connection)
	s.Require().Len(audited, 1)
	s.Equal("request-1", audited[0].RequestID)
	s.Equal(OwnerUser, audited[0].OwnerType)
	s.Equal([]ApprovedTool{
		{Name: "fetch", SchemaDigest: digestOf("c"), PreviousSchemaDigest: digestOf("a")},
		{Name: "search", SchemaDigest: digestOf("b"), PreviousSchemaDigest: digestOf("a")},
	}, audited[0].Tools)
	s.Nil(audited[0].Credential)
}

// TestAReapprovalOfAPinThatMovedSinceItWasReadWritesNothing: another approve came first.
func (s *StoreSuite) TestAReapprovalOfAPinThatMovedSinceItWasReadWritesNothing() {
	connection := s.consented(s.connection("acme-app", userOwned("alice")))
	_, err := s.store.PinConnectorTools(s.ctx, connection.ID, connection.ConnectedAt, map[string]string{"search": digestOf("c")})
	s.Require().NoError(err)

	err = s.store.ReapproveConnectorTools(s.ctx, connection, "", []ApprovedTool{
		{Name: "search", SchemaDigest: digestOf("b"), PreviousSchemaDigest: digestOf("a")},
	})

	s.Require().ErrorIs(err, ErrConnectorToolPinMoved)
	pins, err := s.store.ConnectorToolPins(s.ctx, connection.ID, connection.ConnectedAt)
	s.Require().NoError(err)
	s.Equal(map[string]string{"search": digestOf("c")}, pins)
	s.Empty(s.reapprovals(connection))
}

// TestAReapprovalAfterTheGrantMovedWritesNothing: a consent began a new grant after the
// connection was read, so the pin read belongs to a grant that has ended.
func (s *StoreSuite) TestAReapprovalAfterTheGrantMovedWritesNothing() {
	connection := s.consented(s.connection("acme-app", userOwned("alice")))
	_, err := s.store.PinConnectorTools(s.ctx, connection.ID, connection.ConnectedAt, map[string]string{"search": digestOf("a")})
	s.Require().NoError(err)
	_, err = s.store.DB().ExecContext(s.ctx, "UPDATE connector_connections SET connected_at = ? WHERE id = ?",
		connection.ConnectedAt.Add(time.Minute), connection.ID)
	s.Require().NoError(err)

	err = s.store.ReapproveConnectorTools(s.ctx, connection, "", []ApprovedTool{
		{Name: "search", SchemaDigest: digestOf("b"), PreviousSchemaDigest: digestOf("a")},
	})

	s.Require().ErrorIs(err, ErrConnectorToolPinMoved)
	pins, err := s.store.ConnectorToolPins(s.ctx, connection.ID, connection.ConnectedAt)
	s.Require().NoError(err)
	s.Equal(map[string]string{"search": digestOf("a")}, pins)
	s.Empty(s.reapprovals(connection))
}

// TestAReapprovalIsAllOrNothing: fetch's pin moved, so search's, which had not, stays too.
func (s *StoreSuite) TestAReapprovalIsAllOrNothing() {
	connection := s.consented(s.connection("acme-app", userOwned("alice")))
	_, err := s.store.PinConnectorTools(s.ctx, connection.ID, connection.ConnectedAt,
		map[string]string{"search": digestOf("a"), "fetch": digestOf("c")})
	s.Require().NoError(err)

	err = s.store.ReapproveConnectorTools(s.ctx, connection, "", []ApprovedTool{
		{Name: "search", SchemaDigest: digestOf("b"), PreviousSchemaDigest: digestOf("a")},
		{Name: "fetch", SchemaDigest: digestOf("b"), PreviousSchemaDigest: digestOf("a")},
	})

	s.Require().ErrorIs(err, ErrConnectorToolPinMoved)
	pins, err := s.store.ConnectorToolPins(s.ctx, connection.ID, connection.ConnectedAt)
	s.Require().NoError(err)
	s.Equal(map[string]string{"search": digestOf("a"), "fetch": digestOf("c")}, pins)
	s.Empty(s.reapprovals(connection))
}

// TestOfTwoReapprovalsFromOnePinOnlyOneIsWritten: neither replaces the other's unseen.
func (s *StoreSuite) TestOfTwoReapprovalsFromOnePinOnlyOneIsWritten() {
	connection := s.consented(s.connection("acme-app", userOwned("alice")))
	_, err := s.store.PinConnectorTools(s.ctx, connection.ID, connection.ConnectedAt, map[string]string{"search": digestOf("a")})
	s.Require().NoError(err)
	const approves = 8
	errs := make([]error, approves)
	var wg sync.WaitGroup
	for i := range approves {
		wg.Go(func() {
			errs[i] = s.store.ReapproveConnectorTools(s.ctx, connection, "", []ApprovedTool{
				{Name: "search", SchemaDigest: digestOf(string(rune('b' + i))), PreviousSchemaDigest: digestOf("a")},
			})
		})
	}
	wg.Wait()

	won := -1
	for i, err := range errs {
		if err == nil {
			s.Equal(-1, won, "a second approve was written")
			won = i
			continue
		}
		s.ErrorIs(err, ErrConnectorToolPinMoved)
	}
	s.Require().NotEqual(-1, won)
	pins, err := s.store.ConnectorToolPins(s.ctx, connection.ID, connection.ConnectedAt)
	s.Require().NoError(err)
	s.Equal(map[string]string{"search": digestOf(string(rune('b' + won)))}, pins)
	s.Len(s.reapprovals(connection), 1)
}

// TestAToolUnpinnedForTheGrantIsPinnedByAReapproval, over a pin of an earlier grant, and a
// session's first pin since it was read is not replaced.
func (s *StoreSuite) TestAToolUnpinnedForTheGrantIsPinnedByAReapproval() {
	connection := s.consented(s.connection("acme-app", userOwned("alice")))
	earlier := connection.ConnectedAt.Add(-time.Minute)
	_, err := s.store.PinConnectorTools(s.ctx, connection.ID, &earlier, map[string]string{"search": digestOf("a")})
	s.Require().NoError(err)
	_, err = s.store.PinConnectorTools(s.ctx, connection.ID, connection.ConnectedAt, map[string]string{"fetch": digestOf("c")})
	s.Require().NoError(err)

	pinned := s.store.ReapproveConnectorTools(s.ctx, connection, "", []ApprovedTool{{Name: "search", SchemaDigest: digestOf("b")}})
	raced := s.store.ReapproveConnectorTools(s.ctx, connection, "", []ApprovedTool{{Name: "fetch", SchemaDigest: digestOf("b")}})

	s.Require().NoError(pinned)
	s.Require().ErrorIs(raced, ErrConnectorToolPinMoved)
	pins, err := s.store.ConnectorToolPins(s.ctx, connection.ID, connection.ConnectedAt)
	s.Require().NoError(err)
	s.Equal(map[string]string{"search": digestOf("b"), "fetch": digestOf("c")}, pins)
}

func (s *StoreSuite) TestADeletedConnectionIsNotReapproved() {
	connection := s.consented(s.connection("acme-app", userOwned("alice")))
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", connection.ID))

	err := s.store.ReapproveConnectorTools(s.ctx, connection, "", []ApprovedTool{{Name: "search", SchemaDigest: digestOf("b")}})

	s.Require().ErrorIs(err, ErrNoConnectorConnection)
	s.Empty(s.reapprovals(connection))
}
