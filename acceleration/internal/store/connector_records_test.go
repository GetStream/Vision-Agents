//go:build integration

package store

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
)

// invocation records one call of connection's, started at offset past s.base.
func (s *StoreSuite) invocation(connection ConnectorConnection, offset time.Duration, errorType string) ConnectorInvocation {
	row := &ConnectorInvocation{
		CustomerID: connection.CustomerID, ConnectionID: connection.ID, ConnectorID: connection.ConnectorID,
		ConfigID: "config", Binding: "acme", Tool: "search", SessionID: "session",
		StartedAt: s.base.Add(offset), LatencyMs: 12, ErrorType: errorType,
	}
	s.Require().NoError(s.store.RecordConnectorInvocation(s.ctx, row))
	return *row
}

// granted gives a connection stored credentials, as a consent leaves it.
func (s *StoreSuite) granted(connection ConnectorConnection) {
	_, err := s.store.DB().ExecContext(s.ctx,
		"UPDATE connector_connections SET credentials_sealed = 'sealed credentials', credentials_kek_version = 1, status = ? WHERE id = ?",
		ConnectionConnected, connection.ID)
	s.Require().NoError(err)
}

func (s *StoreSuite) TestAConnectionsInvocationsAreListedNewestFirstAndPaged() {
	connection := s.connection("acme-app", nil)
	first := s.invocation(connection, time.Second, "")
	second := s.invocation(connection, 2*time.Second, InvocationExternalServer)
	third := s.invocation(connection, 3*time.Second, InvocationDenied)

	page, err := s.store.ConnectorInvocations(s.ctx, "acme-app", connection.ID, 2, nil)
	s.Require().NoError(err)
	s.Require().Len(page, 3, "one more than the limit says there is another page")
	rest, err := s.store.ConnectorInvocations(s.ctx, "acme-app", connection.ID, 2,
		&InvocationPosition{StartedAt: page[1].StartedAt, ID: page[1].ID})
	s.Require().NoError(err)

	s.Equal([]string{third.ID, second.ID}, []string{page[0].ID, page[1].ID})
	s.Require().Len(rest, 1)
	s.Equal(first.ID, rest[0].ID)
	s.Equal(InvocationDenied, page[0].ErrorType)
	s.Equal(int64(12), page[0].LatencyMs)
}

func (s *StoreSuite) TestAnInvocationIsOnlyItsOwnCustomersAndConnections() {
	connection := s.connection("acme-app", nil)
	other := s.connection("acme-app", nil)
	s.invocation(connection, time.Second, "")

	ofOther, err := s.store.ConnectorInvocations(s.ctx, "acme-app", other.ID, 0, nil)
	s.Require().NoError(err)
	ofStranger, err := s.store.ConnectorInvocations(s.ctx, "other-app", connection.ID, 0, nil)
	s.Require().NoError(err)

	s.Empty(ofOther)
	s.Empty(ofStranger)
}

func (s *StoreSuite) TestAnInvocationWithAnUnknownErrorTypeIsRefused() {
	connection := s.connection("acme-app", nil)

	err := s.store.RecordConnectorInvocation(s.ctx, &ConnectorInvocation{
		CustomerID: "acme-app", ConnectionID: connection.ID, ConnectorID: "acme", Tool: "search",
		StartedAt: s.base, ErrorType: "timeout",
	})

	s.ErrorContains(err, `"timeout" is not an invocation error type`)
}

func (s *StoreSuite) TestTheAuditListsTheCustomersRowsNewestFirstWholeOrForOneConnection() {
	one := s.connection("acme-app", nil)
	two := s.connection("acme-app", nil)
	for _, event := range []*ConnectorAuditEvent{
		{CustomerID: "acme-app", ConnectionID: one.ID, ConnectorID: "acme", OwnerType: OwnerApp, Action: AuditGrantCreated, Reason: AuditReasonConsent, Revision: 2},
		{CustomerID: "acme-app", ConnectionID: two.ID, ConnectorID: "acme", OwnerType: OwnerApp, Action: AuditGrantCreated, Reason: AuditReasonCredentials, Revision: 2},
		{CustomerID: "acme-app", ConnectionID: one.ID, ConnectorID: "acme", OwnerType: OwnerApp, Action: AuditGrantRefreshed, Revision: 3},
		{CustomerID: "other-app", ConnectionID: one.ID, ConnectorID: "acme", OwnerType: OwnerApp, Action: AuditGrantRevoked},
	} {
		s.Require().NoError(s.store.RecordConnectorAudit(s.ctx, event))
	}

	all, err := s.store.ConnectorAuditEvents(s.ctx, "acme-app", AuditFilter{})
	s.Require().NoError(err)
	ofOne, err := s.store.ConnectorAuditEvents(s.ctx, "acme-app", AuditFilter{ConnectionID: one.ID, Limit: 1})
	s.Require().NoError(err)
	rest, err := s.store.ConnectorAuditEvents(s.ctx, "acme-app", AuditFilter{ConnectionID: one.ID, Limit: 1,
		After: &AuditPosition{CreatedAt: ofOne[0].CreatedAt, ID: ofOne[0].ID}})
	s.Require().NoError(err)

	s.Len(all, 3, "another customer's row is not listed")
	s.Equal(AuditGrantRefreshed, all[0].Action)
	s.Require().Len(ofOne, 2)
	s.Equal(AuditGrantRefreshed, ofOne[0].Action)
	s.Require().Len(rest, 1)
	s.Equal(AuditGrantCreated, rest[0].Action)
	s.Equal(AuditReasonConsent, rest[0].Reason)
}

// TestAnAuditRowKeepsTheFingerprintsOfItsTokens: a grant row's fingerprints come back with
// it, and a row written without them comes back without.
func (s *StoreSuite) TestAnAuditRowKeepsTheFingerprintsOfItsTokens() {
	connection := s.connection("acme-app", nil)
	expires := s.base.Add(12 * time.Hour)
	refreshed := &ConnectorAuditEvent{CustomerID: "acme-app", ConnectionID: connection.ID, ConnectorID: "acme",
		OwnerType: OwnerApp, Action: AuditGrantRefreshed, Revision: 3,
		Credential: AuditCredential(core.CredentialChange{
			Previous: core.CredentialFingerprints{Access: "0a0a0a0a", Refresh: "1b1b1b1b"},
			Current:  core.CredentialFingerprints{Access: "2c2c2c2c", Refresh: "3d3d3d3d", AccessExpiresAt: expires},
		})}
	s.Require().NoError(s.store.RecordConnectorAudit(s.ctx, refreshed))
	s.Require().NoError(s.store.RecordConnectorAudit(s.ctx, &ConnectorAuditEvent{CustomerID: "acme-app",
		ConnectionID: connection.ID, ConnectorID: "acme", OwnerType: OwnerApp, Action: AuditProxyCall}))

	rows, err := s.store.ConnectorAuditEvents(s.ctx, "acme-app", AuditFilter{ConnectionID: connection.ID})

	s.Require().NoError(err)
	s.Require().Len(rows, 2)
	s.Nil(rows[0].Credential, "the proxy call names no token")
	got := rows[1].Credential
	s.Require().NotNil(got)
	s.Equal(rows[1].ID, got.AuditID)
	s.Equal("0a0a0a0a", got.PreviousAccessFingerprint)
	s.Equal("2c2c2c2c", got.AccessFingerprint)
	s.Equal("1b1b1b1b", got.PreviousRefreshFingerprint)
	s.Equal("3d3d3d3d", got.RefreshFingerprint)
	s.True(got.Rotated)
	s.True(expires.Equal(got.AccessExpiresAt))
	s.True(got.RefreshExpiresAt.IsZero(), "an expiry the provider did not give stays unknown")
}

func (s *StoreSuite) TestAnAuditRowWithAnUnknownActionIsRefused() {
	err := s.store.RecordConnectorAudit(s.ctx, &ConnectorAuditEvent{
		CustomerID: "acme-app", ConnectionID: "one", ConnectorID: "acme", OwnerType: OwnerApp, Action: "grant_renamed",
	})

	s.ErrorContains(err, `"grant_renamed" is not an audit action`)
}

func (s *StoreSuite) TestTheAuditTakesTheProxyAndExportRowsThatComeLater() {
	status, latency := 200, int64(31)
	for _, event := range []*ConnectorAuditEvent{
		{CustomerID: "acme-app", ConnectionID: "one", ConnectorID: "acme", OwnerType: OwnerApp, Action: AuditProxyCall,
			StatusCode: &status, LatencyMs: &latency, Target: "api.acme.test"},
		{CustomerID: "acme-app", ConnectionID: "one", ConnectorID: "acme", OwnerType: OwnerApp, Action: AuditTokenExport},
	} {
		s.Require().NoError(s.store.RecordConnectorAudit(s.ctx, event))
	}

	found, err := s.store.ConnectorAuditEvents(s.ctx, "acme-app", AuditFilter{})

	s.Require().NoError(err)
	s.Require().Len(found, 2)
	s.Equal("api.acme.test", found[1].Target)
	s.Equal(200, *found[1].StatusCode)
}

func (s *StoreSuite) TestDeletingAUsersConnectionsRemovesEveryOneOfThemAndNothingElse() {
	live := s.connection("acme-app", userOwned("alice"))
	s.granted(live)
	gone := s.connection("acme-app", userOwned("alice"))
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", gone.ID))
	pending := s.connection("acme-app", userOwned("alice"))
	s.attempt(pending, "state-of-alices-consent")
	s.invocation(live, time.Second, "")
	bob := s.connection("acme-app", userOwned("bob"))
	app := s.connection("acme-app", nil)
	elsewhere := s.connection("other-app", userOwned("alice"))

	deleted, err := s.store.DeleteUserConnectorConnections(s.ctx, "acme-app", "alice")

	s.Require().NoError(err)
	s.ElementsMatch([]DeletedConnection{
		{ID: live.ID, ConnectorID: "acme", OwnerType: OwnerUser, HadGrant: true},
		{ID: gone.ID, ConnectorID: "acme", OwnerType: OwnerUser},
		{ID: pending.ID, ConnectorID: "acme", OwnerType: OwnerUser},
	}, deleted)
	var left int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT count(*) FROM connector_connections WHERE customer_id = 'acme-app' AND owner_id = 'alice'").Scan(&left))
	s.Zero(left, "the soft deleted row, which still held the owner and account ids, is gone too")
	var invocations, attempts int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT count(*) FROM connector_invocations WHERE connection_id = ?", live.ID).Scan(&invocations))
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT count(*) FROM connector_authorization_attempts WHERE connection_id = ?", pending.ID).Scan(&attempts))
	s.Zero(invocations)
	s.Zero(attempts)
	for _, kept := range []ConnectorConnection{bob, app, elsewhere} {
		_, err := s.store.ConnectorConnection(s.ctx, kept.CustomerID, kept.ID)
		s.NoError(err, "another user's, the app's and another customer's stay")
	}
}

// TestDeletingAUsersConnectionsUnlinksTheirAuditFromTheUser: the rows stay, with no id that
// leads back to the user; another user's keep theirs.
func (s *StoreSuite) TestDeletingAUsersConnectionsUnlinksTheirAuditFromTheUser() {
	hers := s.connection("acme-app", userOwned("alice"))
	his := s.connection("acme-app", userOwned("bob"))
	for _, connection := range []ConnectorConnection{hers, his} {
		s.Require().NoError(s.store.RecordConnectorAudit(s.ctx, &ConnectorAuditEvent{
			CustomerID: "acme-app", ConnectionID: connection.ID, ConnectorID: "acme", OwnerType: OwnerUser,
			Action: AuditGrantCreated, Reason: AuditReasonConsent, Revision: 2,
			RequestID: "request", SessionID: "session", AttemptID: "attempt",
		}))
	}

	_, err := s.store.DeleteUserConnectorConnections(s.ctx, "acme-app", "alice")

	s.Require().NoError(err)
	kept, err := s.store.ConnectorAuditEvents(s.ctx, "acme-app", AuditFilter{ConnectionID: hers.ID})
	s.Require().NoError(err)
	s.Require().Len(kept, 1, "the row stays")
	s.Empty(kept[0].RequestID)
	s.Empty(kept[0].SessionID)
	s.Empty(kept[0].AttemptID)
	s.Equal(AuditGrantCreated, kept[0].Action)
	others, err := s.store.ConnectorAuditEvents(s.ctx, "acme-app", AuditFilter{ConnectionID: his.ID})
	s.Require().NoError(err)
	s.Require().Len(others, 1)
	s.Equal("session", others[0].SessionID)
}

func (s *StoreSuite) TestDeletingTheConnectionsOfAUserWithNoneDeletesNothing() {
	s.connection("acme-app", userOwned("bob"))

	deleted, err := s.store.DeleteUserConnectorConnections(s.ctx, "acme-app", "alice")

	s.Require().NoError(err)
	s.Empty(deleted)
}

func (s *StoreSuite) TestAConnectionIsUsedByTheFixedBindingsOfLiveConfigsThatNameIt() {
	bound := s.connection("acme-app", nil)
	unbound := s.connection("acme-app", nil)
	config := s.bind("acme-app", fixedBinding(bound.ID))
	s.bind("acme-app", `[{"name": "acme", "connector_id": "acme", "connection": {"type": "session", "connection_id": "`+unbound.ID+`"}}]`)
	deleted := s.bind("acme-app", fixedBinding(bound.ID))
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE agent_configs SET deleted_at = now() WHERE id = ?", deleted)
	s.Require().NoError(err)
	s.bind("other-app", fixedBinding(bound.ID))
	s.bind("acme-app", `{"not": "a list"}`)

	uses, err := s.store.ConnectorConnectionUses(s.ctx, "acme-app", []string{bound.ID, unbound.ID})

	s.Require().NoError(err)
	s.Require().Len(uses[bound.ID], 1, "a deleted config and another customer's do not use it")
	s.Equal(config, uses[bound.ID][0].ConfigID)
	s.Equal("acme", uses[bound.ID][0].Binding)
	s.Empty(uses[unbound.ID], "a session binding names no connection")
}
