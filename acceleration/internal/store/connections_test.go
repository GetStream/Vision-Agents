//go:build integration

package store

import (
	"errors"
	"strings"
	"sync"
	"time"
)

// connection creates a connection to the acme built-in, app-owned unless change says
// otherwise, after seeding acme at revision 1.
func (s *StoreSuite) connection(customerID string, change func(*ConnectorConnection)) ConnectorConnection {
	s.seed(acmeManifest)
	connection := appConnection()
	connection.CustomerID = customerID
	if change != nil {
		change(connection)
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, testSchemes, connection))
	return *connection
}

// userOwned makes a connection the named user's.
func userOwned(userID string) func(*ConnectorConnection) {
	return func(connection *ConnectorConnection) {
		connection.OwnerType = OwnerUser
		connection.OwnerID = userID
	}
}

// attempt stores an open consent attempt for a connection under state.
func (s *StoreSuite) attempt(connection ConnectorConnection, state string) ConnectorAuthorizationAttempt {
	attempt := &ConnectorAuthorizationAttempt{
		ID:            newID(),
		CustomerID:    connection.CustomerID,
		ConnectionID:  connection.ID,
		Kind:          AttemptConsent,
		StateHash:     AuthorizationStateHash(state),
		AttemptSealed: []byte("sealed attempt"),
		KEKVersion:    1,
		ExpiresAt:     time.Now().Add(10 * time.Minute),
	}
	s.Require().NoError(s.store.CreateConnectorAuthorizationAttempt(s.ctx, attempt))
	return *attempt
}

// expire moves an attempt's expiry into the past, which no caller can ask for.
func (s *StoreSuite) expire(id string) {
	_, err := s.store.DB().ExecContext(s.ctx,
		"UPDATE connector_authorization_attempts SET expires_at = now() - interval '1 second' WHERE id = ?", id)
	s.Require().NoError(err)
}

// bind makes a live agent config of the customer's carry bindings, written as T20 will
// write them. The store has no write for the column until then.
func (s *StoreSuite) bind(customerID, bindings string) string {
	config := &AgentConfig{CustomerID: customerID, Name: "agent-" + newID()}
	s.Require().NoError(s.store.CreateAgentConfig(s.ctx, config))
	_, err := s.store.DB().ExecContext(s.ctx, "UPDATE agent_configs SET connectors = ?::jsonb WHERE id = ?", bindings, config.ID)
	s.Require().NoError(err)
	return config.ID
}

func fixedBinding(connectionID string) string {
	return `[{"name": "acme", "connector_id": "acme", "connection": {"type": "fixed", "connection_id": "` + connectionID + `"},` +
		` "tools": [{"name": "search", "schema_digest": "` + strings.Repeat("a", 64) + `"}], "required": true}]`
}

// insertRaw writes a connection row the way a writer that skips the store would, such as a
// data move's import, so only the database's own rules stand in its way.
func (s *StoreSuite) insertRaw(ownerType, ownerID string) error {
	_, err := s.store.DB().ExecContext(s.ctx,
		"INSERT INTO connector_connections (id, customer_id, connector_id, definition_revision, owner_type, owner_id, auth_scheme)"+
			" VALUES (?, 'acme-app', 'acme', 1, ?, ?, 'test_key')", newID(), ownerType, ownerID)
	return err
}

func (s *StoreSuite) TestACreatedConnectionIsPendingAtRevisionOneWithNoCredentials() {
	created := s.connection("acme-app", func(connection *ConnectorConnection) {
		connection.Inputs = map[string]string{"region": "eu"}
		connection.Label = "Support workspace"
	})

	found, err := s.store.ConnectorConnection(s.ctx, "acme-app", created.ID)
	s.Require().NoError(err)
	s.Equal(created.ID, found.ID)
	s.Equal("acme", found.ConnectorID)
	s.Equal(1, found.DefinitionRevision)
	s.Equal(OwnerApp, found.OwnerType)
	s.Empty(found.OwnerID)
	s.Equal("test_key", found.AuthScheme)
	s.Empty(found.TLSScheme)
	s.Equal(map[string]string{"region": "eu"}, found.Inputs)
	s.Equal(map[string]string{}, found.Metadata)
	s.Equal("Support workspace", found.Label)
	s.Equal(ConnectionPending, found.Status)
	s.Equal(1, found.Revision)
	s.Empty(found.CredentialsSealed)
	s.Zero(found.CredentialsKEKVersion)
	s.Equal([]string{}, found.GrantedScopes)
	s.True(created.CreatedAt.Equal(found.CreatedAt), "the row handed back is the row a read returns")
	s.Nil(found.DeletedAt)
}

func (s *StoreSuite) TestATLSSchemeIsStoredBesideTheAuthScheme() {
	created := s.connection("acme-app", func(connection *ConnectorConnection) { connection.TLSScheme = "test_mtls" })

	found, err := s.store.ConnectorConnection(s.ctx, "acme-app", created.ID)
	s.Require().NoError(err)
	s.Equal("test_mtls", found.TLSScheme)

	var stored *string
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT tls_scheme FROM connector_connections WHERE id = ?", created.ID).Scan(&stored))
	s.Require().NotNil(stored)

	plain := s.connection("acme-app", nil)
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT tls_scheme FROM connector_connections WHERE id = ?", plain.ID).Scan(&stored))
	s.Nil(stored, "no TLS scheme is NULL, not an empty name")
}

func (s *StoreSuite) TestARegisteredAuthSchemeItsConnectorDoesNotListIsRefused() {
	s.seed(acmeManifest)
	connection := appConnection()
	connection.AuthScheme = "test_unlisted"

	err := s.store.CreateConnectorConnection(s.ctx, testSchemes, connection)

	s.ErrorIs(err, ErrSchemeNotAllowed)
	s.ErrorContains(err, `acme revision 1 does not list auth scheme "test_unlisted"`)
}

func (s *StoreSuite) TestARegisteredTLSSchemeItsConnectorDoesNotListIsRefused() {
	s.seed(acmeManifest)
	connection := appConnection()
	connection.TLSScheme = "test_unlisted"

	err := s.store.CreateConnectorConnection(s.ctx, testSchemes, connection)

	s.ErrorIs(err, ErrSchemeNotAllowed)
	s.ErrorContains(err, `acme revision 1 does not list tls scheme "test_unlisted"`)
}

func (s *StoreSuite) TestTheDatabaseRefusesAnAppOwnedConnectionNamingAUser() {
	s.ErrorContains(s.insertRaw(OwnerApp, "alice"), "connector_connections_owner")
}

func (s *StoreSuite) TestTheDatabaseRefusesAUserOwnedConnectionNamingNoUser() {
	s.ErrorContains(s.insertRaw(OwnerUser, ""), "connector_connections_owner")
}

func (s *StoreSuite) TestTheDatabaseRefusesAThirdKindOfOwner() {
	s.ErrorContains(s.insertRaw("organization", "acme-org"), "connector_connections_owner")
}

func (s *StoreSuite) TestTheDatabaseTakesBothOwners() {
	s.NoError(s.insertRaw(OwnerApp, ""))
	s.NoError(s.insertRaw(OwnerUser, "alice"))
}

func (s *StoreSuite) TestAConnectionCannotPinARevisionThatDoesNotExist() {
	s.seed(acmeManifest)
	connection := appConnection()
	connection.DefinitionRevision = 2

	err := s.store.CreateConnectorConnection(s.ctx, testSchemes, connection)

	s.ErrorIs(err, ErrNoConnectorDefinition)
	s.ErrorContains(err, "acme revision 2")
}

func (s *StoreSuite) TestAConnectionCanPinAnOlderRevision() {
	s.seed(acmeManifest)
	s.seed(acmeChanged)
	connection := appConnection()

	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, testSchemes, connection))

	found, err := s.store.ConnectorConnection(s.ctx, "acme-app", connection.ID)
	s.Require().NoError(err)
	s.Equal(1, found.DefinitionRevision, "a new revision does not move a connection made from the old one")
}

func (s *StoreSuite) TestAConnectionCanPinTheCustomersOwnDefinition() {
	custom, err := s.store.CreateConnectorDefinition(s.ctx, "acme-app",
		parsed(s.T(), strings.Replace(acmeManifest, "id: acme", "id: custom_acme", 1)))
	s.Require().NoError(err)
	connection := appConnection()
	connection.ConnectorID = custom.ID

	s.NoError(s.store.CreateConnectorConnection(s.ctx, testSchemes, connection))
}

func (s *StoreSuite) TestAConnectionCannotPinAnotherCustomersDefinition() {
	_, err := s.store.CreateConnectorDefinition(s.ctx, "other-app",
		parsed(s.T(), strings.Replace(acmeManifest, "id: acme", "id: custom_acme", 1)))
	s.Require().NoError(err)
	connection := appConnection()
	connection.ConnectorID = "custom_acme"

	s.ErrorIs(s.store.CreateConnectorConnection(s.ctx, testSchemes, connection), ErrNoConnectorDefinition)
}

func (s *StoreSuite) TestAConnectionIsOnlyItsOwnCustomers() {
	mine := s.connection("acme-app", nil)

	_, err := s.store.ConnectorConnection(s.ctx, "other-app", mine.ID)
	s.ErrorIs(err, ErrNoConnectorConnection)

	listed, err := s.store.ConnectorConnectionsByOwner(s.ctx, "other-app", ConnectionFilter{OwnerType: OwnerApp})
	s.Require().NoError(err)
	s.Empty(listed)

	_, err = s.store.ConnectorConnectionReferenced(s.ctx, "other-app", mine.ID)
	s.ErrorIs(err, ErrNoConnectorConnection)

	s.ErrorIs(s.store.DeleteConnectorConnection(s.ctx, "other-app", mine.ID), ErrNoConnectorConnection)
	_, err = s.store.ConnectorConnection(s.ctx, "acme-app", mine.ID)
	s.NoError(err, "another customer's delete leaves it alone")
}

func (s *StoreSuite) TestListingByOwnerReturnsOnlyThatOwnersConnectionsNewestFirst() {
	first := s.connection("acme-app", userOwned("alice"))
	s.connection("acme-app", userOwned("bob"))
	s.connection("acme-app", nil)
	second := s.connection("acme-app", userOwned("alice"))

	listed, err := s.store.ConnectorConnectionsByOwner(s.ctx, "acme-app", ConnectionFilter{OwnerType: OwnerUser, OwnerID: "alice"})
	s.Require().NoError(err)
	s.Require().Len(listed, 2)
	s.Equal(second.ID, listed[0].ID)
	s.Equal(first.ID, listed[1].ID)

	apps, err := s.store.ConnectorConnectionsByOwner(s.ctx, "acme-app", ConnectionFilter{OwnerType: OwnerApp})
	s.Require().NoError(err)
	s.Len(apps, 1, "the app's own connections are not any user's, and the other way round")
}

func (s *StoreSuite) TestListingByOwnerNarrowsToOneConnector() {
	_, err := s.store.CreateConnectorDefinition(s.ctx, "acme-app",
		parsed(s.T(), strings.Replace(acmeManifest, "id: acme", "id: custom_acme", 1)))
	s.Require().NoError(err)
	s.connection("acme-app", nil)
	custom := s.connection("acme-app", func(connection *ConnectorConnection) { connection.ConnectorID = "custom_acme" })

	listed, err := s.store.ConnectorConnectionsByOwner(s.ctx, "acme-app", ConnectionFilter{OwnerType: OwnerApp, ConnectorID: "custom_acme"})
	s.Require().NoError(err)
	s.Require().Len(listed, 1)
	s.Equal(custom.ID, listed[0].ID)
}

func (s *StoreSuite) TestListingByOwnerPagesFromWhereTheLastPageEnded() {
	oldest := s.connection("acme-app", nil)
	middle := s.connection("acme-app", nil)
	newest := s.connection("acme-app", nil)

	page, err := s.store.ConnectorConnectionsByOwner(s.ctx, "acme-app", ConnectionFilter{OwnerType: OwnerApp, Limit: 1})
	s.Require().NoError(err)
	s.Require().Len(page, 2, "one more than the limit says there is another page")
	s.Equal(newest.ID, page[0].ID)

	next, err := s.store.ConnectorConnectionsByOwner(s.ctx, "acme-app", ConnectionFilter{
		OwnerType: OwnerApp, Limit: 2,
		After: &ConnectionPosition{CreatedAt: page[0].CreatedAt, ID: page[0].ID},
	})
	s.Require().NoError(err)
	s.Require().Len(next, 2)
	s.Equal(middle.ID, next[0].ID)
	s.Equal(oldest.ID, next[1].ID)
}

func (s *StoreSuite) TestADeletedConnectionCannotBeRead() {
	connection := s.connection("acme-app", nil)
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", connection.ID))

	_, err := s.store.ConnectorConnection(s.ctx, "acme-app", connection.ID)
	s.ErrorIs(err, ErrNoConnectorConnection)
}

func (s *StoreSuite) TestADeletedConnectionIsNotListed() {
	kept := s.connection("acme-app", userOwned("alice"))
	deleted := s.connection("acme-app", userOwned("alice"))
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", deleted.ID))

	listed, err := s.store.ConnectorConnectionsByOwner(s.ctx, "acme-app", ConnectionFilter{OwnerType: OwnerUser, OwnerID: "alice"})
	s.Require().NoError(err)
	s.Require().Len(listed, 1)
	s.Equal(kept.ID, listed[0].ID)
}

func (s *StoreSuite) TestADeletedConnectionIsNotAskedAboutReferences() {
	connection := s.connection("acme-app", nil)
	s.bind("acme-app", fixedBinding(connection.ID))
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", connection.ID))

	_, err := s.store.ConnectorConnectionReferenced(s.ctx, "acme-app", connection.ID)
	s.ErrorIs(err, ErrNoConnectorConnection)
}

func (s *StoreSuite) TestADeletedConnectionCannotBeDeletedAgain() {
	connection := s.connection("acme-app", nil)
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", connection.ID))

	s.ErrorIs(s.store.DeleteConnectorConnection(s.ctx, "acme-app", connection.ID), ErrNoConnectorConnection)
}

func (s *StoreSuite) TestAnUnforcedDeleteRefusesABoundConnectionAndLeavesItLive() {
	connection := s.connection("acme-app", nil)
	s.bind("acme-app", fixedBinding(connection.ID))

	s.ErrorIs(s.store.DeleteUnboundConnectorConnection(s.ctx, "acme-app", connection.ID), ErrConnectorConnectionBound)
	found, err := s.store.ConnectorConnection(s.ctx, "acme-app", connection.ID)
	s.Require().NoError(err, "a refused delete leaves the connection live")
	s.Equal(ConnectionPending, found.Status)
}

func (s *StoreSuite) TestAnUnforcedDeleteDeletesAnUnboundConnection() {
	connection := s.connection("acme-app", nil)
	other := s.connection("acme-app", nil)
	s.bind("acme-app", fixedBinding(other.ID))

	s.Require().NoError(s.store.DeleteUnboundConnectorConnection(s.ctx, "acme-app", connection.ID))
	_, err := s.store.ConnectorConnection(s.ctx, "acme-app", connection.ID)
	s.ErrorIs(err, ErrNoConnectorConnection)
}

func (s *StoreSuite) TestAnUnforcedDeleteOfAMissingConnectionIsNotFound() {
	deleted := s.connection("acme-app", nil)
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", deleted.ID))
	others := s.connection("other-app", nil)
	s.bind("other-app", fixedBinding(others.ID))

	s.ErrorIs(s.store.DeleteUnboundConnectorConnection(s.ctx, "acme-app", newID()), ErrNoConnectorConnection)
	s.ErrorIs(s.store.DeleteUnboundConnectorConnection(s.ctx, "acme-app", deleted.ID), ErrNoConnectorConnection)
	s.ErrorIs(s.store.DeleteUnboundConnectorConnection(s.ctx, "acme-app", others.ID), ErrNoConnectorConnection,
		"another customer's connection is not found, bound or not")
}

// The race a check before the delete had: a bind that commits after the check and before the
// UPDATE. LOCK TABLE in SHARE mode holds every UPDATE of the table at its ROW EXCLUSIVE lock
// and lets a plain SELECT through (https://www.postgresql.org/docs/current/explicit-locking.html,
// table 13.2). Postgres takes that lock while it parses the UPDATE and only then the snapshot
// the UPDATE runs with (exec_simple_query in src/backend/tcop/postgres.c: execution does not
// reuse "a snapshot that has been acquired before locking any of the tables mentioned in the
// query"). So a delete that checked in a SELECT of its own has checked when it stops here,
// and one that checks inside the UPDATE has not. A wait on the connection's row lock comes
// after the snapshot, so a bind that commits during it is missed (AI-889).
func (s *StoreSuite) TestABindThatCommitsBeforeTheDeleteTakesItsSnapshotStopsIt() {
	connection := s.connection("acme-app", nil)
	locker := s.router()
	held, err := locker.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer held.Rollback() //nolint:errcheck // after the commit below there is nothing to roll back
	_, err = held.ExecContext(s.ctx, "LOCK TABLE connector_connections IN SHARE MODE")
	s.Require().NoError(err)

	deleter := s.router()
	deleted := make(chan error, 1)
	go func() { deleted <- deleter.DeleteUnboundConnectorConnection(s.ctx, "acme-app", connection.ID) }()
	// The two seconds and the ten milliseconds are assertTheWaitEnded's (credentials_test.go).
	s.Require().Eventually(func() bool { return s.waitingForTable("connector_connections") == 1 },
		2*time.Second, 10*time.Millisecond, "the delete waits for the table lock")
	s.bind("acme-app", fixedBinding(connection.ID))
	s.Require().NoError(held.Commit())

	s.ErrorIs(<-deleted, ErrConnectorConnectionBound)
	_, err = s.store.ConnectorConnection(s.ctx, "acme-app", connection.ID)
	s.NoError(err, "the bound connection stays live")
}

// waitingForTable counts the callers in this database queued for a lock on the table
// (https://www.postgresql.org/docs/current/view-pg-locks.html).
func (s *StoreSuite) waitingForTable(table string) int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, `
SELECT count(*) FROM pg_locks
WHERE locktype = 'relation' AND relation = ?::regclass AND NOT granted
  AND database = (SELECT oid FROM pg_database WHERE datname = current_database())`, table).Scan(&count))
	return count
}

func (s *StoreSuite) TestDeletingAConnectionDropsItsCredentials() {
	connection := s.connection("acme-app", nil)
	// The revisioned save that writes credentials is T8's; this is the state it leaves.
	_, err := s.store.DB().ExecContext(s.ctx,
		"UPDATE connector_connections SET credentials_sealed = 'sealed credentials', credentials_kek_version = 1, status = ?, expires_at = now() WHERE id = ?",
		ConnectionConnected, connection.ID)
	s.Require().NoError(err)

	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", connection.ID))

	var sealed []byte
	var version int
	var status string
	var expires *time.Time
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT credentials_sealed, credentials_kek_version, status, expires_at FROM connector_connections WHERE id = ?",
		connection.ID).Scan(&sealed, &version, &status, &expires))
	s.Empty(sealed)
	s.Zero(version)
	s.Equal(ConnectionDisconnected, status)
	s.Nil(expires)
}

func (s *StoreSuite) TestADeletedConnectionsAttemptCannotBeReadOrConsumed() {
	connection := s.connection("acme-app", nil)
	attempt := s.attempt(connection, "state-of-a-deleted-connection")
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", connection.ID))

	_, err := s.store.ConnectorAuthorizationAttemptByState(s.ctx, "state-of-a-deleted-connection")
	s.ErrorIs(err, ErrNoAuthorizationAttempt)
	_, err = s.store.ConnectorAuthorizationAttemptByID(s.ctx, attempt.ID)
	s.ErrorIs(err, ErrNoAuthorizationAttempt)
	_, err = s.store.ConsumeConnectorAuthorizationAttempt(s.ctx, "state-of-a-deleted-connection")
	s.ErrorIs(err, ErrNoAuthorizationAttempt)
}

func (s *StoreSuite) TestADeletedConnectionTakesNoNewAttempt() {
	connection := s.connection("acme-app", nil)
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", connection.ID))

	err := s.store.CreateConnectorAuthorizationAttempt(s.ctx, &ConnectorAuthorizationAttempt{
		ID: newID(), CustomerID: "acme-app", ConnectionID: connection.ID, Kind: AttemptReconnect,
		StateHash: AuthorizationStateHash("too-late"), AttemptSealed: []byte("sealed"), KEKVersion: 1,
		ExpiresAt: time.Now().Add(time.Minute),
	})
	s.ErrorIs(err, ErrNoConnectorConnection)
}

func (s *StoreSuite) TestAFixedBindingReferencesItsConnection() {
	connection := s.connection("acme-app", nil)
	s.bind("acme-app", fixedBinding(connection.ID))

	referenced, err := s.store.ConnectorConnectionReferenced(s.ctx, "acme-app", connection.ID)
	s.Require().NoError(err)
	s.True(referenced)
}

func (s *StoreSuite) TestAnUnboundConnectionIsNotReferenced() {
	connection := s.connection("acme-app", nil)
	other := s.connection("acme-app", nil)
	s.bind("acme-app", fixedBinding(other.ID))

	referenced, err := s.store.ConnectorConnectionReferenced(s.ctx, "acme-app", connection.ID)
	s.Require().NoError(err)
	s.False(referenced)
}

func (s *StoreSuite) TestASessionBindingReferencesNoConnection() {
	connection := s.connection("acme-app", userOwned("alice"))
	// A session binding names no connection; it carries an id here only to show the type is
	// what decides.
	s.bind("acme-app", `[{"name": "acme", "connector_id": "acme", "connection": {"type": "session", "connection_id": "`+connection.ID+`"}}]`)

	referenced, err := s.store.ConnectorConnectionReferenced(s.ctx, "acme-app", connection.ID)
	s.Require().NoError(err)
	s.False(referenced)
}

func (s *StoreSuite) TestADeletedConfigReferencesNothing() {
	connection := s.connection("acme-app", nil)
	configID := s.bind("acme-app", fixedBinding(connection.ID))
	s.Require().NoError(s.store.DeleteAgentConfig(s.ctx, "acme-app", configID))

	referenced, err := s.store.ConnectorConnectionReferenced(s.ctx, "acme-app", connection.ID)
	s.Require().NoError(err)
	s.False(referenced)
}

func (s *StoreSuite) TestAnotherCustomersConfigReferencesNothingOfMine() {
	connection := s.connection("acme-app", nil)
	s.bind("other-app", fixedBinding(connection.ID))

	referenced, err := s.store.ConnectorConnectionReferenced(s.ctx, "acme-app", connection.ID)
	s.Require().NoError(err)
	s.False(referenced)
}

func (s *StoreSuite) TestAnAttemptIsFoundByItsStateAndByItsID() {
	connection := s.connection("acme-app", nil)
	created := s.attempt(connection, "the-state")

	byState, err := s.store.ConnectorAuthorizationAttemptByState(s.ctx, "the-state")
	s.Require().NoError(err)
	s.Equal(created.ID, byState.ID)
	s.Equal(AttemptConsent, byState.Kind)
	s.Equal([]byte("sealed attempt"), byState.AttemptSealed)
	s.Equal(1, byState.KEKVersion)
	s.Equal(connection.ID, byState.ConnectionID)
	s.Nil(byState.ConsumedAt)

	byID, err := s.store.ConnectorAuthorizationAttemptByID(s.ctx, created.ID)
	s.Require().NoError(err)
	s.Equal(created.ID, byID.ID)
}

func (s *StoreSuite) TestAnAttemptIsNotFoundByAnotherState() {
	s.attempt(s.connection("acme-app", nil), "the-state")

	_, err := s.store.ConnectorAuthorizationAttemptByState(s.ctx, "a-guessed-state")
	s.ErrorIs(err, ErrNoAuthorizationAttempt)
}

func (s *StoreSuite) TestTheStateItselfIsNotStored() {
	s.attempt(s.connection("acme-app", nil), "bearer-state")

	var stored int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT count(*) FROM connector_authorization_attempts WHERE to_jsonb(connector_authorization_attempts)::text LIKE '%bearer-state%'").Scan(&stored))
	s.Zero(stored)
}

func (s *StoreSuite) TestAConsumedAttemptCannotBeConsumedOrReadAgain() {
	created := s.attempt(s.connection("acme-app", nil), "the-state")

	consumed, err := s.store.ConsumeConnectorAuthorizationAttempt(s.ctx, "the-state")
	s.Require().NoError(err)
	s.Equal(created.ID, consumed.ID)
	s.NotNil(consumed.ConsumedAt)

	_, err = s.store.ConsumeConnectorAuthorizationAttempt(s.ctx, "the-state")
	s.ErrorIs(err, ErrNoAuthorizationAttempt, "a replayed callback is refused")
	_, err = s.store.ConnectorAuthorizationAttemptByState(s.ctx, "the-state")
	s.ErrorIs(err, ErrNoAuthorizationAttempt)
	_, err = s.store.ConnectorAuthorizationAttemptByID(s.ctx, created.ID)
	s.ErrorIs(err, ErrNoAuthorizationAttempt)
}

func (s *StoreSuite) TestAnExpiredAttemptCannotBeReadOrConsumed() {
	created := s.attempt(s.connection("acme-app", nil), "the-state")
	s.expire(created.ID)

	_, err := s.store.ConnectorAuthorizationAttemptByState(s.ctx, "the-state")
	s.ErrorIs(err, ErrNoAuthorizationAttempt)
	_, err = s.store.ConnectorAuthorizationAttemptByID(s.ctx, created.ID)
	s.ErrorIs(err, ErrNoAuthorizationAttempt)
	_, err = s.store.ConsumeConnectorAuthorizationAttempt(s.ctx, "the-state")
	s.ErrorIs(err, ErrNoAuthorizationAttempt)
}

func (s *StoreSuite) TestCreatingAnAttemptDeletesOnlyTheCustomersExpiredOnes() {
	mine := s.connection("acme-app", nil)
	theirs := s.connection("other-app", nil)
	expired := s.attempt(mine, "expired")
	open := s.attempt(mine, "open")
	theirsExpired := s.attempt(theirs, "theirs")
	s.expire(expired.ID)
	s.expire(theirsExpired.ID)

	fresh := s.attempt(mine, "new")

	var left []string
	s.Require().NoError(s.store.DB().NewSelect().Table("connector_authorization_attempts").
		Column("id").Scan(s.ctx, &left))
	s.ElementsMatch([]string{open.ID, theirsExpired.ID, fresh.ID}, left,
		"another customer's expired attempt waits for that customer's next one")
}

func (s *StoreSuite) TestAnAttemptForAnotherCustomersConnectionIsRefused() {
	theirs := s.connection("other-app", nil)

	err := s.store.CreateConnectorAuthorizationAttempt(s.ctx, &ConnectorAuthorizationAttempt{
		ID: newID(), CustomerID: "acme-app", ConnectionID: theirs.ID, Kind: AttemptConsent,
		StateHash: AuthorizationStateHash("mine"), AttemptSealed: []byte("sealed"), KEKVersion: 1,
		ExpiresAt: time.Now().Add(time.Minute),
	})
	s.ErrorIs(err, ErrNoConnectorConnection)
}

func (s *StoreSuite) TestCallbacksRacingWithOneStateConsumeItExactlyOnce() {
	s.attempt(s.connection("acme-app", nil), "raced-state")

	// Each callback is a router of its own with a pool of its own, connected before they
	// all start at once, so the consumes overlap in Postgres rather than queue on one pool.
	const callbacks = 16
	routers := make([]*Store, callbacks)
	for i := range routers {
		router, err := Open(s.dsn)
		s.Require().NoError(err)
		s.T().Cleanup(func() { router.Close() })
		s.Require().NoError(router.Ping(s.ctx))
		routers[i] = router
	}
	errs := make([]error, callbacks)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i, router := range routers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			_, errs[i] = router.ConsumeConnectorAuthorizationAttempt(s.ctx, "raced-state")
		}()
	}
	close(start)
	wg.Wait()

	won := 0
	for _, err := range errs {
		if err == nil {
			won++
			continue
		}
		s.True(errors.Is(err, ErrNoAuthorizationAttempt), "a callback that loses is told the state is used: %v", err)
	}
	s.Equal(1, won)
}

func (s *StoreSuite) TestAHandoffReplacesTheBlobItWasReadWithOnce() {
	created := s.attempt(s.connection("acme-app", nil), "the-state")

	s.Require().NoError(s.store.HandOffConnectorAuthorizationAttempt(s.ctx, created.ID, created.AttemptSealed, []byte("handed off"), 2))

	handedOff, err := s.store.ConnectorAuthorizationAttemptByID(s.ctx, created.ID)
	s.Require().NoError(err)
	s.Equal([]byte("handed off"), handedOff.AttemptSealed)
	s.Equal(2, handedOff.KEKVersion)
	s.Nil(handedOff.ConsumedAt, "the callback still consumes it")
	err = s.store.HandOffConnectorAuthorizationAttempt(s.ctx, created.ID, created.AttemptSealed, []byte("again"), 2)
	s.ErrorIs(err, ErrNoAuthorizationAttempt, "the blob it was read with is gone")
}

func (s *StoreSuite) TestAnExpiredAttemptCannotBeHandedOff() {
	created := s.attempt(s.connection("acme-app", nil), "the-state")
	s.expire(created.ID)

	err := s.store.HandOffConnectorAuthorizationAttempt(s.ctx, created.ID, created.AttemptSealed, []byte("handed off"), 1)

	s.ErrorIs(err, ErrNoAuthorizationAttempt)
}

func (s *StoreSuite) TestAConsumedAttemptCannotBeHandedOff() {
	created := s.attempt(s.connection("acme-app", nil), "the-state")
	_, err := s.store.ConsumeConnectorAuthorizationAttempt(s.ctx, "the-state")
	s.Require().NoError(err)

	err = s.store.HandOffConnectorAuthorizationAttempt(s.ctx, created.ID, created.AttemptSealed, []byte("handed off"), 1)

	s.ErrorIs(err, ErrNoAuthorizationAttempt)
}

func (s *StoreSuite) TestADeletedConnectionsAttemptCannotBeHandedOff() {
	connection := s.connection("acme-app", nil)
	created := s.attempt(connection, "the-state")
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", connection.ID))

	err := s.store.HandOffConnectorAuthorizationAttempt(s.ctx, created.ID, created.AttemptSealed, []byte("handed off"), 1)

	s.ErrorIs(err, ErrNoAuthorizationAttempt)
}

func (s *StoreSuite) TestHandoffsRacingFromOneReadReplaceTheBlobExactlyOnce() {
	created := s.attempt(s.connection("acme-app", nil), "raced-state")

	// As TestCallbacksRacingWithOneStateConsumeItExactlyOnce: a router and a pool per
	// handoff, so the updates overlap in Postgres.
	const handoffs = 16
	routers := make([]*Store, handoffs)
	for i := range routers {
		router, err := Open(s.dsn)
		s.Require().NoError(err)
		s.T().Cleanup(func() { router.Close() })
		s.Require().NoError(router.Ping(s.ctx))
		routers[i] = router
	}
	errs := make([]error, handoffs)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i, router := range routers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			errs[i] = router.HandOffConnectorAuthorizationAttempt(s.ctx, created.ID, created.AttemptSealed, []byte("handed off "+newID()), 1)
		}()
	}
	close(start)
	wg.Wait()

	won := 0
	for _, err := range errs {
		if err == nil {
			won++
			continue
		}
		s.True(errors.Is(err, ErrNoAuthorizationAttempt), "a handoff that loses is told the attempt is handed off: %v", err)
	}
	s.Equal(1, won)
}
