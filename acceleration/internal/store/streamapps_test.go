//go:build integration

package store

import (
	"context"
	"encoding/json"
	"time"
)

// registration is an app with the keys given, each sealed as nothing in particular: what
// the store keeps is opaque to it.
func registration(customer string, app int64, keys ...string) StreamAppRegistration {
	sealed := make([]StreamAppKey, 0, len(keys))
	for _, key := range keys {
		sealed = append(sealed, StreamAppKey{APIKey: key, Sealed: []byte("sealed " + key), KEKVersion: 1, Last4: "1234"})
	}
	return StreamAppRegistration{
		CustomerID: customer, StreamAppPK: app, Keys: sealed, PrimaryKey: keys[0],
		UpdatedBy: "test", VerifiedAt: time.Now(),
	}
}

func revision(n int64) *int64 { return &n }

func (s *StoreSuite) TestAStreamAppIsRegisteredWithItsKeys() {
	stored, err := s.store.PutStreamApp(s.ctx, registration("4242", 4242, "first", "second"))
	s.Require().NoError(err)

	s.Equal(StreamAppConnected, stored.State)
	s.Equal(int64(1), stored.Revision)
	s.Equal("first", stored.PrimaryKey)
	s.Require().Len(stored.Keys, 2)
	read, err := s.store.StreamApp(s.ctx, "4242")
	s.Require().NoError(err)
	s.Equal([]byte("sealed first"), read.Keys[0].Sealed)
	s.Equal(StreamAppKeyActive, read.Keys[0].Status)
	s.NotNil(read.Keys[0].VerifiedAt)
}

func (s *StoreSuite) TestACustomerWithNoAppHasNone() {
	_, err := s.store.StreamApp(s.ctx, "nobody")
	s.ErrorIs(err, ErrNoStreamApp)
}

func (s *StoreSuite) TestAnAPIKeyBelongsToOneApp() {
	_, err := s.store.PutStreamApp(s.ctx, registration("4242", 4242, "shared"))
	s.Require().NoError(err)

	_, err = s.store.PutStreamApp(s.ctx, registration("7", 7, "shared"))

	s.ErrorIs(err, ErrStreamAppKeyTaken)
	held, err := s.store.StreamApp(s.ctx, "4242")
	s.Require().NoError(err)
	s.Equal([]byte("sealed shared"), held.Keys[0].Sealed, "the first app's key is left as it was")
	_, err = s.store.StreamApp(s.ctx, "7")
	s.ErrorIs(err, ErrNoStreamApp, "nothing of the refused registration is kept")
}

func (s *StoreSuite) TestOneStreamAppHasOneCustomer() {
	_, err := s.store.PutStreamApp(s.ctx, registration("4242", 4242, "first"))
	s.Require().NoError(err)

	_, err = s.store.PutStreamApp(s.ctx, registration("someone-else", 4242, "other"))

	s.ErrorIs(err, ErrStreamAppTaken)
}

func (s *StoreSuite) TestReplacingKeysBumpsTheRevision() {
	_, err := s.store.PutStreamApp(s.ctx, registration("4242", 4242, "first", "second"))
	s.Require().NoError(err)

	rotated := registration("4242", 4242, "second", "third")
	rotated.ExpectedRevision = revision(1)
	stored, err := s.store.PutStreamApp(s.ctx, rotated)

	s.Require().NoError(err)
	s.Equal(int64(2), stored.Revision)
	var held []string
	for _, key := range stored.Keys {
		held = append(held, key.APIKey)
	}
	s.ElementsMatch([]string{"second", "third"}, held, "a key left out is deleted")
}

func (s *StoreSuite) TestAStaleRevisionIsAConflict() {
	_, err := s.store.PutStreamApp(s.ctx, registration("4242", 4242, "first"))
	s.Require().NoError(err)
	stale := registration("4242", 4242, "second")
	stale.ExpectedRevision = revision(0)

	_, err = s.store.PutStreamApp(s.ctx, stale)

	s.ErrorIs(err, ErrStreamAppChanged)
	held, err := s.store.StreamApp(s.ctx, "4242")
	s.Require().NoError(err)
	s.Equal("first", held.PrimaryKey)
}

func (s *StoreSuite) TestARevisionNamedForAnAppNeverRegisteredIsAConflict() {
	stale := registration("4242", 4242, "first")
	stale.ExpectedRevision = revision(3)

	_, err := s.store.PutStreamApp(s.ctx, stale)

	s.ErrorIs(err, ErrStreamAppChanged)
}

func (s *StoreSuite) TestARegistrationNeedsItsPrimaryKeyAmongItsKeys() {
	bad := registration("4242", 4242, "first")
	bad.PrimaryKey = "elsewhere"

	_, err := s.store.PutStreamApp(s.ctx, bad)

	s.ErrorContains(err, "not one of the app's keys")
}

func (s *StoreSuite) TestDisconnectingDeletesTheSecretsAndLeavesATombstone() {
	_, err := s.store.PutStreamApp(s.ctx, registration("4242", 4242, "first", "second"))
	s.Require().NoError(err)

	stored, err := s.store.DisconnectStreamApp(s.ctx, "4242", revision(1), "test")

	s.Require().NoError(err)
	s.Equal(StreamAppDisconnected, stored.State)
	s.Empty(stored.PrimaryKey)
	s.Empty(stored.Keys)
	s.Equal(int64(2), stored.Revision)
	var sealed int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT count(*) FROM stream_app_keys WHERE customer_id = '4242'").Scan(&sealed))
	s.Zero(sealed)
	_, err = s.store.StreamAppByAPIKey(s.ctx, "first")
	s.ErrorIs(err, ErrNoStreamApp)
}

func (s *StoreSuite) TestRegisteringAgainConnectsADisconnectedApp() {
	_, err := s.store.PutStreamApp(s.ctx, registration("4242", 4242, "first"))
	s.Require().NoError(err)
	_, err = s.store.DisconnectStreamApp(s.ctx, "4242", nil, "test")
	s.Require().NoError(err)

	again := registration("4242", 4242, "second")
	again.ExpectedRevision = revision(2)
	stored, err := s.store.PutStreamApp(s.ctx, again)

	s.Require().NoError(err)
	s.Equal(StreamAppConnected, stored.State)
	s.Equal(int64(3), stored.Revision)
}

func (s *StoreSuite) TestForgettingRemovesTheTombstone() {
	_, err := s.store.PutStreamApp(s.ctx, registration("4242", 4242, "first"))
	s.Require().NoError(err)
	_, err = s.store.DisconnectStreamApp(s.ctx, "4242", nil, "test")
	s.Require().NoError(err)

	s.Require().NoError(s.store.ForgetStreamApp(s.ctx, "4242"))

	_, err = s.store.StreamApp(s.ctx, "4242")
	s.ErrorIs(err, ErrNoStreamApp)
}

func (s *StoreSuite) TestAKeyIsFoundByItsAPIKey() {
	_, err := s.store.PutStreamApp(s.ctx, registration("4242", 4242, "first", "second"))
	s.Require().NoError(err)

	found, err := s.store.StreamAppByAPIKey(s.ctx, "second")

	s.Require().NoError(err)
	s.Equal("4242", found.CustomerID)
	_, held := found.Key("second")
	s.True(held)
}

func (s *StoreSuite) TestARejectedKeyIsMarkedAndRegisteringAgainRestoresIt() {
	_, err := s.store.PutStreamApp(s.ctx, registration("4242", 4242, "first"))
	s.Require().NoError(err)

	s.Require().NoError(s.store.RejectStreamAppKey(s.ctx, "4242", "first", "Stream answered 401", time.Now()))

	held, err := s.store.StreamApp(s.ctx, "4242")
	s.Require().NoError(err)
	s.Equal(StreamAppKeyRejected, held.Keys[0].Status)
	s.Equal("Stream answered 401", held.Keys[0].RejectedReason)
	_, err = s.store.PutStreamApp(s.ctx, registration("4242", 4242, "first"))
	s.Require().NoError(err)
	held, err = s.store.StreamApp(s.ctx, "4242")
	s.Require().NoError(err)
	s.Equal(StreamAppKeyActive, held.Keys[0].Status)
}

func (s *StoreSuite) TestARewrapAppliesOnlyToTheSecretItRead() {
	_, err := s.store.PutStreamApp(s.ctx, registration("4242", 4242, "first"))
	s.Require().NoError(err)

	applied, err := s.store.RewrapStreamAppKey(s.ctx, "4242", "first", []byte("replaced meanwhile"), []byte("rewrapped"), 2)
	s.Require().NoError(err)
	s.False(applied)
	applied, err = s.store.RewrapStreamAppKey(s.ctx, "4242", "first", []byte("sealed first"), []byte("rewrapped"), 2)
	s.Require().NoError(err)
	s.True(applied)

	held, err := s.store.StreamApp(s.ctx, "4242")
	s.Require().NoError(err)
	s.Equal([]byte("rewrapped"), held.Keys[0].Sealed)
	s.Equal(2, held.Keys[0].KEKVersion)
}

func (s *StoreSuite) TestBlockingStopsOnlyAConnectedApp() {
	_, err := s.store.PutStreamApp(s.ctx, registration("4242", 4242, "first"))
	s.Require().NoError(err)

	blocked, err := s.store.BlockStreamApp(s.ctx, "4242", "auth checks are off")
	s.Require().NoError(err)
	s.True(blocked)
	blocked, err = s.store.BlockStreamApp(s.ctx, "4242", "again")
	s.Require().NoError(err)
	s.False(blocked)

	held, err := s.store.StreamApp(s.ctx, "4242")
	s.Require().NoError(err)
	s.Equal(StreamAppBlocked, held.State)
	s.Equal("auth checks are off", held.StateReason)
}

func (s *StoreSuite) TestAWebhookIsRecordedAtMostOnceAMinute() {
	_, err := s.store.PutStreamApp(s.ctx, registration("4242", 4242, "first"))
	s.Require().NoError(err)
	first := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)

	s.Require().NoError(s.store.TouchStreamAppWebhook(s.ctx, "first", first))
	s.Require().NoError(s.store.TouchStreamAppWebhook(s.ctx, "first", first.Add(30*time.Second)))
	held, err := s.store.StreamApp(s.ctx, "4242")
	s.Require().NoError(err)
	s.Require().NotNil(held.Keys[0].LastWebhookAt)
	s.True(first.Equal(*held.Keys[0].LastWebhookAt))

	s.Require().NoError(s.store.TouchStreamAppWebhook(s.ctx, "first", first.Add(2*time.Minute)))
	held, err = s.store.StreamApp(s.ctx, "4242")
	s.Require().NoError(err)
	s.True(first.Add(2 * time.Minute).Equal(*held.Keys[0].LastWebhookAt))
}

func (s *StoreSuite) TestChecksAreKeptWithWhenTheyRan() {
	_, err := s.store.PutStreamApp(s.ctx, registration("4242", 4242, "first"))
	s.Require().NoError(err)
	at := time.Date(2026, 10, 1, 12, 0, 0, 0, time.UTC)

	s.Require().NoError(s.store.RecordStreamAppChecks(context.Background(), "4242",
		json.RawMessage(`{"channel_type":"present"}`), at))

	held, err := s.store.StreamApp(s.ctx, "4242")
	s.Require().NoError(err)
	s.JSONEq(`{"channel_type":"present"}`, string(held.Checks))
	s.True(at.Equal(*held.CheckedAt))
}
