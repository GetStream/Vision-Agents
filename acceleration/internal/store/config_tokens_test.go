//go:build integration

package store

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/uptrace/bun"
)

// configToken stores a configuration token for customer and connector, sealed as the bytes
// sealed say, expiring at expires.
func (s *StoreSuite) configToken(customerID, connectorID, sealed string, expires time.Time) ConnectorConfigToken {
	token := ConnectorConfigToken{
		CustomerID: customerID, ConnectorID: connectorID,
		TokensSealed: []byte(sealed), KEKVersion: 1, ExpiresAt: expires,
	}
	s.Require().NoError(s.store.PutConnectorConfigToken(s.ctx, &token))
	return token
}

func (s *StoreSuite) TestAConfigTokenPutAgainReplacesItsOneRow() {
	first := s.configToken("acme-app", "slack", "first", s.base)
	s.configToken("acme-app", "slack", "rotated", s.base.Add(12*time.Hour))

	found, err := s.store.ConnectorConfigToken(s.ctx, "acme-app", "slack")

	s.Require().NoError(err)
	s.Equal([]byte("rotated"), found.TokensSealed)
	s.True(found.ExpiresAt.Equal(s.base.Add(12 * time.Hour)))
	s.True(found.CreatedAt.Equal(first.CreatedAt), "the row is the first put's")
	var rows int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, "SELECT count(*) FROM connector_config_tokens").Scan(&rows))
	s.Equal(1, rows)
}

func (s *StoreSuite) TestAConfigTokenIsOnlyItsOwnCustomers() {
	s.configToken("acme-app", "slack", "acme's", s.base)

	_, err := s.store.ConnectorConfigToken(s.ctx, "other-app", "slack")

	s.ErrorIs(err, ErrNoConnectorConfigToken)
}

func (s *StoreSuite) TestAConfigTokenWithoutAnExpiryIsRefused() {
	err := s.store.PutConnectorConfigToken(s.ctx, &ConnectorConfigToken{
		CustomerID: "acme-app", ConnectorID: "slack", TokensSealed: []byte("sealed"), KEKVersion: 1,
	})

	s.ErrorContains(err, "an expiry")
}

func (s *StoreSuite) TestDeletingAConfigTokenRemovesItAndAGoneOneIsNoError() {
	s.configToken("acme-app", "slack", "sealed", s.base)

	s.Require().NoError(s.store.DeleteConnectorConfigToken(s.ctx, "acme-app", "slack"))
	s.Require().NoError(s.store.DeleteConnectorConfigToken(s.ctx, "acme-app", "slack"))

	_, err := s.store.ConnectorConfigToken(s.ctx, "acme-app", "slack")
	s.ErrorIs(err, ErrNoConnectorConfigToken)
}

func (s *StoreSuite) TestAConfigTokenSavedAfterTheCallerGaveUpIsKept() {
	gone, cancel := context.WithCancel(s.ctx)
	cancel()

	err := s.store.PutConnectorConfigToken(gone, &ConnectorConfigToken{
		CustomerID: "acme-app", ConnectorID: "slack", TokensSealed: []byte("rotated"), KEKVersion: 1, ExpiresAt: s.base,
	})

	s.Require().NoError(err, "a token Slack already rotated is kept")
	found, err := s.store.ConnectorConfigToken(s.ctx, "acme-app", "slack")
	s.Require().NoError(err)
	s.Equal([]byte("rotated"), found.TokensSealed)
}

func (s *StoreSuite) TestProviderAppLocksOnOneAppNeverOverlapAcrossRouters() {
	const routers = 8
	stores := make([]*Store, routers)
	for i := range stores {
		stores[i] = s.router()
	}
	errs := make([]error, routers)
	start := make(chan struct{})
	var active atomic.Int32
	var overlapped atomic.Bool
	var wg sync.WaitGroup
	for i, router := range stores {
		wg.Go(func() {
			<-start
			errs[i] = router.WithConnectorProviderAppLock(s.ctx, "acme-app", "slack", func() error {
				if active.Add(1) != 1 {
					overlapped.Store(true)
				}
				time.Sleep(10 * time.Millisecond)
				active.Add(-1)
				return nil
			})
		})
	}
	close(start)
	wg.Wait()

	for _, err := range errs {
		s.NoError(err)
	}
	s.False(overlapped.Load(), "two routers managed one customer's app at once")
	s.Zero(s.providerAppLocks(), "every lock is released")
}

func (s *StoreSuite) TestAnotherCustomersProviderAppLockDoesNotWait() {
	other := s.router()
	held := make(chan struct{})
	release := make(chan struct{})
	go func() {
		_ = s.store.WithConnectorProviderAppLock(s.ctx, "acme-app", "slack", func() error {
			close(held)
			<-release
			return nil
		})
	}()
	<-held
	defer close(release)

	ctx, cancel := context.WithTimeout(s.ctx, time.Second)
	defer cancel()
	ran := false
	err := other.WithConnectorProviderAppLock(ctx, "other-app", "slack", func() error {
		ran = true
		return nil
	})

	s.Require().NoError(err)
	s.True(ran)
}

func (s *StoreSuite) TestAProviderAppLockIsNotAConnectionsCredentialLock() {
	held := make(chan struct{})
	release := make(chan struct{})
	go func() {
		_ = s.store.WithConnectorProviderAppLock(s.ctx, "acme-app", "slack", func() error {
			close(held)
			<-release
			return nil
		})
	}()
	<-held
	defer close(release)

	ctx, cancel := context.WithTimeout(s.ctx, time.Second)
	defer cancel()
	err := s.router().withCredentialLock(ctx, "acme-app", "slack", func(bun.Conn) error { return nil })

	s.NoError(err, "the same two ids in the credential namespace are another lock")
}

// providerAppLocks counts the advisory locks held in this database under the provider app
// namespace, as credentialLocks does for credential locks.
func (s *StoreSuite) providerAppLocks() int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, `
SELECT count(*) FROM pg_locks
WHERE locktype = 'advisory' AND classid = ? AND objsubid = 2 AND granted
  AND database = (SELECT oid FROM pg_database WHERE datname = current_database())`,
		providerAppLockNamespace).Scan(&count))
	return count
}
