//go:build integration

package store

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

func (s *StoreSuite) TestConnectorCredentialCoordinationSerializesAcrossConnections() {
	connection := ConnectorConnection{
		CustomerID:  "connector-lock-test",
		ConnectorID: "remote-mcp",
		OwnerType:   "app",
		Endpoint:    "https://mcp.example.com/mcp",
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, &connection))

	const workers = 8
	start := make(chan struct{})
	errs := make(chan error, workers)
	var active atomic.Int32
	var overlap atomic.Bool
	var wait sync.WaitGroup
	wait.Add(workers)
	for i := 0; i < workers; i++ {
		go func() {
			defer wait.Done()
			<-start
			errs <- s.store.WithLockedConnectorConnection(s.ctx, connection.CustomerID, connection.ID,
				func(locked *ConnectorConnection, _ func() error) (bool, error) {
					if active.Add(1) != 1 {
						overlap.Store(true)
					}
					time.Sleep(10 * time.Millisecond)
					locked.Revision++
					locked.LastError = fmt.Sprintf("revision %d", locked.Revision)
					active.Add(-1)
					return true, nil
				})
		}()
	}
	close(start)
	wait.Wait()
	close(errs)
	for err := range errs {
		s.Require().NoError(err)
	}
	s.False(overlap.Load(), "connector refresh callbacks must not overlap across router connections")

	stored, err := s.store.ConnectorConnection(s.ctx, connection.CustomerID, connection.ID)
	s.Require().NoError(err)
	s.Equal(1+workers, stored.Revision)
}

func (s *StoreSuite) TestConnectorCredentialSaveRejectsStaleRevision() {
	connection := ConnectorConnection{
		CustomerID:  "connector-revision-test",
		ConnectorID: "remote-mcp",
		OwnerType:   "app",
		Endpoint:    "https://mcp.example.com/mcp",
	}
	s.Require().NoError(s.store.CreateConnectorConnection(s.ctx, &connection))

	connection.Revision++
	connection.LastError = "updated grant"
	s.Require().NoError(s.store.SaveConnectorConnectionAtRevision(s.ctx, &connection, 1))

	stale := connection
	stale.Revision = 1
	err := s.store.SaveConnectorConnectionAtRevision(s.ctx, &stale, 1)
	s.ErrorIs(err, ErrConnectorConnectionChanged)
}
