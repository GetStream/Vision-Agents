//go:build integration

package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// ReplicateIDsSuite copies agent logs into a target with a shard of its own the way `router
// replicate` does, from a source served here, and then writes a log there the way the router
// does. The source keeps its ids, so the target's sequence is what decides whether the write
// lands.
type ReplicateIDsSuite struct {
	suite.Suite
	ctx    context.Context
	target *store.Store
}

func TestReplicateIDsSuite(t *testing.T) {
	suite.Run(t, new(ReplicateIDsSuite))
}

func (s *ReplicateIDsSuite) SetupTest() {
	s.ctx = context.Background()
	shared, shard := s.emptyDatabase("replicate"), s.emptyDatabase("replicate_shard")
	target, err := store.OpenShards(store.Options{DSN: shared, ShardDSN: shard, Shards: map[string]string{"one": shard}})
	s.Require().NoError(err)
	s.Require().NoError(target.Migrate(s.ctx))
	s.T().Cleanup(func() { target.Close() })
	s.target = target
}

func (s *ReplicateIDsSuite) emptyDatabase(name string) string {
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", dsnEnvVar)
	}
	own := testenv.Database(dsn, name)
	empty, err := store.Open(own)
	s.Require().NoError(err)
	defer empty.Close()
	var database string
	s.Require().NoError(empty.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&database))
	s.Require().True(strings.HasSuffix(database, "_test"), "refusing to drop the schema of %s, which is not a test database", database)
	_, err = empty.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	return own
}

// replicate copies acme's agent logs with the ids exported, then follows one page of the
// changes given, as `router replicate --once` does.
func (s *ReplicateIDsSuite) replicate(exported []int64, changes ...store.DataChange) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/data/export":
			for _, id := range exported {
				fmt.Fprintf(w, `{"table":"agent_logs","row":%s}`+"\n", copiedLog(id))
			}
			fmt.Fprintln(w, `{"cursor":1,"customer":"acme"}`)
		case "/v1/data/changes":
			page := changePage{Changes: changes, Cursor: 1 + int64(len(changes)), CaughtUp: true}
			s.NoError(json.NewEncoder(w).Encode(page))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	source := &deployment{url: server.URL, key: sourceKey, secret: sourceSecret, authType: auth.AuthTypeServer, client: server.Client()}
	logger := slog.New(slog.DiscardHandler)

	customerID, cursor, err := copyEverything(s.ctx, source, s.target, "", logger)
	s.Require().NoError(err)
	s.Require().NoError(follow(s.ctx, source, s.target, customerID, cursor, time.Millisecond, true, logger))
}

// copiedLog is an agent log of acme's as an export carries it, under the source's id.
func copiedLog(id int64) json.RawMessage {
	return json.RawMessage(fmt.Sprintf(`{"id":%d,"customer_id":"acme","config_id":"","agent_id":"","session_id":"",`+
		`"user_id":"","source":"agent","severity":"info","event_type":"turn","message":"copied",`+
		`"occurred_at":"2026-10-30T00:00:00Z","ingested_at":"2026-10-30T00:00:00Z","details":{}}`, id))
}

// inserted is the change recording an insert of an agent log of acme's with the id given.
func inserted(id int64) store.DataChange {
	return store.DataChange{Table: "agent_logs", Op: store.ChangeInsert, Key: json.RawMessage(fmt.Sprintf(`{"id":%d}`, id)),
		Payload: copiedLog(id), At: time.Now().UTC()}
}

// logged writes a log of another app's on the target, the way the router does, and answers it.
func (s *ReplicateIDsSuite) logged() (*store.AgentLog, error) {
	entry := &store.AgentLog{CustomerID: "other", Source: "agent", Severity: "info", EventType: "turn", Message: "live"}
	return entry, s.target.RecordAgentLog(s.ctx, entry)
}

func (s *ReplicateIDsSuite) TestALogWrittenAfterACopyDrawsAnIDAboveTheCopiedOnes() {
	_, err := s.logged()
	s.Require().NoError(err)

	s.replicate([]int64{2, 3})
	written, err := s.logged()

	s.Require().NoError(err, "the target's next id is not one a copied row holds")
	s.Greater(written.ID, int64(3))
}

func (s *ReplicateIDsSuite) TestALogWrittenAfterAReplayedChangeDrawsAnIDAboveIt() {
	_, err := s.logged()
	s.Require().NoError(err)

	s.replicate(nil, inserted(2))
	written, err := s.logged()

	s.Require().NoError(err, "the target's next id is not one a replayed row holds")
	s.Greater(written.ID, int64(2))
}

func (s *ReplicateIDsSuite) TestACopyIntoATargetThatNeverWroteALogTakesItsFirstIDToo() {
	s.replicate([]int64{1})
	written, err := s.logged()

	s.Require().NoError(err, "a sequence nothing has drawn from yet hands out its first number next")
	s.Greater(written.ID, int64(1))
}

func (s *ReplicateIDsSuite) TestATargetSequenceAlreadyPastTheCopiedIDsStaysWhereItIs() {
	// The target is live and has drawn further than the source on its own.
	_, err := s.target.DB().ExecContext(s.ctx, "SELECT setval(pg_get_serial_sequence('agent_logs', 'id'), 5000)")
	s.Require().NoError(err)

	s.replicate([]int64{2, 3}, inserted(4))
	written, err := s.logged()

	s.Require().NoError(err)
	s.Equal(int64(5001), written.ID, "a copy never moves a sequence back")
}

func (s *ReplicateIDsSuite) TestAReplayedDeleteCarriesNoRowToReserveAndIsApplied() {
	s.replicate([]int64{2}, store.DataChange{Table: "agent_logs", Op: store.ChangeDelete,
		Key: json.RawMessage(`{"id":2}`), At: time.Now().UTC()})

	_, err := s.target.AgentLog(s.ctx, "acme", 2)
	s.ErrorIs(err, sql.ErrNoRows)
}
