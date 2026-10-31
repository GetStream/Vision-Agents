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
	s.Require().NoError(s.replicateWith(s.reserved(), exported, nil, changes...))
}

// reserved is the reservation a run of `router replicate` starts with on the target.
func (s *ReplicateIDsSuite) reserved() *store.IDReservation {
	reservation, err := s.target.ReserveImportedIDs(s.ctx)
	s.Require().NoError(err)
	return reservation
}

// replicateWith is replicate with the reservation given and, when live is given, live run
// on the target once each copied row has landed and once more before the changes are followed:
// the target is live while it is copied into.
func (s *ReplicateIDsSuite) replicateWith(reservation *store.IDReservation, exported []int64, live func(), changes ...store.DataChange) error {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/data/export":
			for _, id := range exported {
				fmt.Fprintf(w, `{"table":"agent_logs","row":%s}`+"\n", copiedLog(id))
				if live == nil {
					continue
				}
				w.(http.Flusher).Flush()
				s.Eventually(func() bool {
					_, err := s.target.AgentLog(s.ctx, "acme", id)
					return err == nil
				}, 10*time.Second, 10*time.Millisecond, "copied log %d landed", id)
				live()
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

	customerID, cursor, err := copyEverything(s.ctx, source, s.target, reservation, "", logger)
	if err != nil {
		return err
	}
	if live != nil {
		live()
	}
	return follow(s.ctx, source, s.target, reservation, customerID, cursor, time.Millisecond, true, logger)
}

// logsLive is a live hook writing a log of another app's on the target each time it runs.
func (s *ReplicateIDsSuite) logsLive() func() {
	return func() {
		_, err := s.logged()
		s.NoError(err, "a live insert on the target draws a number no copied row holds")
	}
}

// drawn sets the target's agent log sequence to having handed out last.
func (s *ReplicateIDsSuite) drawn(last int64) {
	_, err := s.target.DB().ExecContext(s.ctx, "SELECT setval(pg_get_serial_sequence('agent_logs', 'id'), ?)", last)
	s.Require().NoError(err)
}

// asInt4 makes the target's agent log sequence an int4 one, as a serial column's is. The
// databases are this suite's own and are built again for every test.
func (s *ReplicateIDsSuite) asInt4() {
	_, err := s.target.DB().ExecContext(s.ctx, "ALTER SEQUENCE agent_logs_id_seq AS integer")
	s.Require().NoError(err)
}

// logsAlongsideAcme makes every insert of an acme log on the target insert a log of another
// app's in the same statement, drawing from the sequence right after the acme row lands.
func (s *ReplicateIDsSuite) logsAlongsideAcme() {
	_, err := s.target.DB().ExecContext(s.ctx, `
CREATE FUNCTION log_alongside() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
	INSERT INTO agent_logs (customer_id, source, severity, event_type, message) VALUES ('other', 'agent', 'info', 'turn', 'alongside');
	RETURN NULL;
END $$;
CREATE TRIGGER log_alongside AFTER INSERT ON agent_logs FOR EACH ROW WHEN (NEW.customer_id = 'acme') EXECUTE FUNCTION log_alongside();`)
	s.Require().NoError(err)
}

// alongside is the id the log written alongside acme's drew.
func (s *ReplicateIDsSuite) alongside() int64 {
	var id int64
	s.Require().NoError(s.target.DB().QueryRowContext(s.ctx, "SELECT id FROM agent_logs WHERE message = 'alongside'").Scan(&id))
	return id
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

func (s *ReplicateIDsSuite) TestARunReservesOnceForTheIDsWithinHalfAGapOfItsFirst() {
	s.replicate([]int64{100, 101, 102}, inserted(103))
	written, err := s.logged()

	s.Require().NoError(err)
	s.Equal(int64(100+1<<40+1), written.ID, "one reservation, a gap past the first copied id, covers the rest")
}

func (s *ReplicateIDsSuite) TestATargetSequenceAlreadyPastTheCopiedIDsStaysWhereItIs() {
	// The target is live and has drawn further than the source on its own, by more than the
	// half gap a copy keeps between a copied id and the target's next.
	s.drawn(1 << 40)

	s.replicate([]int64{2, 3}, inserted(4))
	written, err := s.logged()

	s.Require().NoError(err)
	s.Equal(int64(1<<40+1), written.ID, "a copy leaves a sequence far enough ahead where it is")
}

func (s *ReplicateIDsSuite) TestASequenceTheTargetDrewPastAfterTheRunReadItIsNotMovedBack() {
	reservation := s.reserved()
	// Live inserts on the target draw on after the run read its sequence.
	s.drawn(3 << 40)

	s.Require().NoError(s.replicateWith(reservation, []int64{1}, nil))
	written, err := s.logged()

	s.Require().NoError(err)
	s.Greater(written.ID, int64(3<<40), "a copy never moves a sequence back")
}

func (s *ReplicateIDsSuite) TestALiveLogWrittenDuringTheCopyLeavesTheNextCopiedLogItsID() {
	_, err := s.logged()
	s.Require().NoError(err)

	s.Require().NoError(s.replicateWith(s.reserved(), []int64{100, 101}, s.logsLive()))

	_, err = s.target.AgentLog(s.ctx, "acme", 101)
	s.NoError(err, "acme's log copied after a live insert landed under its own id")
}

func (s *ReplicateIDsSuite) TestALiveLogWrittenBetweenTheCopyAndAReplayedInsertLeavesItItsID() {
	_, err := s.logged()
	s.Require().NoError(err)

	s.Require().NoError(s.replicateWith(s.reserved(), []int64{100}, s.logsLive(), inserted(101)))

	_, err = s.target.AgentLog(s.ctx, "acme", 101)
	s.NoError(err, "acme's log replayed after a live insert landed under its own id")
}

func (s *ReplicateIDsSuite) TestACopiedLogIsReservedBeforeItLands() {
	s.drawn(99)
	s.logsAlongsideAcme()

	s.replicate([]int64{100})

	s.Greater(s.alongside(), int64(100), "a log written as the copied one lands draws past it")
}

func (s *ReplicateIDsSuite) TestAReplayedLogIsReservedBeforeItLands() {
	s.drawn(99)
	s.logsAlongsideAcme()

	s.replicate(nil, inserted(100))

	s.Greater(s.alongside(), int64(100), "a log written as the replayed one lands draws past it")
}

func (s *ReplicateIDsSuite) TestAReplayedUpdateOfALogTheTargetLacksIsReservedToo() {
	_, err := s.logged()
	s.Require().NoError(err)
	updated := inserted(2)
	updated.Op = store.ChangeUpdate

	s.replicate(nil, updated)
	written, err := s.logged()

	s.Require().NoError(err, "a replayed update writes the row when the target lacks it, so its id is reserved too")
	s.Greater(written.ID, int64(2))
}

func (s *ReplicateIDsSuite) TestAnInt4SequenceIsReservedAheadByAShareOfItsRange() {
	s.asInt4()

	s.replicate([]int64{100})
	written, err := s.logged()

	s.Require().NoError(err)
	// 1/256 of an int4 sequence's range: 2^40 does not fit in one.
	s.Equal(int64(100+(1<<31-1)/256+1), written.ID)
}

func (s *ReplicateIDsSuite) TestAnInt4SequenceWithoutRoomForAReservationStopsTheRunBeforeItCopies() {
	s.asInt4()
	s.drawn(1<<31 - 1 - 1000)

	_, err := s.target.ReserveImportedIDs(s.ctx)

	s.ErrorContains(err, "agent_logs_id_seq has 1000 numbers left")
}

func (s *ReplicateIDsSuite) TestAReplayedDeleteCarriesNoRowToReserveAndIsApplied() {
	s.replicate([]int64{2}, store.DataChange{Table: "agent_logs", Op: store.ChangeDelete,
		Key: json.RawMessage(`{"id":2}`), At: time.Now().UTC()})

	_, err := s.target.AgentLog(s.ctx, "acme", 2)
	s.ErrorIs(err, sql.ErrNoRows)
}
