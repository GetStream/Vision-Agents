//go:build integration

package main

import (
	"context"
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

// ReplicateIDsSuite copies phone numbers into a target with a shard of its own the way `router
// replicate` does, from a source served here, and then records a number there the way the
// router does. A copied number keeps the source's id, so the target's sequence is what decides
// whether the write lands. An agent log no longer keeps it (DataMoveSuite).
type ReplicateIDsSuite struct {
	suite.Suite
	ctx    context.Context
	target *store.Store
	// recorded counts the numbers this test recorded on the target, each a number of its own.
	recorded int
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

// replicate copies acme's numbers with the ids exported, then follows one page of the
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
				fmt.Fprintf(w, `{"table":"phone_numbers","row":%s}`+"\n", copiedNumber(id))
				if live == nil {
					continue
				}
				w.(http.Flusher).Flush()
				s.Eventually(func() bool { return s.held(id) }, 10*time.Second, 10*time.Millisecond, "copied number %d landed", id)
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

// numbersLive is a live hook recording a number of another app's on the target each time it runs.
func (s *ReplicateIDsSuite) numbersLive() func() {
	return func() {
		_, err := s.numbered()
		s.NoError(err, "a live insert on the target draws a number no copied row holds")
	}
}

// drawn sets the target's phone number sequence to having handed out last.
func (s *ReplicateIDsSuite) drawn(last int64) {
	_, err := s.target.DB().ExecContext(s.ctx, "SELECT setval(pg_get_serial_sequence('phone_numbers', 'id'), ?)", last)
	s.Require().NoError(err)
}

// asInt4 makes the target's phone number sequence an int4 one, as a serial column's is. The
// databases are this suite's own and are built again for every test.
func (s *ReplicateIDsSuite) asInt4() {
	_, err := s.target.DB().ExecContext(s.ctx, "ALTER SEQUENCE phone_numbers_id_seq AS integer")
	s.Require().NoError(err)
}

// numbersAlongsideAcme makes every insert of an acme number on the target insert a number of
// another app's in the same statement, drawing from the sequence right after the acme row lands.
func (s *ReplicateIDsSuite) numbersAlongsideAcme() {
	_, err := s.target.DB().ExecContext(s.ctx, `
CREATE FUNCTION number_alongside() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
	INSERT INTO phone_numbers (e164, vendor, country, customer_id) VALUES ('+15559999999', 'telnyx', 'US', 'alongside');
	RETURN NULL;
END $$;
CREATE TRIGGER number_alongside AFTER INSERT ON phone_numbers FOR EACH ROW WHEN (NEW.customer_id = 'acme') EXECUTE FUNCTION number_alongside();`)
	s.Require().NoError(err)
}

// alongside is the id the number recorded alongside acme's drew.
func (s *ReplicateIDsSuite) alongside() int64 {
	var id int64
	s.Require().NoError(s.target.DB().QueryRowContext(s.ctx, "SELECT id FROM phone_numbers WHERE customer_id = 'alongside'").Scan(&id))
	return id
}

// held says whether acme holds the number with the id given on the target.
func (s *ReplicateIDsSuite) held(id int64) bool {
	var held bool
	s.Require().NoError(s.target.DB().QueryRowContext(s.ctx,
		"SELECT EXISTS (SELECT 1 FROM phone_numbers WHERE customer_id = 'acme' AND id = ?)", id).Scan(&held))
	return held
}

// copiedNumber is a number of acme's as an export carries it, under the source's id, which
// also makes its digits, so no two copied numbers are one.
func copiedNumber(id int64) json.RawMessage {
	return json.RawMessage(fmt.Sprintf(`{"id":%d,"e164":"+1555%07d","vendor":"telnyx","country":"US","customer_id":"acme",`+
		`"capabilities":["voice"],"monthly_cost_micros":0,"tags":{},"purchased_at":"2026-10-30T00:00:00Z"}`, id, id))
}

// inserted is the change recording an insert of a number of acme's with the id given.
func inserted(id int64) store.DataChange {
	return store.DataChange{Table: "phone_numbers", Op: store.ChangeInsert, Key: json.RawMessage(fmt.Sprintf(`{"id":%d}`, id)),
		Payload: copiedNumber(id), At: time.Now().UTC()}
}

// numbered records a number of another app's on the target, the way the router does, and
// answers it. Its digits are the 8000s, which no copied number's are.
func (s *ReplicateIDsSuite) numbered() (*store.PhoneNumber, error) {
	s.recorded++
	number := &store.PhoneNumber{E164: fmt.Sprintf("+1555%07d", 8000+s.recorded), Vendor: "telnyx", Country: "US", CustomerID: "other"}
	return number, s.target.RecordNumber(s.ctx, number)
}

func (s *ReplicateIDsSuite) TestANumberRecordedAfterACopyDrawsAnIDAboveTheCopiedOnes() {
	_, err := s.numbered()
	s.Require().NoError(err)

	s.replicate([]int64{2, 3})
	written, err := s.numbered()

	s.Require().NoError(err, "the target's next id is not one a copied row holds")
	s.Greater(written.ID, int64(3))
}

func (s *ReplicateIDsSuite) TestANumberRecordedAfterAReplayedChangeDrawsAnIDAboveIt() {
	_, err := s.numbered()
	s.Require().NoError(err)

	s.replicate(nil, inserted(2))
	written, err := s.numbered()

	s.Require().NoError(err, "the target's next id is not one a replayed row holds")
	s.Greater(written.ID, int64(2))
}

func (s *ReplicateIDsSuite) TestACopyIntoATargetThatNeverRecordedANumberTakesItsFirstIDToo() {
	s.replicate([]int64{1})
	written, err := s.numbered()

	s.Require().NoError(err, "a sequence nothing has drawn from yet hands out its first number next")
	s.Greater(written.ID, int64(1))
}

func (s *ReplicateIDsSuite) TestARunReservesOnceForTheIDsWithinHalfAGapOfItsFirst() {
	s.replicate([]int64{100, 101, 102}, inserted(103))
	written, err := s.numbered()

	s.Require().NoError(err)
	s.Equal(int64(100+1<<40+1), written.ID, "one reservation, a gap past the first copied id, covers the rest")
}

func (s *ReplicateIDsSuite) TestATargetSequenceAlreadyPastTheCopiedIDsStaysWhereItIs() {
	// The target is live and has drawn further than the source on its own, by more than the
	// half gap a copy keeps between a copied id and the target's next.
	s.drawn(1 << 40)

	s.replicate([]int64{2, 3}, inserted(4))
	written, err := s.numbered()

	s.Require().NoError(err)
	s.Equal(int64(1<<40+1), written.ID, "a copy leaves a sequence far enough ahead where it is")
}

func (s *ReplicateIDsSuite) TestASequenceTheTargetDrewPastAfterTheRunReadItIsNotMovedBack() {
	reservation := s.reserved()
	// Live inserts on the target draw on after the run read its sequence.
	s.drawn(3 << 40)

	s.Require().NoError(s.replicateWith(reservation, []int64{1}, nil))
	written, err := s.numbered()

	s.Require().NoError(err)
	s.Greater(written.ID, int64(3<<40), "a copy never moves a sequence back")
}

func (s *ReplicateIDsSuite) TestALiveNumberRecordedDuringTheCopyLeavesTheNextCopiedNumberItsID() {
	_, err := s.numbered()
	s.Require().NoError(err)

	s.Require().NoError(s.replicateWith(s.reserved(), []int64{100, 101}, s.numbersLive()))

	s.True(s.held(101), "acme's number copied after a live insert landed under its own id")
}

func (s *ReplicateIDsSuite) TestALiveNumberRecordedBetweenTheCopyAndAReplayedInsertLeavesItItsID() {
	_, err := s.numbered()
	s.Require().NoError(err)

	s.Require().NoError(s.replicateWith(s.reserved(), []int64{100}, s.numbersLive(), inserted(101)))

	s.True(s.held(101), "acme's number replayed after a live insert landed under its own id")
}

func (s *ReplicateIDsSuite) TestACopiedNumberIsReservedBeforeItLands() {
	s.drawn(99)
	s.numbersAlongsideAcme()

	s.replicate([]int64{100})

	s.Greater(s.alongside(), int64(100), "a number recorded as the copied one lands draws past it")
}

func (s *ReplicateIDsSuite) TestAReplayedNumberIsReservedBeforeItLands() {
	s.drawn(99)
	s.numbersAlongsideAcme()

	s.replicate(nil, inserted(100))

	s.Greater(s.alongside(), int64(100), "a number recorded as the replayed one lands draws past it")
}

func (s *ReplicateIDsSuite) TestAReplayedUpdateOfANumberTheTargetLacksIsReservedToo() {
	_, err := s.numbered()
	s.Require().NoError(err)
	updated := inserted(2)
	updated.Op = store.ChangeUpdate

	s.replicate(nil, updated)
	written, err := s.numbered()

	s.Require().NoError(err, "a replayed update writes the row when the target lacks it, so its id is reserved too")
	s.Greater(written.ID, int64(2))
}

func (s *ReplicateIDsSuite) TestAnInt4SequenceIsReservedAheadByAShareOfItsRange() {
	s.asInt4()

	s.replicate([]int64{100})
	written, err := s.numbered()

	s.Require().NoError(err)
	// 1/256 of an int4 sequence's range: 2^40 does not fit in one.
	s.Equal(int64(100+(1<<31-1)/256+1), written.ID)
}

func (s *ReplicateIDsSuite) TestAnInt4SequenceWithoutRoomForAReservationStopsTheRunBeforeItCopies() {
	s.asInt4()
	s.drawn(1<<31 - 1 - 1000)

	_, err := s.target.ReserveImportedIDs(s.ctx)

	s.ErrorContains(err, "phone_numbers_id_seq has 1000 numbers left")
}

func (s *ReplicateIDsSuite) TestAReplayedDeleteCarriesNoRowToReserveAndIsApplied() {
	s.replicate([]int64{2}, store.DataChange{Table: "phone_numbers", Op: store.ChangeDelete,
		Key: json.RawMessage(`{"id":2}`), At: time.Now().UTC()})

	s.False(s.held(2))
}
