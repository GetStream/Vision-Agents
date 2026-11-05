//go:build integration

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// ReplicateRepeatSuite runs `router replicate` more than once between two real stores: a source
// served the way the router serves /v1/data/export and /v1/data/changes for the app acme, and a
// target.
type ReplicateRepeatSuite struct {
	suite.Suite
	ctx    context.Context
	source *store.Store
	target *store.Store
	server *httptest.Server
	// exports counts the exports the source was asked for.
	exports int
	// truncated makes the source's export stop before its cursor line, as a broken stream does.
	truncated bool
	// changesDown makes the source answer the changes 503.
	changesDown bool
}

func TestReplicateRepeatSuite(t *testing.T) {
	suite.Run(t, new(ReplicateRepeatSuite))
}

func (s *ReplicateRepeatSuite) SetupTest() {
	s.ctx = context.Background()
	s.exports, s.truncated, s.changesDown = 0, false, false
	s.source = s.emptyStore("repeat_source")
	s.target = s.emptyStore("repeat_target")
	s.server = httptest.NewServer(http.HandlerFunc(s.serveSource))
	s.T().Cleanup(s.server.Close)
}

// emptyStore is a migrated store on a database of this suite's own, emptied first.
func (s *ReplicateRepeatSuite) emptyStore(name string) *store.Store {
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", dsnEnvVar)
	}
	own := testenv.Database(dsn, name)
	opened, err := store.Open(own)
	s.Require().NoError(err)
	s.T().Cleanup(func() { opened.Close() })
	var database string
	s.Require().NoError(opened.DB().QueryRowContext(s.ctx, "SELECT current_database()").Scan(&database))
	s.Require().True(strings.HasSuffix(database, "_test"), "refusing to drop the schema of %s, which is not a test database", database)
	_, err = opened.DB().ExecContext(s.ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public")
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	return opened
}

// serveSource answers the export and the changes from the source store, as internal/api/datamove.go
// does.
func (s *ReplicateRepeatSuite) serveSource(w http.ResponseWriter, r *http.Request) {
	if err := s.source.StartDataCapture(r.Context(), "acme", time.Hour); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	switch r.URL.Path {
	case "/v1/data/export":
		s.exports++
		encoder := json.NewEncoder(w)
		cursor, err := s.source.ExportCustomer(r.Context(), "acme", func(table string, row json.RawMessage) error {
			return encoder.Encode(map[string]any{"table": table, "row": row})
		})
		if err != nil || s.truncated {
			return
		}
		_ = encoder.Encode(map[string]any{"cursor": cursor, "customer": "acme"})
	case "/v1/data/changes":
		if s.changesDown {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		after, _ := strconv.ParseInt(r.URL.Query().Get("after"), 10, 64)
		changes, cursor, err := s.source.Changes(r.Context(), "acme", after, 500)
		if errors.Is(err, store.ErrChangesExpired) {
			w.WriteHeader(http.StatusGone)
			return
		}
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		_ = json.NewEncoder(w).Encode(changePage{Changes: changes, Cursor: cursor, CaughtUp: len(changes) < 500})
	default:
		w.WriteHeader(http.StatusNotFound)
	}
}

// once runs `router replicate --once` against the source, with --as when as is not empty and
// with --full when full is.
func (s *ReplicateRepeatSuite) once(as string, full bool) error {
	source := s.deployment()
	logger := slog.New(slog.DiscardHandler)
	reservation, err := s.target.ReserveImportedIDs(s.ctx)
	s.Require().NoError(err)
	customerID, cursor, err := copyEverything(s.ctx, source, s.target, reservation, as, full, time.Hour, logger)
	if err != nil {
		return err
	}
	return follow(s.ctx, source, s.target, reservation, customerID, cursor, time.Millisecond, true, logger)
}

func (s *ReplicateRepeatSuite) deployment() *deployment {
	return &deployment{url: s.server.URL, key: sourceKey, secret: sourceSecret, authType: auth.AuthTypeServer, client: s.server.Client()}
}

// copiedBefore is a first copy of acme made by a run that saved no cursor, with numbers 1 and 2,
// number 2 then edited here.
func (s *ReplicateRepeatSuite) copiedBefore() {
	s.recorded(1)
	s.recorded(2)
	s.Require().NoError(s.once("", false))
	_, err := s.target.DB().ExecContext(s.ctx, "DELETE FROM data_move_cursors")
	s.Require().NoError(err)
	s.tagged(s.target, 2, "edited here")
}

// recorded records a number of acme's at the source, its digits the number given.
func (s *ReplicateRepeatSuite) recorded(digits int) {
	number := &store.PhoneNumber{E164: e164(digits), Vendor: "telnyx", Country: "US", CustomerID: "acme"}
	s.Require().NoError(s.source.RecordNumber(s.ctx, number))
}

// tagged sets the label tag of a number of acme's in a store.
func (s *ReplicateRepeatSuite) tagged(in *store.Store, digits int, label string) {
	_, err := in.DB().ExecContext(s.ctx, "UPDATE phone_numbers SET tags = jsonb_build_object('label', ?::text) WHERE e164 = ?",
		label, e164(digits))
	s.Require().NoError(err)
}

// label is a number's label tag on the target, empty when it has none and "missing" when the
// target lacks the number.
func (s *ReplicateRepeatSuite) label(digits int) string {
	var label *string
	var held bool
	s.Require().NoError(s.target.DB().QueryRowContext(s.ctx,
		"SELECT count(*) > 0, max(tags->>'label') FROM phone_numbers WHERE customer_id = 'acme' AND e164 = ?",
		e164(digits)).Scan(&held, &label))
	if !held {
		return "missing"
	}
	if label == nil {
		return ""
	}
	return *label
}

// saved is the cursor the target saved for acme's copy from the source.
func (s *ReplicateRepeatSuite) saved() *store.MoveCursor {
	saved, err := s.target.MoveCursor(s.ctx, "acme", s.server.URL)
	s.Require().NoError(err)
	return saved
}

func e164(digits int) string {
	return fmt.Sprintf("+1555%07d", digits)
}

// The values a first copy lands are what one landed before runs saved a cursor (a probe on
// 60c9d2245: "source", "", one export).
func (s *ReplicateRepeatSuite) TestAFirstCopyBringsEverythingAndSavesTheCursor() {
	s.recorded(1)
	s.recorded(2)
	s.tagged(s.source, 1, "source")

	s.Require().NoError(s.once("", false))

	s.Equal("source", s.label(1))
	s.Equal("", s.label(2))
	s.Equal(1, s.exports)
	saved := s.saved()
	s.Require().NotNil(saved)
	s.NotNil(saved.Cursor, "a finished copy saves the cursor it carries on from")
	s.WithinDuration(time.Now(), saved.ReadAt, time.Minute)
}

func (s *ReplicateRepeatSuite) TestARepeatOnceBringsOnlyTheChangesSinceItsCursor() {
	s.recorded(1)
	s.recorded(2)
	s.Require().NoError(s.once("", false))
	s.tagged(s.target, 2, "edited here")
	s.tagged(s.source, 1, "source again")
	s.recorded(3)

	s.Require().NoError(s.once("acme", false))

	s.Equal("source again", s.label(1), "a change at the source since the first copy arrives")
	s.Equal("", s.label(3), "a number recorded at the source since the first copy arrives")
	s.Equal("edited here", s.label(2), "a row the source left alone keeps what was done to it here")
	s.Equal(1, s.exports, "a repeat run with --as asks for no export")
}

func (s *ReplicateRepeatSuite) TestARepeatOnceWithoutAsWritesNoRowOfTheExport() {
	s.recorded(1)
	s.recorded(2)
	s.Require().NoError(s.once("", false))
	s.tagged(s.target, 1, "edited here")
	s.tagged(s.target, 2, "edited here")
	s.tagged(s.source, 2, "source again")

	s.Require().NoError(s.once("", false))

	s.Equal("edited here", s.label(1), "the export's first row, which names the app, is not written")
	s.Equal("source again", s.label(2), "the change since the cursor is")
}

func (s *ReplicateRepeatSuite) TestAThirdRunDoesNotReplayWhatTheSecondBrought() {
	s.recorded(1)
	s.Require().NoError(s.once("acme", false))
	s.tagged(s.source, 1, "source again")
	s.Require().NoError(s.once("acme", false))
	s.tagged(s.target, 1, "edited here")

	s.Require().NoError(s.once("acme", false))

	s.Equal("edited here", s.label(1), "each page followed saves the cursor past it")
}

func (s *ReplicateRepeatSuite) TestARepeatOnceForAnAppTheSourceNowHoldsNothingOfAppliesItsDeletes() {
	s.recorded(1)
	s.recorded(2)
	s.Require().NoError(s.once("", false))
	_, err := s.source.DB().ExecContext(s.ctx, "DELETE FROM phone_numbers WHERE customer_id = 'acme'")
	s.Require().NoError(err)

	// The export is empty, so only its last line names the app.
	s.Require().NoError(s.once("", false))

	s.Equal("missing", s.label(1))
	s.Equal("missing", s.label(2))
}

func (s *ReplicateRepeatSuite) TestACopyWhoseChangesWereNotReadStillSavesTheExportsCursor() {
	s.recorded(1)
	s.changesDown = true
	s.Require().Error(s.once("acme", false))
	s.changesDown = false
	s.tagged(s.target, 1, "edited here")

	s.Require().NoError(s.once("acme", false))

	s.Equal("edited here", s.label(1), "the next run carries on from the export's cursor")
	s.Equal(1, s.exports)
}

func (s *ReplicateRepeatSuite) TestACopyThatStoppedPartwayIsCopiedAgainWithoutFull() {
	s.recorded(1)
	s.truncated = true
	s.Require().ErrorContains(s.once("", false), "the export stopped before it finished")
	s.truncated = false
	s.tagged(s.target, 1, "edited here")
	s.tagged(s.source, 1, "source")

	s.Require().NoError(s.once("", false), "a copy that never finished is not one to protect")

	s.Equal("source", s.label(1))
	s.Equal(2, s.exports)
}

func (s *ReplicateRepeatSuite) TestAnAppCopiedBeforeRunsSavedACursorIsRefused() {
	s.copiedBefore()

	err := s.once("", false)

	s.Require().ErrorContains(err, "acme already has data here")
	s.ErrorContains(err, "--full")
	s.Equal("edited here", s.label(2), "nothing is written over it")
}

func (s *ReplicateRepeatSuite) TestAnAppCopiedBeforeRunsSavedACursorIsRefusedWithAsBeforeAnyExport() {
	s.copiedBefore()

	s.Require().ErrorContains(s.once("acme", false), "--full")

	s.Equal(1, s.exports, "the refusal asks the source for nothing")
	s.Equal("edited here", s.label(2))
}

func (s *ReplicateRepeatSuite) TestFullCopiesEverythingAgainAndSavesTheCursor() {
	s.copiedBefore()
	s.tagged(s.source, 1, "source again")

	s.Require().NoError(s.once("acme", true))

	s.Equal("", s.label(2), "the source's row is written over what was done here")
	s.Equal("source again", s.label(1))
	s.Equal(2, s.exports)
	s.tagged(s.target, 2, "edited here")
	s.Require().NoError(s.once("acme", false), "the run after --full carries on from its cursor")
	s.Equal("edited here", s.label(2))
	s.Equal(2, s.exports)
}

func (s *ReplicateRepeatSuite) TestACursorReadLongerAgoThanTheRetentionIsRefused() {
	s.recorded(1)
	s.Require().NoError(s.once("acme", false))
	_, err := s.target.DB().ExecContext(s.ctx, "UPDATE data_move_cursors SET read_at = now() - interval '61 minutes'")
	s.Require().NoError(err)
	s.tagged(s.target, 1, "edited here")

	err = s.once("acme", false)

	s.Require().ErrorContains(err, "ROUTER_DATA_MOVE_RETENTION")
	s.ErrorContains(err, "--full")
	s.Equal("edited here", s.label(1))
	s.Equal(1, s.exports)
}

func (s *ReplicateRepeatSuite) TestACursorTheSourceNoLongerKeepsTheChangesOfNamesFull() {
	// A change before the first copy gives its cursor a number the source can lose.
	s.recorded(1)
	s.Require().NoError(s.source.StartDataCapture(s.ctx, "acme", time.Hour))
	s.tagged(s.source, 1, "before the copy")
	s.Require().NoError(s.once("acme", false))
	s.tagged(s.source, 1, "pruned")
	s.tagged(s.source, 1, "kept")
	// What pruning does to every change but the newest.
	_, err := s.source.DB().ExecContext(s.ctx, "DELETE FROM data_changes WHERE seq < (SELECT max(seq) FROM data_changes)")
	s.Require().NoError(err)

	s.Require().ErrorContains(s.once("acme", false), "no longer keeps the changes since that cursor: run this again with --full")

	s.Equal("before the copy", s.label(1))
}
