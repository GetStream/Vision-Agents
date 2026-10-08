//go:build integration

package api

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// DataMoveSuite covers the three ways a customer's data goes in or out: the snapshot, the
// changes since it, and writing either of them into a deployment.
type DataMoveSuite struct {
	RouterSuite
}

func TestDataMoveSuite(t *testing.T) {
	runSuite(t, new(DataMoveSuite))
}

func (s *DataMoveSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *DataMoveSuite) TestAnExportEndsWithTheCursorItsChangesCarryOnFrom() {
	// The cursor is last rather than first because it is also what says the export
	// finished: a stream that died halfway has none.
	lines := s.export()

	last := lines[len(lines)-1]
	s.Require().NotNil(last.Cursor, "the export did not finish")
	s.Equal(s.customerID(), last.Customer)
	s.NotNil(last.At)
}

func (s *DataMoveSuite) TestAnExportCarriesTheAppsOwnRows() {
	agent := s.data.createAgentConfig()

	exported := s.export()

	var configs []string
	for _, line := range exported {
		if line.Table == "agent_configs" {
			configs = append(configs, string(line.Row))
		}
	}
	s.Require().Len(configs, 1, "the app has one agent")
	s.Contains(configs[0], agent.Id)
}

func (s *DataMoveSuite) TestAnotherAppsRowsAreNotInTheExport() {
	stranger := s.data.backendOfAnotherApp()
	var theirs AgentConfig
	s.Require().Equal(http.StatusCreated, stranger.do(http.MethodPost, "/v1/agents/configs",
		AgentConfigRequest{Name: "agent-" + s.utils.uuid(), Llm: pointerTo("llm-flow")}, &theirs))

	status, exported := s.serverClient.call(http.MethodGet, "/v1/data/export", nil)

	s.Require().Equal(http.StatusOK, status)
	s.NotContains(string(exported), theirs.Id)
}

func (s *DataMoveSuite) TestWhatHappensAfterAnExportIsInTheChangesSinceIt() {
	s.export()
	agent := s.data.createAgentConfig()

	changed := s.changesSince(0)

	s.True(changed.CaughtUp, "fewer changes than were asked for is the end of them")
	var tables []string
	for _, change := range changed.Changes {
		tables = append(tables, change.Table)
	}
	s.Contains(tables, "agent_configs")
	s.Contains(s.rendered(changed.Changes), agent.Id)
}

func (s *DataMoveSuite) TestFollowingTheChangesFromWhereTheLastPageEndedRepeatsNothing() {
	s.export()
	s.data.createAgentConfig()

	first := s.changesSince(0)
	s.data.createAgentConfig()
	second := s.changesSince(first.Cursor)

	s.NotContains(s.rendered(second.Changes), s.rendered(first.Changes))
}

func (s *DataMoveSuite) TestACursorThatIsNotOneIsRefused() {
	status, failure := s.serverClient.failure(http.MethodGet, "/v1/data/changes?after=yesterday", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "cursor")
}

func (s *DataMoveSuite) TestMoreChangesThanEitherEndCanHoldAreRefused() {
	status, failure := s.serverClient.failure(http.MethodGet, "/v1/data/changes?limit=5000", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "between 1 and")
}

func (s *DataMoveSuite) TestImportingAnExportWritesTheRowsItCarried() {
	s.data.createAgentConfig()
	exported := s.export()

	var written struct {
		Rows   int64            `json:"rows"`
		Tables map[string]int64 `json:"tables"`
		Cursor *int64           `json:"cursor"`
	}
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodPost, "/v1/data/import", ndjson(exported), &written))

	s.Equal(int64(1), written.Tables["agent_configs"])
	s.Equal(int64(len(exported)-1), written.Rows, "every line but the cursor was a row")
	s.Require().NotNil(written.Cursor)
}

func (s *DataMoveSuite) TestSomethingThatIsNotAnExportIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPost, "/v1/data/import", "not an export at all")

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "not an export")
}

func (s *DataMoveSuite) TestOnlyTheAppsOwnBackendMayTakeTheDataOut() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		status, _ := as.call(http.MethodGet, "/v1/data/export", nil)
		return status
	})
}

func (s *DataMoveSuite) TestOnlyTheAppsOwnBackendMayWriteDataIn() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		status, _ := as.call(http.MethodPost, "/v1/data/import", nil)
		return status
	})
}

func (s *DataMoveSuite) TestOnlyTheAppsOwnBackendMayFollowTheChanges() {
	// An export somebody may not have is no safer if the change feed hands the same rows
	// over one at a time.
	s.assertPosture(serverOnly, func(as *testClient) int {
		status, _ := as.call(http.MethodGet, "/v1/data/changes", nil)
		return status
	})
}

// export reads the whole snapshot, one JSON object per line.
func (s *DataMoveSuite) export() []dataLine {
	status, body := s.serverClient.call(http.MethodGet, "/v1/data/export", nil)
	s.Require().Equal(http.StatusOK, status)

	var lines []dataLine
	decoder := json.NewDecoder(strings.NewReader(string(body)))
	for decoder.More() {
		var line dataLine
		s.Require().NoError(decoder.Decode(&line))
		lines = append(lines, line)
	}
	s.Require().NotEmpty(lines)
	return lines
}

// changesSince waits for a change to be readable, asking from the beginning of what is
// recorded. The feed deliberately stops below the oldest transaction still open on the
// deployment, so a change written a moment ago is readable once what was in flight beside
// it has committed.
func (s *DataMoveSuite) changesSince(after int64) changedRows {
	var page changedRows
	answered := 0
	s.Require().Eventually(func() bool {
		answered = s.serverClient.do(http.MethodGet,
			"/v1/data/changes?after="+strconv.FormatInt(after, 10), nil, &page)
		return answered == http.StatusOK && len(page.Changes) > 0
	}, settleFor, 50*time.Millisecond, "no change became readable, the last answer was %d", answered)
	return page
}

// changedRows is what the change feed answers with.
type changedRows struct {
	Changes  []store.DataChange `json:"changes"`
	Cursor   int64              `json:"cursor"`
	CaughtUp bool               `json:"caught_up"`
}

// rendered is the changes as the JSON they carry, for asking whether a row is among them.
func (s *DataMoveSuite) rendered(changes []store.DataChange) string {
	encoded, err := json.Marshal(changes)
	s.Require().NoError(err)
	return string(encoded)
}

// ndjson is an export as the body an import takes.
func ndjson(lines []dataLine) string {
	var written strings.Builder
	for _, line := range lines {
		encoded, err := json.Marshal(line)
		if err != nil {
			continue
		}
		written.Write(encoded)
		written.WriteByte('\n')
	}
	return written.String()
}
