//go:build integration

package api

import (
	"bufio"
	"context"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type LogsSuite struct {
	RouterSuite
}

func TestLogsSuite(t *testing.T) {
	runSuite(t, new(LogsSuite))
}

// SetupTest gives every test an app of its own, because a page of logs is everything one
// customer's agents have written.
func (s *LogsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *LogsSuite) TestAPageOfLogsIsReadBackWithACursorToResumeFrom() {
	s.record("info", "First")

	page := s.page("?limit=1")
	s.Require().Len(page.Items, 1)
	s.Equal("First", page.Items[0].Message)
	s.NotEmpty(page.ResumeCursor, "a reader that wants the rest has to know where it got to")
}

func (s *LogsSuite) TestAskingForWarningsLeavesOutTheRunningCommentary() {
	s.record("info", "Ordinary")
	s.record("warn", "Worth a look")
	s.record("error", "Broken")

	messages := func(page AgentLogPage) []string {
		var seen []string
		for _, item := range page.Items {
			seen = append(seen, item.Message)
		}
		return seen
	}

	s.ElementsMatch([]string{"Worth a look", "Broken"}, messages(s.page("?severity=warn")),
		"a severity is the least serious level to show, not the only one")
	s.ElementsMatch([]string{"Broken"}, messages(s.page("?severity=error")))
	s.ElementsMatch([]string{"Ordinary", "Worth a look", "Broken"}, messages(s.page("")),
		"naming no severity still shows everything")
}

func (s *LogsSuite) TestALiveStreamReplaysWhatWasRecordedAfterTheSnapshot() {
	s.record("info", "First")
	page := s.page("?limit=1")

	stream := s.stream(page.ResumeCursor, "")
	defer stream.Body.Close()
	s.Require().Equal(http.StatusOK, stream.StatusCode)

	s.record("error", "After snapshot")

	_, seen := s.until(stream, "After snapshot")
	s.True(seen, "a live stream replays a record written after the snapshot")
}

func (s *LogsSuite) TestAReconnectAsksFromTheLastEventItAcknowledged() {
	s.record("info", "First")
	page := s.page("?limit=1")

	stream := s.stream(page.ResumeCursor, "")
	s.record("error", "After snapshot")
	acknowledged, seen := s.until(stream, "After snapshot")
	s.Require().True(seen)
	s.Require().NoError(stream.Body.Close())

	s.record("error", "After reconnect")

	// The same snapshot cursor, so what keeps the acknowledged event from arriving twice
	// is the id the reader last saw rather than where it first asked from.
	resumed := s.stream(page.ResumeCursor, acknowledged)
	defer resumed.Body.Close()

	reader := bufio.NewScanner(resumed.Body)
	for reader.Scan() {
		line := reader.Text()
		s.Require().NotContains(line, "After snapshot", "a reconnect does not replay what was acknowledged")
		if strings.Contains(line, "After reconnect") {
			return
		}
	}
	s.Fail("the reconnected stream never reached the record written after it")
}

func (s *LogsSuite) TestAPageBiggerThanTheOneAllowedIsRefused() {
	s.Equal(http.StatusBadRequest,
		s.serverClient.do(http.MethodGet, "/v1/agents/logs?limit=251", nil, nil))
}

func (s *LogsSuite) TestAnotherAppsLogIsNotFound() {
	s.record("info", "First")
	page := s.page("?limit=1")
	s.Require().Len(page.Items, 1)

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet,
			"/v1/agents/logs/"+page.Items[0].Id, nil, nil)
	})
}

func (s *LogsSuite) TestOnlyTheAppsOwnBackendMayReadItsLogs() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/logs?limit=1", nil, nil)
	})
}

// record writes one line as an agent of the suite's app would.
func (s *LogsSuite) record(severity, message string) {
	s.Require().NoError(s.store.RecordAgentLog(context.Background(), &store.AgentLog{
		CustomerID: s.customerID(), Source: "agent", Severity: severity,
		EventType: "test", Message: message,
	}))
}

// page reads a page of logs.
func (s *LogsSuite) page(query string) AgentLogPage {
	var page AgentLogPage
	s.Require().Equal(http.StatusOK,
		s.serverClient.do(http.MethodGet, "/v1/agents/logs"+query, nil, &page))
	return page
}

// stream opens the live stream from a snapshot cursor, resuming after an event when one is
// named. The caller closes the body.
func (s *LogsSuite) stream(cursor, lastEventID string) *http.Response {
	request, err := http.NewRequest(http.MethodGet,
		s.server.URL+"/v1/agents/logs/stream?cursor="+cursor+"&severity=error", nil)
	s.Require().NoError(err)
	request.Header = s.serverClient.header.Clone()
	if lastEventID != "" {
		request.Header.Set("Last-Event-ID", lastEventID)
	}

	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	return response
}

// until reads the stream up to the line holding what was written, and returns the id of
// the last event it saw on the way.
func (s *LogsSuite) until(stream *http.Response, written string) (lastEventID string, seen bool) {
	deadline := time.Now().Add(settleFor)
	reader := bufio.NewScanner(stream.Body)
	for reader.Scan() {
		line := reader.Text()
		if id, ok := strings.CutPrefix(line, "id: "); ok {
			lastEventID = id
		}
		if strings.Contains(line, written) {
			return lastEventID, true
		}
		if time.Now().After(deadline) {
			break
		}
	}
	return lastEventID, false
}
