package mem0

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

// queueLog is the store's log, which names the event each conversation was queued as.
type queueLog struct {
	bytes.Buffer
	read int
}

// unread is the events queued since it was last called, so a test waits only for its own:
// one an earlier test's wait has already seen is not asked about again.
func (l *queueLog) unread() []string {
	log := l.String()
	var events []string
	for line := range strings.Lines(log[l.read:]) {
		var record struct {
			Msg   string `json:"msg"`
			Event string `json:"event"`
		}
		if json.Unmarshal([]byte(line), &record) == nil && record.Msg == "queued a conversation to remember" {
			events = append(events, record.Event)
		}
	}
	l.read = len(log)
	return events
}

// extractionWait waits for mem0 to finish the conversations it was handed. mem0 may start one
// minutes after queueing it, and recall cannot tell one still queued from one that was lost.
type extractionWait struct {
	client  *http.Client
	baseURL string
	apiKey  string
	// bound is how long mem0 may take to finish an event it is still working on.
	bound time.Duration
	every time.Duration
}

// until returns once every event has SUCCEEDED. An event mem0 does not call PENDING or RUNNING
// (FAILED, an unknown status, a refused request) ends the wait at once, since waiting longer
// cannot change it. A request that did not get an answer, or got a 5xx, is asked again.
func (w extractionWait) until(ctx context.Context, events []string) error {
	deadline := time.Now().Add(w.bound)
	for _, event := range events {
		for {
			status, working, err := w.status(ctx, event)
			if err != nil {
				return err
			}
			if status == "SUCCEEDED" {
				break
			}
			if !working {
				return fmt.Errorf("mem0 will not extract event %s: it is %q", event, status)
			}
			if !time.Now().Before(deadline) {
				return fmt.Errorf("mem0 never extracted event %s: it is %q", event, status)
			}
			time.Sleep(w.every)
		}
	}
	return nil
}

// status is what mem0 says of an event, and whether it may still change: true for PENDING,
// RUNNING and an answer that is no answer yet.
func (w extractionWait) status(ctx context.Context, event string) (string, bool, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, w.baseURL+"/v1/event/"+event+"/", nil)
	if err != nil {
		return "", false, err
	}
	request.Header.Set("Authorization", "Token "+w.apiKey)
	response, err := w.client.Do(request)
	if err != nil {
		return err.Error(), true, nil
	}
	defer response.Body.Close()
	if response.StatusCode >= http.StatusInternalServerError {
		return response.Status, true, nil
	}
	var answer struct {
		Status string `json:"status"`
	}
	if response.StatusCode != http.StatusOK || json.NewDecoder(response.Body).Decode(&answer) != nil {
		return response.Status, false, nil
	}
	return answer.Status, answer.Status == "PENDING" || answer.Status == "RUNNING", nil
}

// ExtractionWaitSuite proves the wait ends at once on what waiting cannot change and keeps to
// its bound on what it can. mem0 is a stand-in that answers each event with a canned status.
type ExtractionWaitSuite struct {
	suite.Suite
	answers  map[string]func(int) (int, string)
	asked    atomic.Int32
	wait     extractionWait
	server   *httptest.Server
	finished time.Duration
}

func TestExtractionWaitSuite(t *testing.T) { suite.Run(t, new(ExtractionWaitSuite)) }

func (s *ExtractionWaitSuite) SetupTest() {
	s.answers = map[string]func(int) (int, string){}
	s.asked.Store(0)
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		event := r.URL.Path[len("/v1/event/") : len(r.URL.Path)-1]
		status, body := s.answers[event](int(s.asked.Add(1)))
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	s.T().Cleanup(s.server.Close)
	s.wait = extractionWait{
		client: s.server.Client(), baseURL: s.server.URL, apiKey: "key",
		bound: 2 * time.Second, every: 20 * time.Millisecond,
	}
}

func (s *ExtractionWaitSuite) until(events ...string) error {
	started := time.Now()
	err := s.wait.until(context.Background(), events)
	s.finished = time.Since(started)
	return err
}

func (s *ExtractionWaitSuite) TestAnEventStillWorkedOnIsWaitedForUntilItSucceeds() {
	s.answers["ev"] = func(n int) (int, string) {
		switch {
		case n < 3:
			return http.StatusOK, `{"status":"PENDING"}`
		case n < 5:
			return http.StatusOK, `{"status":"RUNNING"}`
		}
		return http.StatusOK, `{"status":"SUCCEEDED"}`
	}

	s.Require().NoError(s.until("ev"))
	s.GreaterOrEqual(int(s.asked.Load()), 5)
}

func (s *ExtractionWaitSuite) TestAnEventThatStaysPendingFailsAtTheBound() {
	s.answers["ev"] = func(int) (int, string) { return http.StatusOK, `{"status":"PENDING"}` }

	s.ErrorContains(s.until("ev"), "never extracted event ev")
	s.GreaterOrEqual(s.finished, s.wait.bound)
}

func (s *ExtractionWaitSuite) TestAFailedEventFailsAtOnce() {
	s.answers["ev"] = func(int) (int, string) { return http.StatusOK, `{"status":"FAILED"}` }

	s.ErrorContains(s.until("ev"), `will not extract event ev: it is "FAILED"`)
	s.Less(s.finished, s.wait.bound/2)
}

func (s *ExtractionWaitSuite) TestAnUnknownStatusFailsAtOnce() {
	s.answers["ev"] = func(int) (int, string) { return http.StatusOK, `{"status":"WEIRD"}` }

	s.ErrorContains(s.until("ev"), "will not extract event ev")
	s.Less(s.finished, s.wait.bound/2)
}

func (s *ExtractionWaitSuite) TestARefusedRequestFailsAtOnce() {
	s.answers["ev"] = func(int) (int, string) { return http.StatusNotFound, `{}` }

	s.ErrorContains(s.until("ev"), "will not extract event ev")
	s.Less(s.finished, s.wait.bound/2)
}

func (s *ExtractionWaitSuite) TestAServerErrorIsAskedAgain() {
	s.answers["ev"] = func(n int) (int, string) {
		if n < 3 {
			return http.StatusBadGateway, ``
		}
		return http.StatusOK, `{"status":"SUCCEEDED"}`
	}

	s.Require().NoError(s.until("ev"))
}

func (s *ExtractionWaitSuite) TestOnlyTheEventsQueuedSinceTheLastReadAreWaitedFor() {
	var log queueLog
	line := func(msg, event string) string { return fmt.Sprintf(`{"msg":%q,"event":%q}`+"\n", msg, event) }
	log.WriteString(line("queued a conversation to remember", "first"))
	log.WriteString(line("called mem0", "other"))
	s.Equal([]string{"first"}, log.unread())

	log.WriteString(line("queued a conversation to remember", "second"))
	s.Equal([]string{"second"}, log.unread())
	s.Empty(log.unread())
}
