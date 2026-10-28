//go:build integration

package api

import (
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

// mem0Extraction waits for mem0 to finish the exchanges the router handed it. mem0 may start
// one minutes after queueing it, and recall cannot tell one still queued from one that was lost.
type mem0Extraction struct {
	client  *http.Client
	baseURL string
	apiKey  string
	// log is the mem0 client's log, which names the event each exchange was queued as.
	log func() string
	// grace is how long the router has to hand a session's exchange over to mem0.
	grace time.Duration
	// bound is how long mem0 may take to finish an event it is still working on.
	bound time.Duration
	every time.Duration
}

// until returns once mem0 has finished every session's exchange, or by the latest at the
// suite's deadline. A session whose exchange the router does not hand over within grace is
// the router's fault, and one mem0 does not call PENDING or RUNNING (FAILED, an unknown status,
// a refused request) ends the wait at once, since waiting longer cannot change it.
func (w mem0Extraction) until(sessions []string, suiteDeadline time.Time) error {
	handover := time.Now().Add(w.grace)
	deadline := time.Now().Add(w.bound)
	if suiteDeadline.Before(deadline) {
		deadline = suiteDeadline
	}
	for _, session := range sessions {
		event := w.queuedAs(session)
		for event == "" {
			if !time.Now().Before(handover) {
				return fmt.Errorf("router never handed session %s to mem0", session)
			}
			time.Sleep(w.every)
			event = w.queuedAs(session)
		}
		for {
			status, working := w.status(event)
			if status == "SUCCEEDED" {
				break
			}
			if !working {
				return fmt.Errorf("mem0 will not extract what session %s was told: event %s is %q", session, event, status)
			}
			if !time.Now().Before(deadline) {
				return fmt.Errorf("mem0 never extracted what session %s was told: event %s is %q", session, event, status)
			}
			time.Sleep(w.every)
		}
	}
	return nil
}

// status is what mem0 says of an event, and whether it may still change: true for PENDING,
// RUNNING and an answer that is no answer yet.
func (w mem0Extraction) status(event string) (string, bool) {
	request, err := http.NewRequest(http.MethodGet, w.baseURL+event+"/", nil)
	if err != nil {
		return err.Error(), false
	}
	request.Header.Set("Authorization", "Token "+w.apiKey)
	response, err := w.client.Do(request)
	if err != nil {
		return err.Error(), true
	}
	defer response.Body.Close()
	if response.StatusCode >= http.StatusInternalServerError {
		return response.Status, true
	}
	var answer struct {
		Status string `json:"status"`
	}
	if response.StatusCode != http.StatusOK || json.NewDecoder(response.Body).Decode(&answer) != nil {
		return response.Status, false
	}
	return answer.Status, answer.Status == "PENDING" || answer.Status == "RUNNING"
}

// queuedAs is the mem0 event a session's exchange was queued as, or empty until the router
// has handed it over.
func (w mem0Extraction) queuedAs(session string) string {
	for line := range strings.Lines(w.log()) {
		var record struct {
			Msg   string `json:"msg"`
			Event string `json:"event"`
			RunID string `json:"run_id"`
		}
		if json.Unmarshal([]byte(line), &record) == nil &&
			record.Msg == "queued a conversation to remember" && record.RunID == session {
			return record.Event
		}
	}
	return ""
}

// Mem0ExtractionSuite proves the wait fails at once on what waiting cannot change and keeps to
// its deadlines on what it can. mem0 is a stand-in that answers each event with a canned status.
type Mem0ExtractionSuite struct {
	suite.Suite
	answers  map[string]func(int) (int, string)
	handed   map[string]string
	asked    atomic.Int32
	wait     mem0Extraction
	finished time.Duration
}

func TestMem0ExtractionSuite(t *testing.T) { suite.Run(t, new(Mem0ExtractionSuite)) }

func (s *Mem0ExtractionSuite) SetupTest() {
	s.answers = map[string]func(int) (int, string){}
	s.handed = map[string]string{}
	s.asked.Store(0)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		event := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, "/"), "/")
		status, body := s.answers[event](int(s.asked.Add(1)))
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
	s.T().Cleanup(server.Close)
	s.wait = mem0Extraction{
		client: server.Client(), baseURL: server.URL + "/", apiKey: "key",
		log: func() string {
			var log strings.Builder
			for session, event := range s.handed {
				fmt.Fprintf(&log, `{"msg":"queued a conversation to remember","event":%q,"run_id":%q}`+"\n", event, session)
			}
			return log.String()
		},
		grace: 300 * time.Millisecond, bound: 2 * time.Second, every: 20 * time.Millisecond,
	}
}

func (s *Mem0ExtractionSuite) until(suiteDeadline time.Time, sessions ...string) error {
	started := time.Now()
	err := s.wait.until(sessions, suiteDeadline)
	s.finished = time.Since(started)
	return err
}

func (s *Mem0ExtractionSuite) TestASessionTheRouterNeverHandedOverFailsAtTheGrace() {
	err := s.until(time.Now().Add(time.Minute), "session-1")

	s.ErrorContains(err, "router never handed session session-1 to mem0")
	s.GreaterOrEqual(s.finished, s.wait.grace)
	s.Less(s.finished, s.wait.bound/2)
}

func (s *Mem0ExtractionSuite) TestASessionHandedOverLateIsWaitedFor() {
	s.answers["ev"] = func(int) (int, string) { return http.StatusOK, `{"status":"SUCCEEDED"}` }
	go func() {
		time.Sleep(100 * time.Millisecond)
		s.handed["session-1"] = "ev"
	}()

	s.Require().NoError(s.until(time.Now().Add(time.Minute), "session-1"))
}

func (s *Mem0ExtractionSuite) TestAFailedEventFailsAtOnce() {
	s.handed["session-1"] = "ev"
	s.answers["ev"] = func(int) (int, string) { return http.StatusOK, `{"status":"FAILED"}` }

	s.ErrorContains(s.until(time.Now().Add(time.Minute), "session-1"), `will not extract what session session-1 was told: event ev is "FAILED"`)
	s.Less(s.finished, s.wait.bound/2)
}

func (s *Mem0ExtractionSuite) TestARefusedRequestAndAnUnknownStatusFailAtOnce() {
	s.handed["session-1"] = "refused"
	s.handed["session-2"] = "unknown"
	s.answers["refused"] = func(int) (int, string) { return http.StatusNotFound, `{}` }
	s.answers["unknown"] = func(int) (int, string) { return http.StatusOK, `{"status":"WEIRD"}` }

	s.ErrorContains(s.until(time.Now().Add(time.Minute), "session-1"), "will not extract")
	s.Less(s.finished, s.wait.bound/2)
	s.ErrorContains(s.until(time.Now().Add(time.Minute), "session-2"), "will not extract")
	s.Less(s.finished, s.wait.bound/2)
}

func (s *Mem0ExtractionSuite) TestAnEventStillWorkedOnIsWaitedForUntilItSucceeds() {
	s.handed["session-1"] = "ev"
	s.answers["ev"] = func(n int) (int, string) {
		switch {
		case n < 3:
			return http.StatusOK, `{"status":"PENDING"}`
		case n < 5:
			return http.StatusOK, `{"status":"RUNNING"}`
		}
		return http.StatusOK, `{"status":"SUCCEEDED"}`
	}

	s.Require().NoError(s.until(time.Now().Add(time.Minute), "session-1"))
	s.GreaterOrEqual(int(s.asked.Load()), 5)
}

func (s *Mem0ExtractionSuite) TestAnEventThatStaysPendingFailsAtTheBound() {
	s.handed["session-1"] = "ev"
	s.answers["ev"] = func(int) (int, string) { return http.StatusOK, `{"status":"PENDING"}` }

	s.ErrorContains(s.until(time.Now().Add(time.Minute), "session-1"), "never extracted what session session-1 was told")
	s.GreaterOrEqual(s.finished, s.wait.bound)
}

func (s *Mem0ExtractionSuite) TestTheSuitesDeadlineEndsTheWaitBeforeTheBound() {
	s.handed["session-1"] = "ev"
	s.answers["ev"] = func(int) (int, string) { return http.StatusOK, `{"status":"PENDING"}` }

	s.ErrorContains(s.until(time.Now().Add(200*time.Millisecond), "session-1"), "never extracted")
	s.Less(s.finished, s.wait.bound/2)
}
