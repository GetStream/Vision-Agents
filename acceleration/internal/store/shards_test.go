//go:build integration

package store

import (
	"encoding/json"
	"time"
)

func (s *ShardedStoreSuite) TestACallAnotherShardRanIsListedAndFound() {
	s.Require().NoError(s.store.StartCall(s.ctx, &Call{ID: "here", CustomerID: "app", StartedAt: s.base}))
	s.Require().NoError(s.other.StartCall(s.ctx, &Call{ID: "there", CustomerID: "app", StartedAt: s.base.Add(time.Minute)}))

	calls, err := s.store.CustomerCalls(s.ctx, "app", CallFilter{})
	s.Require().NoError(err)
	s.Require().Len(calls, 2)
	s.Equal("there", calls[0].ID, "newest first, whichever shard ran it")
	s.Equal("here", calls[1].ID)

	found, err := s.store.Call(s.ctx, "app", "there")
	s.Require().NoError(err)
	s.Equal("there", found.ID)
}

func (s *ShardedStoreSuite) TestSpendIsCountedInEveryShard() {
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{Modality: "llm", CustomerID: "app", Provider: "openai", Model: "gpt", StartedAt: s.base, CostMicros: 300}))
	s.Require().NoError(s.other.RecordRequest(s.ctx, &Request{Modality: "llm", CustomerID: "app", Provider: "openai", Model: "gpt", StartedAt: s.base, CostMicros: 200}))

	spent, err := s.store.SpendSince(s.ctx, "app", s.base.Add(-time.Hour))
	s.Require().NoError(err)
	s.EqualValues(500, spent)
}

func (s *ShardedStoreSuite) TestARequestsLogIsInTheSharedDatabase() {
	s.Require().NoError(s.other.StartCall(s.ctx, &Call{ID: "call", CustomerID: "app", AgentID: "agent", ConfigID: "cfg", StartedAt: s.base}))
	s.Require().NoError(s.other.RecordRequest(s.ctx, &Request{Modality: "tts", CustomerID: "app", AgentID: "agent", Provider: "cartesia", Model: "sonic", StartedAt: s.base.Add(time.Second), Success: true}))

	logs, err := s.store.AgentLogs(s.ctx, LogFilter{CustomerID: "app", Limit: 10})
	s.Require().NoError(err)
	s.Require().Len(logs, 1, "one row, in the shared database, rather than one in the shard as well")
	s.Equal("tts request completed", logs[0].Message)
	s.Equal("cfg", logs[0].ConfigID)
	s.Equal("call", logs[0].SessionID)

	var inShard int
	s.Require().NoError(s.other.shard.NewSelect().Table("agent_logs").ColumnExpr("count(*)").Scan(s.ctx, &inShard))
	s.Zero(inShard)
}

// requestLogged is the log row a request the agent made writes, in a shard. Each test names its
// own customer, since nothing clears the log between tests.
func (s *ShardedStoreSuite) requestLogged(customer string, success bool, errorCode string) AgentLog {
	s.Require().NoError(s.other.StartCall(s.ctx, &Call{ID: "call", CustomerID: customer, AgentID: "agent", ConfigID: "cfg", StartedAt: s.base}))
	s.Require().NoError(s.other.RecordRequest(s.ctx, &Request{Modality: "llm", CustomerID: customer, AgentID: "agent", Provider: "openai", Model: "gpt",
		StartedAt: s.base.Add(time.Second), InputTokens: 10, OutputTokens: 4, Success: success, ErrorCode: errorCode}))
	logs, err := s.store.AgentLogs(s.ctx, LogFilter{CustomerID: customer, Limit: 10})
	s.Require().NoError(err)
	s.Require().Len(logs, 1)
	logged, err := s.store.AgentLog(s.ctx, customer, logs[0].ID)
	s.Require().NoError(err)
	return *logged
}

func (s *ShardedStoreSuite) TestAFailedRequestIsLoggedAsAnError() {
	logged := s.requestLogged("failed-log", false, "provider_error")

	s.Equal("llm request failed", logged.Message)
	s.Equal("error", logged.Severity)
	s.Equal("provider_error", logged.Details["error_code"])
}

func (s *ShardedStoreSuite) TestARequestItsCallerCancelledIsLoggedAsNoError() {
	logged := s.requestLogged("cancelled-log", false, ErrorCancelled)

	s.Equal("llm request cancelled", logged.Message)
	s.Equal("info", logged.Severity, "a superseded flow decision or a barge-in is routine")
	s.Equal(ErrorCancelled, logged.Details["error_code"])
	s.EqualValues(4, logged.Details["output_tokens"], "what it generated is still there")
}

func (s *ShardedStoreSuite) TestASessionCarriedOnInAnotherShardIsListedOnce() {
	s.opened("chat", "app", s.base, func(session *AgentSession) { session.Title = "before" })
	s.Require().NoError(s.store.CloseSession(s.ctx, "chat", s.base.Add(time.Minute)))
	carried := &AgentSession{ID: "chat", CustomerID: "app", AgentName: "docs", UserID: "jlahey", CallerKind: "user", Title: "after", CreatedAt: s.base}
	s.Require().NoError(s.other.SaveSession(s.ctx, carried))

	sessions, err := s.store.QuerySessions(s.ctx, "app", SessionFilter{})
	s.Require().NoError(err)
	s.Require().Len(sessions, 1)
	s.Equal("after", sessions[0].Title, "the row it was carried on in is newer")
	s.Equal(SessionRunning, sessions[0].State)
}

func (s *ShardedStoreSuite) TestSessionsOpenInTwoShardsAtOnceAreOnePeak() {
	s.opened("here", "app", s.base, nil)
	s.Require().NoError(s.store.CloseSession(s.ctx, "here", s.base.Add(10*time.Minute)))
	there := &AgentSession{ID: "there", CustomerID: "app", AgentName: "docs", UserID: "jlahey", CallerKind: "user", CreatedAt: s.base.Add(5 * time.Minute)}
	s.Require().NoError(s.other.SaveSession(s.ctx, there))
	s.Require().NoError(s.other.CloseSession(s.ctx, "there", s.base.Add(20*time.Minute)))

	peak, err := s.store.CustomerConcurrencyPeak(s.ctx, "app", s.base.Add(-time.Hour), s.base.Add(time.Hour))
	s.Require().NoError(err)
	s.EqualValues(2, peak.Sessions)
	s.Require().NotNil(peak.At)
	s.True(peak.At.Equal(s.base.Add(5 * time.Minute)))
}

func (s *ShardedStoreSuite) TestAGuestsSessionsInEveryShardAreClaimed() {
	s.Require().NoError(s.store.RecordGuest(s.ctx, &User{ID: "guest-1", CustomerID: "app", CreatedAt: s.base}))
	s.opened("here", "app", s.base, func(session *AgentSession) { session.UserID = "guest-1" })
	there := &AgentSession{ID: "there", CustomerID: "app", AgentName: "docs", UserID: "guest-1", CallerKind: "guest", CreatedAt: s.base}
	s.Require().NoError(s.other.SaveSession(s.ctx, there))

	moved, err := s.store.ClaimGuest(s.ctx, "app", "guest-1", "jlahey")
	s.Require().NoError(err)
	s.EqualValues(2, moved)

	again, err := s.store.ClaimGuest(s.ctx, "app", "guest-1", "jlahey")
	s.Require().NoError(err, "the same user claiming again finishes a claim rather than failing")
	s.Zero(again)
	_, err = s.store.ClaimGuest(s.ctx, "app", "guest-1", "randy")
	s.Error(err)
}

func (s *ShardedStoreSuite) TestAnExportOfTwoShardsFoldsTheirRollups() {
	latency := 100.0
	s.Require().NoError(s.store.RecordRequest(s.ctx, &Request{Modality: "stt", CustomerID: "app", Provider: "deepgram", Model: "flux", StartedAt: s.base, AudioMs: 1000, LatencyMs: &latency, Success: true}))
	slower := 300.0
	s.Require().NoError(s.other.RecordRequest(s.ctx, &Request{Modality: "stt", CustomerID: "app", Provider: "deepgram", Model: "flux", StartedAt: s.base, AudioMs: 500, LatencyMs: &slower, Success: true}))
	_, err := s.store.Rollup(s.ctx, Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)
	_, err = s.other.Rollup(s.ctx, Hourly, s.base, s.base.Add(time.Hour))
	s.Require().NoError(err)

	var hours []map[string]any
	requests := 0
	_, err = s.store.ExportCustomer(s.ctx, "app", func(table string, row json.RawMessage) error {
		switch table {
		case "stats_hourly":
			var hour map[string]any
			s.Require().NoError(json.Unmarshal(row, &hour))
			hours = append(hours, hour)
		case "requests":
			requests++
		}
		return nil
	})
	s.Require().NoError(err)
	s.Equal(2, requests, "every shard's requests")
	s.Require().Len(hours, 1, "one row for the hour both shards counted")
	s.EqualValues(1500, hours[0]["audio_ms_total"])
	s.EqualValues(2, hours[0]["request_count"])
	s.EqualValues(200, hours[0]["latency_p50_ms"])
}
