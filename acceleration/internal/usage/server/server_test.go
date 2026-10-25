//go:build integration

package server_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/server"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

// ServerSuite calls the real service over HTTP, on a database of its own and the suites' Redis.
type ServerSuite struct {
	suite.Suite
	ctx     context.Context
	dsn     string
	redis   rueidis.Client
	service *usagetest.Service
	logs    *lockedBuffer
	// organization is this test's own, so its Redis keys are too.
	organization string
}

func TestServerSuite(t *testing.T) {
	suite.Run(t, new(ServerSuite))
}

func (s *ServerSuite) SetupSuite() {
	s.ctx = context.Background()
	s.dsn = usagetest.Database(s.T(), "usage_server")
	s.redis = usagetest.Redis(s.T())
}

func (s *ServerSuite) SetupTest() {
	s.organization = usagetest.Organization()
	s.logs = &lockedBuffer{}
	s.service = usagetest.Start(s.T(), usagetest.Options{
		DSN: s.dsn, RedisURL: usagetest.RedisURL(s.T()), Logger: slog.New(slog.NewTextHandler(s.logs, nil)),
		ManageKey: "manage-" + uuid.NewString(),
	})
	usagetest.Empty(s.T(), s.service.Ledger)
}

func (s *ServerSuite) TearDownTest() {
	usagetest.Forget(s.T(), s.redis, s.organization)
}

// post sends body to path with token, and returns the status and the answer.
func (s *ServerSuite) post(path, token string, body any) (int, []byte) {
	payload, ok := body.([]byte)
	if !ok {
		var err error
		payload, err = json.Marshal(body)
		s.Require().NoError(err)
	}
	request, err := http.NewRequestWithContext(s.ctx, http.MethodPost, s.service.URL+path, bytes.NewReader(payload))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close() //nolint:errcheck // read-only body
	var answer bytes.Buffer
	_, err = answer.ReadFrom(response.Body)
	s.Require().NoError(err)
	return response.StatusCode, answer.Bytes()
}

func (s *ServerSuite) ship(events ...usage.Event) usage.Ack {
	status, body := s.post(usage.PathEvents, s.service.Token, usage.Batch{Events: events})
	s.Require().Equal(http.StatusOK, status, string(body))
	var ack usage.Ack
	s.Require().NoError(json.Unmarshal(body, &ack))
	return ack
}

func (s *ServerSuite) stored() []ledger.UsageEvent {
	var rows []ledger.UsageEvent
	s.Require().NoError(s.service.Ledger.DB().NewSelect().Model(&rows).Order("occurred_at", "usage_id").Scan(s.ctx))
	return rows
}

func (s *ServerSuite) TestTheIngestTokenIsRequired() {
	for _, path := range []string{usage.PathEvents, usage.PathWindows, usage.PathStandings} {
		for _, token := range []string{"", "wrong", s.service.Token + "x"} {
			status, body := s.post(path, token, map[string]any{})
			s.Equal(http.StatusUnauthorized, status, "%s with %q: %s", path, token, body)
		}
	}
	s.Empty(s.stored())
}

func (s *ServerSuite) TestTheHealthCheckTakesNoToken() {
	response, err := http.Get(s.service.URL + "/healthz")
	s.Require().NoError(err)
	defer response.Body.Close() //nolint:errcheck // read-only body

	s.Equal(http.StatusOK, response.StatusCode)
}

func (s *ServerSuite) TestAnInvalidEventIsRejectedAndTheRestRecorded() {
	for reason, broken := range map[string]func(*usage.Event){
		usage.ReasonOrganization: func(e *usage.Event) { e.OrganizationID = "" },
		usage.ReasonAPIKey:       func(e *usage.Event) { e.APIKey = "" },
	} {
		before := testutil.ToFloat64(server.EventsRejected.WithLabelValues(reason))
		bad := usagetest.Event(s.organization, broken)

		ack := s.ship(usagetest.Event(s.organization), bad, usagetest.Event(s.organization))

		s.Equal(2, ack.Accepted, reason)
		s.Zero(ack.Duplicates, reason)
		s.Require().Len(ack.Rejected, 1, reason)
		s.Equal(bad.UsageID, ack.Rejected[0].UsageID)
		s.Equal(reason, ack.Rejected[0].Reason)
		s.Equal(before+1, testutil.ToFloat64(server.EventsRejected.WithLabelValues(reason)))
		s.Contains(s.logs.String(), `msg="usage event rejected" reason=`+reason)
		s.Contains(s.logs.String(), bad.UsageID.String(), "the whole event is logged")
		for _, row := range s.stored() {
			s.NotEqual(bad.UsageID, row.UsageID)
		}
	}
	s.Len(s.stored(), 4)
}

func (s *ServerSuite) TestAnIDTooLongIsRejectedAndTheRestRecorded() {
	before := testutil.ToFloat64(server.EventsRejected.WithLabelValues(usage.ReasonIDLength))
	long := usagetest.Event(s.organization, func(e *usage.Event) { e.UserID = strings.Repeat("u", 3000) })

	ack := s.ship(usagetest.Event(s.organization), long, usagetest.Event(s.organization))

	s.Equal(2, ack.Accepted)
	s.Require().Len(ack.Rejected, 1)
	s.Equal(usage.Rejection{Index: 1, UsageID: long.UsageID, Reason: usage.ReasonIDLength, Message: "user_id is 3000 bytes, at most 255"}, ack.Rejected[0])
	s.Equal(before+1, testutil.ToFloat64(server.EventsRejected.WithLabelValues(usage.ReasonIDLength)))
	s.Len(s.stored(), 2)
}

func (s *ServerSuite) TestTheRejectedEventIsLoggedByItsPlaceInTheBatch() {
	broken := usagetest.Event(s.organization, func(e *usage.Event) { e.APIKey, e.Source = "", "router/broken/pod" })
	sameID := usagetest.Event(s.organization, func(e *usage.Event) { e.UsageID = broken.UsageID })

	ack := s.ship(broken, sameID)

	s.Equal(1, ack.Accepted)
	s.Require().Len(ack.Rejected, 1)
	s.Equal(0, ack.Rejected[0].Index)
	s.Contains(s.logs.String(), `msg="usage event rejected" reason=missing_api_key`)
	s.Contains(s.logs.String(), "index=0")
	s.Contains(s.logs.String(), "router/broken/pod", "the event rejected is logged, not the one sharing its id")
}

func (s *ServerSuite) TestAMalformedEventIsRejectedAloneAndTheRestRecorded() {
	before := testutil.ToFloat64(server.EventsRejected.WithLabelValues(usage.ReasonMalformed))
	good, other := usagetest.Event(s.organization), usagetest.Event(s.organization)
	id := uuid.New()
	encode := func(event usage.Event) string {
		encoded, err := json.Marshal(event)
		s.Require().NoError(err)
		return string(encoded)
	}
	body := `{"events":[` + encode(good) + `,{"usage_id":"` + id.String() + `","cost_micros":"lots"},` + encode(other) + `,7]}`

	status, answer := s.post(usage.PathEvents, s.service.Token, []byte(body))

	s.Require().Equal(http.StatusOK, status, string(answer))
	var ack usage.Ack
	s.Require().NoError(json.Unmarshal(answer, &ack))
	s.Equal(2, ack.Accepted)
	s.Require().Len(ack.Rejected, 2)
	s.Equal([]int{1, 3}, []int{ack.Rejected[0].Index, ack.Rejected[1].Index})
	s.Equal(id, ack.Rejected[0].UsageID, "the id is answered when it decodes")
	s.Equal(usage.ReasonMalformed, ack.Rejected[0].Reason)
	s.Contains(ack.Rejected[0].Message, "cost_micros")
	s.Equal(uuid.Nil, ack.Rejected[1].UsageID)
	s.Equal(before+2, testutil.ToFloat64(server.EventsRejected.WithLabelValues(usage.ReasonMalformed)))
	s.Contains(s.logs.String(), `\"cost_micros\":\"lots\"`, "logged as sent")
	s.Len(s.stored(), 2)
}

func (s *ServerSuite) TestARejectedEventIsLoggedUpToSixteenKiB() {
	body := `{"events":[{"usage_id":"` + uuid.NewString() + `","cost_micros":"lots","padding":"` + strings.Repeat("p", 1<<20) + `"}]}`

	status, answer := s.post(usage.PathEvents, s.service.Token, []byte(body))

	s.Require().Equal(http.StatusOK, status, string(answer))
	logged := s.logs.String()
	s.Less(len(logged), 20<<10, "not the whole megabyte")
	s.Contains(logged, "... (truncated: 1048")
	s.Contains(logged, `\"cost_micros\":\"lots\"`, "the start of the event is there")
}

func (s *ServerSuite) TestABodyWithoutEventsIsRefused() {
	before := testutil.ToFloat64(server.BatchesRefused.WithLabelValues("422"))

	for _, body := range []string{`{}`, `{"events":null}`} {
		status, answer := s.post(usage.PathEvents, s.service.Token, []byte(body))
		s.Equal(http.StatusUnprocessableEntity, status, "%s: %s", body, answer)
	}
	status, answer := s.post(usage.PathEvents, s.service.Token, []byte(`{"events":[]}`))
	s.Equal(http.StatusOK, status, "an empty batch is a batch: %s", answer)

	s.Equal(before+2, testutil.ToFloat64(server.BatchesRefused.WithLabelValues("422")))
}

func (s *ServerSuite) TestOnlyABatchRefusedForWhatItHoldsPages() {
	checks := map[string]alerts.Check{}
	for _, condition := range server.Conditions() {
		checks[condition.Name] = condition.Check
	}
	now := time.Now()
	for _, name := range []string{"batch_refused", "batch_unauthorized"} {
		checks[name](now)
	}

	status, _ := s.post(usage.PathEvents, "wrong", usage.Batch{Events: []usage.Event{usagetest.Event(s.organization)}})
	s.Require().Equal(http.StatusUnauthorized, status)
	refused, _ := checks["batch_refused"](now)
	unauthorized, _ := checks["batch_unauthorized"](now)
	s.False(refused, "a wrong token does not page")
	s.True(unauthorized)

	status, _ = s.post(usage.PathEvents, s.service.Token, []byte(`{"events":`))
	s.Require().Equal(http.StatusUnprocessableEntity, status)
	refused, _ = checks["batch_refused"](now)
	unauthorized, _ = checks["batch_unauthorized"](now)
	s.True(refused, "a batch that is not one pages")
	s.False(unauthorized)
}

func (s *ServerSuite) TestABodyThatIsNotJSONIsRefusedWithoutBeingRepeated() {
	before := testutil.ToFloat64(server.BatchesRefused.WithLabelValues("422"))
	body := `{"events":[{"usage_id":"` + uuid.NewString() + `","padding":"` + strings.Repeat("p", 4096) + `"`

	status, answer := s.post(usage.PathEvents, s.service.Token, []byte(body))

	s.Equal(http.StatusUnprocessableEntity, status, string(answer))
	s.NotContains(string(answer), "ppppp")
	s.Less(len(answer), 1024)
	s.Equal(before+1, testutil.ToFloat64(server.BatchesRefused.WithLabelValues("422")))
	s.Empty(s.stored())
}

func (s *ServerSuite) TestAnEventKeepsItsTierUserAndCategories() {
	full := usagetest.Event(s.organization, func(e *usage.Event) {
		e.OrganizationTier, e.APIKey, e.UserID = "tier-a", "pk_live", "user@example.com"
		e.UsageCategories = []string{"team:finance", "project:onboarding"}
	})
	bare := usagetest.Event(s.organization, func(e *usage.Event) { e.APIKey = "hook:chat" })
	crowded := usagetest.Event(s.organization, func(e *usage.Event) {
		for i := range 11 {
			e.UsageCategories = append(e.UsageCategories, fmt.Sprintf("c%d", i))
		}
	})
	piped := usagetest.Event(s.organization, func(e *usage.Event) { e.UserID = "a|b" })

	ack := s.ship(full, bare, crowded, piped)

	s.Equal(2, ack.Accepted)
	s.Equal([]string{usage.ReasonCategories, usage.ReasonSeparator}, []string{ack.Rejected[0].Reason, ack.Rejected[1].Reason})
	rows := map[uuid.UUID]ledger.UsageEvent{}
	for _, row := range s.stored() {
		rows[row.UsageID] = row
	}
	s.Require().Len(rows, 2)
	kept := rows[full.UsageID]
	s.Equal("tier-a", kept.OrganizationTier)
	s.Equal("pk_live", kept.APIKey)
	s.Equal("user@example.com", kept.UserID)
	s.Equal([]string{"project:onboarding", "team:finance"}, kept.UsageCategories)
	s.Equal("", rows[bare.UsageID].UserID)
	s.Empty(rows[bare.UsageID].UsageCategories)
}

func (s *ServerSuite) TestABatchSentTwiceIsAllDuplicatesTheSecondTime() {
	batch := []usage.Event{usagetest.Event(s.organization), usagetest.Event(s.organization)}

	s.Equal(usage.Ack{Accepted: 2, Duplicates: 0, Rejected: []usage.Rejection{}, Mirrored: true}, s.ship(batch...))
	s.Equal(usage.Ack{Accepted: 0, Duplicates: 2, Rejected: []usage.Rejection{}, Mirrored: true}, s.ship(batch...))
	s.Len(s.stored(), 2)
}

func (s *ServerSuite) TestABatchTooLargeIsRefused() {
	refused := testutil.ToFloat64(server.BatchesRefused.WithLabelValues("413"))
	events := make([]usage.Event, usage.MaxEvents+1)
	for i := range events {
		events[i] = usagetest.Event(s.organization)
	}
	status, body := s.post(usage.PathEvents, s.service.Token, usage.Batch{Events: events})
	s.Equal(http.StatusRequestEntityTooLarge, status, string(body))

	heavy := usagetest.Event(s.organization, func(e *usage.Event) {
		e.Attributes = map[string]string{"padding": strings.Repeat("x", usage.MaxBodyBytes)}
	})
	status, body = s.post(usage.PathEvents, s.service.Token, usage.Batch{Events: []usage.Event{heavy}})
	s.Equal(http.StatusRequestEntityTooLarge, status, string(body))
	s.Empty(s.stored())
	s.Equal(refused+2, testutil.ToFloat64(server.BatchesRefused.WithLabelValues("413")))
}

func (s *ServerSuite) TestAProducerNewerThanTheServiceIsStillRecorded() {
	event, err := json.Marshal(usagetest.Event(s.organization))
	s.Require().NoError(err)
	newer := bytes.Replace(event, []byte(`{`), []byte(`{"field_from_the_future":1,`), 1)

	status, body := s.post(usage.PathEvents, s.service.Token, []byte(`{"events":[`+string(newer)+`],"also_new":true}`))

	s.Equal(http.StatusOK, status, string(body))
	s.Len(s.stored(), 1)
}

func (s *ServerSuite) TestAnEventWithNoQuantitiesIsRecorded() {
	ack := s.ship(usagetest.Event(s.organization, func(e *usage.Event) { e.Quantities = nil }))

	s.Equal(1, ack.Accepted)
}

func (s *ServerSuite) TestADatabaseFailureIsA503TheProducerRetries() {
	before := testutil.ToFloat64(server.IngestErrors)
	rejected := testutil.ToFloat64(server.EventsRejected.WithLabelValues(usage.ReasonAPIKey))
	broken := usagetest.Start(s.T(), usagetest.Options{DSN: s.dsn, Logger: slog.New(slog.NewTextHandler(s.logs, nil))})
	s.Require().NoError(broken.Ledger.Close())

	_, err := broken.Client.Ship(s.ctx, []usage.Event{
		usagetest.Event(s.organization, func(e *usage.Event) { e.APIKey = "" }), usagetest.Event(s.organization),
	})

	var failed *usage.StatusError
	s.Require().ErrorAs(err, &failed)
	s.Equal(http.StatusServiceUnavailable, failed.Status)
	s.Equal(before+1, testutil.ToFloat64(server.IngestErrors))
	s.Equal(rejected, testutil.ToFloat64(server.EventsRejected.WithLabelValues(usage.ReasonAPIKey)),
		"a rejection in a batch that is sent again is counted when that one lands")
	s.NotContains(s.logs.String(), "usage event rejected")
	response, err := http.Get(broken.URL + "/healthz")
	s.Require().NoError(err)
	defer response.Body.Close() //nolint:errcheck // read-only body
	s.Equal(http.StatusServiceUnavailable, response.StatusCode)
}

func (s *ServerSuite) TestWindowsAnswersFromPostgresAndRewritesRedis() {
	event := usagetest.Event(s.organization, func(e *usage.Event) { e.AppID, e.UserID, e.CostMicros = "a1", "u42", 700 })
	s.ship(event, usagetest.Event(s.organization, func(e *usage.Event) { e.AppID, e.UserID, e.CostMicros = "a1", "u42", 300 }))
	user := usage.SubjectRef{OrganizationID: s.organization, Subject: usage.SubjectUser, AppID: "a1", UserID: "u42"}
	s.Require().NoError(s.redis.Do(s.ctx, s.redis.B().Del().Key(user.Key(ledger.KeyPrefix)).Build()).Error())

	windows, err := s.service.Client.Windows(s.ctx, []usage.SubjectRef{user})

	s.Require().NoError(err)
	s.Require().Len(windows, 4)
	fields := map[string]string{}
	for i, window := range windows {
		s.Equal(user, window.SubjectRef)
		s.Equal(usage.Intervals[i], window.Interval)
		start, end := usage.Window(window.Interval, time.Now())
		s.Equal(start, window.WindowStart)
		s.Equal(end, window.WindowEnd)
		s.Equal(int64(1000), window.SpentMicros)
		s.Equal(int64(2), window.Events)
		fields[usage.Field(window.Interval, window.WindowStart)] = "1000"
	}
	stored, err := s.redis.Do(s.ctx, s.redis.B().Hgetall().Key(user.Key(ledger.KeyPrefix)).Build()).AsStrMap()
	s.Require().NoError(err)
	s.Equal(fields, stored, "the key is back")
}

func (s *ServerSuite) TestWindowsAnswersZeroForASubjectThatNeverSpent() {
	never := usage.SubjectRef{OrganizationID: s.organization, Subject: usage.SubjectCategory, Category: "team:nobody"}

	windows, err := s.service.Client.Windows(s.ctx, []usage.SubjectRef{never})

	s.Require().NoError(err)
	s.Require().Len(windows, 4)
	for _, window := range windows {
		s.Equal(never, window.SubjectRef)
		s.Zero(window.SpentMicros)
		s.Zero(window.Events)
	}
	stored, err := s.redis.Do(s.ctx, s.redis.B().Hvals().Key(never.Key(ledger.KeyPrefix)).Build()).AsStrSlice()
	s.Require().NoError(err)
	s.Equal([]string{"0", "0", "0", "0"}, stored, "a subject that never spent has a key, so it is not read through again")
}

func (s *ServerSuite) TestAWindowsReadOfAMalformedSubjectIsRefused() {
	status, body := s.post(usage.PathWindows, s.service.Token, usage.WindowsQuery{Subjects: []usage.SubjectRef{
		{OrganizationID: s.organization, Subject: usage.SubjectOrganization},
		{OrganizationID: s.organization, Subject: usage.SubjectUser, AppID: "a1"},
	}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), "subjects[1]: user_id is required for subject user")
}

func (s *ServerSuite) TestAWindowsReadOfTooManySubjectsIsRefused() {
	subjects := make([]usage.SubjectRef, usage.MaxSubjects+1)
	for i := range subjects {
		subjects[i] = usage.SubjectRef{OrganizationID: s.organization, Subject: usage.SubjectApp, AppID: fmt.Sprint(i)}
	}

	status, _ := s.post(usage.PathWindows, s.service.Token, usage.WindowsQuery{Subjects: subjects})

	s.Equal(http.StatusRequestEntityTooLarge, status)
}

func (s *ServerSuite) TestStandingsAnswerTheOrganizationsAskedAbout() {
	standings, err := s.service.Client.Standings(s.ctx, usage.StandingQuery{Organizations: []usage.StandingOrganization{
		{OrganizationID: "o2", OrganizationTier: "tier-a"}, {OrganizationID: "o1"},
	}})

	s.Require().NoError(err)
	s.Require().Len(standings, 2)
	s.Equal([]string{"o2", "o1"}, []string{standings[0].OrganizationID, standings[1].OrganizationID})
	s.Equal([]string{"tier-a", ""}, []string{standings[0].OrganizationTier, standings[1].OrganizationTier})
}

func (s *ServerSuite) TestAStandingsReadWithoutAnOrganizationIsRefused() {
	status, body := s.post(usage.PathStandings, s.service.Token, usage.StandingQuery{Organizations: []usage.StandingOrganization{{}}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), "organizations[0]: organization_id is required")
}

func (s *ServerSuite) TestAnEventsLagIsMeasured() {
	aboveBefore, allBefore := alerts.Above(server.EventLag, 600)

	s.ship(usagetest.Event(s.organization, func(e *usage.Event) { e.OccurredAt = time.Now().Add(-20 * time.Minute) }),
		usagetest.Event(s.organization),
		usagetest.Event(s.organization, func(e *usage.Event) {
			e.OccurredAt, e.Source = time.Now().Add(-3*time.Hour), "job/hourly-rollup/pod"
		}))

	above, all := alerts.Above(server.EventLag, 600)
	s.Equal(allBefore+2, all, "a router's events alone: a rollup is late by design")
	s.Equal(aboveBefore+1, above, "the event twenty minutes late")
}

type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}
