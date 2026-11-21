//go:build integration

package limitnotice

import (
	"context"
	"log/slog"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/activity"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// DigestSuite posts the activity digest from a database of its own, emptied before each test,
// since the digest counts every organization's activity and another package's suite writes
// some. Slack is an HTTP server in the test process that keeps what it is sent.
type DigestSuite struct {
	suite.Suite
	ctx   context.Context
	db    *store.Store
	slack *slackServer
	// clock is the time the checks run at; reads counts how many times they asked it, once as
	// each check starts.
	clock atomic.Pointer[time.Time]
	reads atomic.Int64
}

func TestDigestSuite(t *testing.T) {
	suite.Run(t, new(DigestSuite))
}

func (s *DigestSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN must be set")
	}
	s.ctx = context.Background()
	db, err := store.Open(testenv.Database(dsn, "digest"))
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.db = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })
}

func (s *DigestSuite) SetupTest() {
	_, err := s.db.DB().ExecContext(s.ctx, "TRUNCATE customer_activity, activity_digests, requests, cost_anomaly_notices")
	s.Require().NoError(err)
	s.slack = newSlackServer()
	s.T().Cleanup(s.slack.Close)
	at := today
	s.clock.Store(&at)
	s.reads.Store(0)
}

// router is one router's check, posting the digest to the suite's Slack every few
// milliseconds, closed when the test ends.
func (s *DigestSuite) router() *Anomalies {
	slack := NewSlack("xoxb-test", "C123")
	slack.baseURL = s.slack.URL
	anomalies := NewAnomalies(AnomalyOptions{
		Store: s.db, Poster: slack, Environment: "prod", Logger: slog.New(slog.DiscardHandler),
		Every: 20 * time.Millisecond, Digest: true,
		Now: func() time.Time {
			s.reads.Add(1)
			return *s.clock.Load()
		},
	})
	s.T().Cleanup(anomalies.Close)
	return anomalies
}

// checked waits until the checks running have read the clock past reads more than once, so
// at least one whole check ran after them.
func (s *DigestSuite) checked(reads int64) {
	s.Require().Eventually(func() bool { return s.reads.Load() >= reads+2 }, settleFor, time.Millisecond)
}

// active records requests, refused of them, of an app of organizationID in area on the day
// daysAgo days before today. An empty organizationID is an app no organization is known for.
func (s *DigestSuite) active(organizationID, area string, daysAgo int, requests, refused int64) {
	day := time.Date(today.Year(), today.Month(), today.Day(), 12, 0, 0, 0, time.UTC).AddDate(0, 0, -daysAgo)
	delta := store.ActivityDelta{
		OrganizationID: organizationID, AppID: "app-" + organizationID, Area: area, Client: store.AuditSourceAPI,
		Day: day, Requests: requests, Refused: refused, FirstAt: day, LastAt: day,
	}
	if organizationID == "" {
		delta.AppID = "app-" + organization()
	}
	if refused > 0 {
		delta.LastRefusalCode, delta.LastRefusalAt = "invalid_api_key", day
	}
	s.Require().NoError(s.db.RecordActivity(s.ctx, []store.ActivityDelta{delta}))
}

// digests is the digests posted.
func (s *DigestSuite) digests() []string {
	var found []string
	for _, post := range s.slack.posts() {
		if strings.HasPrefix(post, "Agents yesterday") {
			found = append(found, post)
		}
	}
	return found
}

// Defect: every active organization listed as new, an app with no organization counted as
// one, or the active count other than the distinct organizations of the day.
func (s *DigestSuite) TestAnOrganizationIsNewOnItsFirstDayAndTheCountIsOfOrganizations() {
	fresh, seen, quiet := organization(), organization(), organization()
	s.active(fresh, activity.AreaAgentSetup, 1, 3, 0)
	s.active(fresh, activity.AreaConversations, 1, 2, 0)
	s.active(seen, activity.AreaConversations, 1, 4, 0)
	s.active(seen, activity.AreaDashboardReads, 2, 1, 0)
	s.active(quiet, activity.AreaDashboardReads, 2, 1, 0)
	s.active(quiet, activity.AreaDashboardReads, 0, 1, 0)
	s.active(organization(), activity.AreaAgentSetup, 2, 1, 0)
	s.active("", activity.AreaHooksIn, 1, 5, 0)
	s.active("", activity.AreaHooksIn, 2, 5, 0)

	s.router().Start()
	s.checked(0)

	s.Require().Len(s.digests(), 1)
	digest := s.digests()[0]
	s.Contains(digest, "Agents yesterday (9 Mar, prod)\nNew orgs (first request yesterday): 1\n"+
		"  • `"+fresh+"`   reached: conversations   agent_setup (api) → conversations (api)\n")
	newOrganizations, _, _ := strings.Cut(digest, "\nAll requests refused")
	s.NotContains(newOrganizations, seen, "active the day before, so not new")
	s.NotContains(digest, quiet, "not active yesterday")
	var counted int
	s.Require().NoError(s.db.DB().NewRaw("SELECT count(DISTINCT organization_id) FROM customer_activity"+
		" WHERE day = '2026-03-09' AND organization_id <> ''").Scan(s.ctx, &counted))
	s.Equal(2, counted)
	s.True(strings.HasSuffix(digest, "\nActive orgs: 2 (day before: 3)"), digest)
}

// Defect: an organization with one request answered flagged as stuck.
func (s *DigestSuite) TestAnOrganizationRefusedEveryTimeIsFlaggedAndOneAnsweredOnceIsNot() {
	stuck, partly := organization(), organization()
	s.active(stuck, activity.AreaAgentSetup, 1, 14, 14)
	s.active(partly, activity.AreaAgentSetup, 1, 14, 13)

	s.router().Start()
	s.checked(0)

	s.Require().Len(s.digests(), 1)
	digest := s.digests()[0]
	s.Contains(digest, "  • `"+stuck+"`   reached: —   all 14 requests refused (invalid_api_key)\n")
	s.Contains(digest, "All requests refused: 1\n  • `"+stuck+"`   14 requests (invalid_api_key)\n")
	s.Contains(digest, "  • `"+partly+"`   reached: agent_setup   agent_setup (api)\n", "listed as new, not as refused")
}

// Defect: a duplicate digest from two routers at once, from a router started again the same
// day, or from every check of the day.
func (s *DigestSuite) TestTheDigestIsPostedOnceADayWhateverTheRouters() {
	s.active(organization(), activity.AreaAgentSetup, 1, 1, 0)

	first, second := s.router(), s.router()
	first.Start()
	second.Start()
	s.checked(0)
	s.checked(s.reads.Load())
	s.Len(s.digests(), 1, "two routers, a check each every tick, one post")
	first.Close()
	second.Close()

	s.router().Start()
	s.checked(s.reads.Load())
	s.Len(s.digests(), 1, "nor does a router started later the same day")
	s.aged()
	s.checked(s.reads.Load())
	s.Len(s.digests(), 1, "nor a check once its claim is old: it was posted")

	next := today.AddDate(0, 0, 1)
	s.clock.Store(&next)
	s.Require().Eventually(func() bool { return len(s.digests()) == 2 }, settleFor, time.Millisecond,
		"the next day has a digest of its own")
	s.Contains(s.digests()[1], "Agents yesterday (10 Mar, prod)")
}

// Defect: the marker decided by the lock alone, so routers that each claim the day at once,
// with no lock between them, both post.
func (s *DigestSuite) TestOnlyOneOfClaimsAtOnceWins() {
	day := time.Date(2026, 3, 9, 0, 0, 0, 0, time.UTC)
	var won atomic.Int64
	var claims sync.WaitGroup
	for range 8 {
		claims.Go(func() {
			claimed, err := s.db.ClaimActivityDigest(s.ctx, day, stale)
			s.NoError(err)
			if claimed {
				won.Add(1)
			}
		})
	}
	claims.Wait()

	s.EqualValues(1, won.Load())
}

// Defect: a claim taken again once posted, or one never posted left unclaimed for good.
func (s *DigestSuite) TestAClaimIsTakenAgainOnlyWhenStaleAndNotPosted() {
	day := time.Date(2026, 3, 9, 0, 0, 0, 0, time.UTC)
	claimed, err := s.db.ClaimActivityDigest(s.ctx, day, stale)
	s.Require().NoError(err)
	s.Require().True(claimed)

	claimed, err = s.db.ClaimActivityDigest(s.ctx, day, stale)
	s.Require().NoError(err)
	s.False(claimed, "its claim is fresh")

	s.aged()
	claimed, err = s.db.ClaimActivityDigest(s.ctx, day, stale)
	s.Require().NoError(err)
	s.True(claimed, "its claim went stale before a post")

	s.Require().NoError(s.db.MarkActivityDigestPosted(s.ctx, day))
	s.aged()
	claimed, err = s.db.ClaimActivityDigest(s.ctx, day, stale)
	s.Require().NoError(err)
	s.False(claimed, "posted")
}

// aged makes every digest's claim older than stale.
func (s *DigestSuite) aged() {
	_, err := s.db.DB().ExecContext(s.ctx, "UPDATE activity_digests SET claimed_at = now() - interval '1 hour'")
	s.Require().NoError(err)
}

func (s *DigestSuite) TestAPostThatFailedIsPostedByALaterCheck() {
	s.active(organization(), activity.AreaAgentSetup, 1, 1, 0)
	s.slack.fail.Store(true)

	s.router().Start()
	s.checked(0)
	s.Empty(s.digests())

	s.slack.fail.Store(false)
	s.checked(s.reads.Load())
	s.Empty(s.digests(), "its claim stands for a while")
	s.aged()
	s.Require().Eventually(func() bool { return len(s.digests()) == 1 }, settleFor, time.Millisecond)
	s.checked(s.reads.Load())
	s.Len(s.digests(), 1, "posted once it went through")
}

// Defect: the digest posted at midnight, before the routers wrote the day's last minute, or
// never on the check after that.
func (s *DigestSuite) TestTheDigestWaitsForTheDaysLastWrites() {
	s.active(organization(), activity.AreaAgentSetup, 1, 1, 0)
	midnight := time.Date(today.Year(), today.Month(), today.Day(), 0, 0, 0, 0, time.UTC)
	early := midnight.Add(digestSettle - time.Second)
	s.clock.Store(&early)

	s.router().Start()
	s.checked(0)
	s.checked(s.reads.Load())
	s.Empty(s.digests())

	settled := midnight.Add(digestSettle)
	s.clock.Store(&settled)
	s.Require().Eventually(func() bool { return len(s.digests()) == 1 }, settleFor, time.Millisecond)
	s.Equal(5*time.Minute, digestSettle)
}

// Defect: the cost anomaly check posting a digest a router was not asked for.
func (s *DigestSuite) TestACheckWithoutTheDigestPostsNone() {
	s.active(organization(), activity.AreaAgentSetup, 1, 1, 0)
	slack := NewSlack("xoxb-test", "C123")
	slack.baseURL = s.slack.URL
	anomalies := NewAnomalies(AnomalyOptions{
		Store: s.db, Poster: slack, Logger: slog.New(slog.DiscardHandler), Every: 20 * time.Millisecond,
		Now: func() time.Time {
			s.reads.Add(1)
			return *s.clock.Load()
		},
	})
	anomalies.Start()
	defer anomalies.Close()
	s.checked(0)
	s.checked(s.reads.Load())

	s.Empty(s.slack.posts())
}

// appActive records requests, refused of them, of the app appID of organizationID in area on the
// day daysAgo days before today at hour, the latest refusal with code.
func (s *DigestSuite) appActive(organizationID, appID, area string, daysAgo int, requests, refused int64, code string, hour int) {
	day := time.Date(today.Year(), today.Month(), today.Day(), hour, 0, 0, 0, time.UTC).AddDate(0, 0, -daysAgo)
	delta := store.ActivityDelta{
		OrganizationID: organizationID, AppID: appID, Area: area, Client: store.AuditSourceAPI,
		Day: day, Requests: requests, Refused: refused, FirstAt: day, LastAt: day,
	}
	if refused > 0 {
		delta.LastRefusalCode, delta.LastRefusalAt = code, day
	}
	s.Require().NoError(s.db.RecordActivity(s.ctx, []store.ActivityDelta{delta}))
}

// Defect: the day before's count of organizations being a count of rows, so an organization with
// several apps and areas counts more than once and the comparison of AC-6 is wrong.
func (s *DigestSuite) TestTheDayBeforeCountsDistinctOrganizations() {
	org := organization()
	s.appActive(org, "app-a", activity.AreaAgentSetup, 2, 1, 0, "", 12)
	s.appActive(org, "app-a", activity.AreaDashboardReads, 2, 1, 0, "", 12)
	s.appActive(org, "app-b", activity.AreaDashboardReads, 2, 1, 0, "", 12)
	s.appActive(organization(), "app-c", activity.AreaAgentSetup, 1, 1, 0, "", 12)
	s.router().Start()
	s.checked(0)

	s.Require().Len(s.digests(), 1)
	s.True(strings.HasSuffix(s.digests()[0], "(day before: 1)"), s.digests()[0])
}

// Defect: the apps of one organization in one area not summed (the larger count taken, or each
// app kept apart), so an organization refused 14 times by one app and answered once by another
// is shown as refused every time.
func (s *DigestSuite) TestTheAppsOfAnOrganizationAreSummed() {
	org := organization()
	s.appActive(org, "app-a", activity.AreaAgentSetup, 1, 14, 14, "invalid_api_key", 10)
	s.appActive(org, "app-b", activity.AreaAgentSetup, 1, 1, 0, "", 11)
	s.router().Start()
	s.checked(0)

	s.Require().Len(s.digests(), 1)
	s.Contains(s.digests()[0], "  • `"+org+"`   reached: agent_setup   agent_setup (api)\n")
	s.Contains(s.digests()[0], "All requests refused: 0")
}

// Defect: the code named for an organization refused every time being the oldest refusal's, not
// the latest across its apps.
func (s *DigestSuite) TestTheCodeNamedIsTheLatestRefusalAcrossApps() {
	org := organization()
	s.appActive(org, "app-a", activity.AreaAgentSetup, 1, 2, 2, "older_code", 10)
	s.appActive(org, "app-b", activity.AreaAgentSetup, 1, 1, 1, "newer_code", 11)
	s.router().Start()
	s.checked(0)

	s.Require().Len(s.digests(), 1)
	s.Contains(s.digests()[0], "`"+org+"`   3 requests (newer_code)")
}

// Defect: an organization with a share of its requests refused not named because some went
// through, or one named for a few refusals among many answers, or one refused every time named
// as stuck too: the shares read wrong once the apps' rows are summed.
func (s *DigestSuite) TestAnOrganizationWithAQuarterOfItsRequestsRefusedIsListedAsStuck() {
	dental, noise, all := organization(), organization(), organization()
	s.appActive(dental, "app-"+dental, activity.AreaConversations, 1, 70, 30, "spend_limit_reached", 10)
	s.appActive(dental, "app-"+dental+"-b", activity.AreaConversations, 1, 30, 10, "spend_limit_reached", 11)
	s.appActive(dental, "app-"+dental, activity.AreaConversations, 2, 50, 50, "invalid_api_key", 10)
	s.appActive(noise, "app-"+noise, activity.AreaDashboardReads, 1, 100, 5, "not_found", 10)
	s.appActive(all, "app-"+all, activity.AreaAgentSetup, 1, 4, 4, "spend_limit_reached", 10)
	s.router().Start()
	s.checked(0)

	s.Require().Len(s.digests(), 1)
	digest := s.digests()[0]
	_, stuck, found := strings.Cut(digest, "\nStuck: 1\n")
	s.Require().True(found, digest)
	s.True(strings.HasPrefix(stuck, "  • `"+dental+"`   40% of 100 requests refused (spend_limit_reached)\nFirst conversation"), digest)
	s.NotContains(stuck, noise)
	s.Contains(digest, "All requests refused: 1\n  • `"+all+"`   4 requests (spend_limit_reached)\n")
}

// Defect: an organization's first conversation not named, named again on a later day, named
// once per app, counted from a refused conversation, listed again because its earlier
// conversation was another app's, or its days counted from its first conversation rather than
// its first request of any area.
func (s *DigestSuite) TestAnOrganizationsFirstAnsweredConversationIsListedWithTheDaysSinceItsFirstRequest() {
	dental, again, refusedFirst, sameDay, today, other, blocked, moved := organization(), organization(), organization(),
		organization(), organization(), organization(), organization(), organization()
	s.active(dental, activity.AreaAgentSetup, 4, 2, 0)
	s.active(dental, activity.AreaDashboardReads, 3, 1, 0)
	s.active(dental, activity.AreaConversations, 1, 5, 0)
	s.appActive(dental, "app-"+dental+"-b", activity.AreaConversations, 1, 2, 0, "", 12)
	s.active(again, activity.AreaAgentSetup, 3, 1, 0)
	s.active(again, activity.AreaConversations, 2, 1, 0)
	s.active(again, activity.AreaConversations, 1, 4, 0)
	s.active(refusedFirst, activity.AreaConversations, 2, 4, 4)
	s.active(refusedFirst, activity.AreaConversations, 1, 3, 1)
	s.active(sameDay, activity.AreaConversations, 1, 1, 0)
	s.active(today, activity.AreaAgentSetup, 2, 1, 0)
	s.active(today, activity.AreaConversations, 0, 1, 0)
	s.active(other, activity.AreaAgentSetup, 5, 1, 0)
	s.active(other, activity.AreaVoiceVideo, 1, 1, 0)
	s.active(blocked, activity.AreaConversations, 1, 2, 2)
	// An earlier conversation of another app of the same organization is the organization's
	// earlier one.
	s.appActive(moved, "app-"+moved, activity.AreaConversations, 2, 3, 0, "", 12)
	s.appActive(moved, "app-"+moved+"-b", activity.AreaConversations, 1, 2, 0, "", 12)
	s.active("", activity.AreaConversations, 1, 1, 0)
	s.router().Start()
	s.checked(0)

	s.Require().Len(s.digests(), 1)
	digest := s.digests()[0]
	_, first, found := strings.Cut(digest, "\nFirst conversation: 3\n")
	s.Require().True(found, digest)
	// Listed by organization id, as the ids are random.
	s.Equal(slices.Sorted(slices.Values([]string{
		"  • `" + dental + "`   3 days after its first request",
		"  • `" + refusedFirst + "`   1 day after its first request",
		"  • `" + sameDay + "`   0 days after its first request",
	})), strings.Split(strings.TrimSuffix(first, "\nActive orgs: 7 (day before: 4)"), "\n"), digest)
}

// days are the UTC days table holds rows of, oldest first.
func (s *DigestSuite) days(table string) []string {
	var days []string
	s.Require().NoError(s.db.DB().NewSelect().TableExpr(table).ColumnExpr("DISTINCT day::text").
		OrderExpr("1").Scan(s.ctx, &days))
	return days
}

// Defect: a keep other than 400 days, the day on its edge deleted, the digests' days kept for
// good, or a delete on every check rather than once a day. The days are written out rather than
// computed: 401 days before 10 Mar 2026 is 2 Feb 2025.
func (s *DigestSuite) TestActivityPastFourHundredDaysIsDeletedOnceADay() {
	pinned := organization()
	for _, daysAgo := range []int{401, 400, 399, 0} {
		s.active(pinned, activity.AreaAgentSetup, daysAgo, 1, 0)
	}
	for _, day := range []time.Time{
		time.Date(2025, 2, 2, 0, 0, 0, 0, time.UTC), time.Date(2025, 2, 3, 0, 0, 0, 0, time.UTC),
	} {
		claimed, err := s.db.ClaimActivityDigest(s.ctx, day, stale)
		s.Require().NoError(err)
		s.Require().True(claimed)
	}

	s.router().Start()
	s.checked(0)
	s.Equal([]string{"2025-02-03", "2025-02-04", "2026-03-10"}, s.days("customer_activity"))
	s.Equal([]string{"2025-02-03", "2026-03-09"}, s.days("activity_digests"), "the day before's posted by the check")

	s.active(organization(), activity.AreaAgentSetup, 401, 1, 0)
	later := today.Add(8 * time.Hour)
	s.clock.Store(&later)
	s.checked(s.reads.Load())
	s.Equal([]string{"2025-02-02", "2025-02-03", "2025-02-04", "2026-03-10"}, s.days("customer_activity"),
		"written after the day's delete, so kept until the next day's, though checked later that day")

	next := today.AddDate(0, 0, 1)
	s.clock.Store(&next)
	s.Require().Eventually(func() bool {
		return slices.Equal(s.days("customer_activity"), []string{"2025-02-04", "2026-03-10"})
	}, settleFor, time.Millisecond, "the day before 400 days ago went too")
	s.Require().Eventually(func() bool {
		return slices.Equal(s.days("activity_digests"), []string{"2026-03-09", "2026-03-10"})
	}, settleFor, time.Millisecond)
}

// Defect: the delete needed a Slack to post to, so a deployment without one kept every row; or a
// check with nowhere to post still posted, or noted anomalies nobody would hear of.
func (s *DigestSuite) TestActivityIsDeletedWithoutASlackAndNothingIsPosted() {
	pinned := organization()
	for _, daysAgo := range []int{401, 399, 1} {
		s.active(pinned, activity.AreaAgentSetup, daysAgo, 1, 0)
	}
	claimed, err := s.db.ClaimActivityDigest(s.ctx, time.Date(2025, 2, 2, 0, 0, 0, 0, time.UTC), stale)
	s.Require().NoError(err)
	s.Require().True(claimed)
	for daysAgo, cost := range []int64{40_000_000, 4_000_000, 4_000_000, 4_000_000, 4_000_000, 4_000_000, 4_000_000, 4_000_000} {
		s.Require().NoError(s.db.RecordRequest(s.ctx, &store.Request{
			Modality: "llm", CustomerID: "app-" + pinned, OrganizationID: pinned, Provider: "openai", Model: "m",
			StartedAt: time.Date(2026, 3, 10, 12, 0, 0, 0, time.UTC).AddDate(0, 0, -daysAgo), CostMicros: cost,
			Billable: true, Success: true,
		}))
	}

	anomalies := NewAnomalies(AnomalyOptions{
		Store: s.db, Logger: slog.New(slog.DiscardHandler), Every: 20 * time.Millisecond, Digest: true,
		Now: func() time.Time {
			s.reads.Add(1)
			return *s.clock.Load()
		},
	})
	s.T().Cleanup(anomalies.Close)
	anomalies.Start()
	s.checked(0)

	s.Equal([]string{"2025-02-04", "2026-03-09"}, s.days("customer_activity"))
	s.Empty(s.days("activity_digests"), "the old day's digest went, none was posted for the day before")
	s.Empty(s.days("cost_anomaly_notices"), "a spike today, and nothing noted that nobody would be told of")
	s.Empty(s.slack.posts())
}

// Defect: the keep counted from the clock's own date rather than the UTC day: at 01:00 on 11
// Mar in UTC+14 it is still 10 Mar in UTC, so 3 Feb 2025 is 400 days back and stays.
func (s *DigestSuite) TestTheKeepCountsFromTheUTCDayWhateverTheClocksZone() {
	pinned := organization()
	s.active(pinned, activity.AreaAgentSetup, 401, 1, 0)
	s.active(pinned, activity.AreaAgentSetup, 400, 1, 0)

	pruned, err := s.db.PruneActivity(s.ctx, time.Date(2026, 3, 11, 1, 0, 0, 0, time.FixedZone("UTC+14", 14*60*60)))
	s.Require().NoError(err)

	s.Equal(1, pruned)
	s.Equal([]string{"2025-02-03"}, s.days("customer_activity"))
}

// Only the router that holds the cost anomaly lock deletes: another router's check leaves the old
// rows, and the holder's next check deletes them.
func (s *DigestSuite) TestOnlyTheRouterHoldingTheLockDeletesTheActivity() {
	pinned := organization()
	s.active(pinned, activity.AreaAgentSetup, 401, 1, 0)
	s.active(pinned, activity.AreaAgentSetup, 0, 1, 0)
	elsewhere := s.db.CostAnomalyLock()
	held, err := elsewhere.Hold(s.ctx)
	s.Require().NoError(err)
	s.Require().True(held)
	s.T().Cleanup(elsewhere.Release)

	s.router().Start()
	s.checked(0)
	s.Equal([]string{"2025-02-02", "2026-03-10"}, s.days("customer_activity"), "another router holds the lock")

	elsewhere.Release()
	s.Require().Eventually(func() bool {
		return slices.Equal(s.days("customer_activity"), []string{"2026-03-10"})
	}, settleFor, time.Millisecond)
}

// A delete that failed is tried again at the next check the same day, not the next day. The
// suite's database is its own, so the delete is made to fail by renaming one of its tables.
func (s *DigestSuite) TestADeleteThatFailedIsTriedAgainTheSameDay() {
	pinned := organization()
	s.active(pinned, activity.AreaAgentSetup, 401, 1, 0)
	s.active(pinned, activity.AreaAgentSetup, 0, 1, 0)
	_, err := s.db.DB().ExecContext(s.ctx, "ALTER TABLE activity_digests RENAME TO activity_digests_renamed")
	s.Require().NoError(err)
	renamed := true
	s.T().Cleanup(func() {
		if renamed {
			_, _ = s.db.DB().ExecContext(s.ctx, "ALTER TABLE activity_digests_renamed RENAME TO activity_digests")
		}
	})
	anomalies := NewAnomalies(AnomalyOptions{
		Store: s.db, Logger: slog.New(slog.DiscardHandler), Every: 20 * time.Millisecond,
		Now: func() time.Time {
			s.reads.Add(1)
			return *s.clock.Load()
		},
	})
	s.T().Cleanup(anomalies.Close)
	anomalies.Start()
	s.checked(0)
	_, err = s.db.DB().ExecContext(s.ctx, "ALTER TABLE activity_digests_renamed RENAME TO activity_digests")
	s.Require().NoError(err)
	renamed = false
	_, err = s.db.DB().ExecContext(s.ctx, "INSERT INTO activity_digests (day) VALUES ('2025-02-01')")
	s.Require().NoError(err)
	s.Require().Eventually(func() bool {
		return len(s.days("activity_digests")) == 0
	}, settleFor, time.Millisecond, "the failed delete is tried again at the next check")
}
