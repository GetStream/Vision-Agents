//go:build integration

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	usageservice "github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

// person is the end user id the tests forget: an email, as a user id often is.
const person = "ada@example.com"

// ForgetUserSuite runs `router usage forget-user` against a shared database and two shards of
// its own, the real usage service on a database of its own mirroring to Redis, and the routers'
// Redis.
type ForgetUserSuite struct {
	suite.Suite
	ctx      context.Context
	settings config.Config
	// shards are a router's store on each shard, here first.
	shards   []*store.Store
	usageDSN string
	service  *usagetest.Service
	// routers is the routers' Redis, which keeps holds and quota buckets; counters is the one
	// the usage service mirrors spend to.
	routers, counters rueidis.Client
	// app and organization are this test's own, so their keys are too.
	app, organization string
	keys              []string
}

func TestForgetUserSuite(t *testing.T) {
	suite.Run(t, new(ForgetUserSuite))
}

func (s *ForgetUserSuite) SetupSuite() {
	s.ctx = context.Background()
	address := os.Getenv("ROUTER_REDIS_ADDR")
	if address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR not set")
	}
	s.usageDSN = usagetest.Database(s.T(), "usage_forget")
	s.counters = usagetest.Redis(s.T())
	routers, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{address}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(routers.Close)
	s.routers = routers
}

func (s *ForgetUserSuite) SetupTest() {
	shared := s.emptyDatabase("forget")
	here, there := s.emptyDatabase("forget_here"), s.emptyDatabase("forget_there")
	shards := map[string]string{"here": here, "there": there}
	s.shards = nil
	for _, shard := range []string{here, there} {
		router, err := store.OpenShards(store.Options{DSN: shared, ShardDSN: shard, Shards: shards})
		s.Require().NoError(err)
		s.Require().NoError(router.Migrate(s.ctx))
		s.T().Cleanup(func() { router.Close() })
		s.shards = append(s.shards, router)
	}
	s.service = usagetest.Start(s.T(), usagetest.Options{
		DSN: s.usageDSN, RedisURL: usagetest.RedisURL(s.T()), ManageKey: "manage-" + uuid.NewString(),
	})
	usagetest.Empty(s.T(), s.service.Ledger)
	s.app, s.organization = "app-"+uuid.NewString(), usagetest.Organization()
	s.keys = nil

	s.settings = config.Defaults()
	s.settings.Postgres = config.Postgres{DSN: shared, ShardDSN: here, Shards: shards}
	s.settings.Redis.Addr = os.Getenv("ROUTER_REDIS_ADDR")
	s.settings.Usage = config.Usage{
		URL: s.service.URL, Token: s.service.Token, ManageKey: s.service.ManageKey, RedisURL: usagetest.RedisURL(s.T()),
	}
}

// TearDownTest deletes, by name, every key a test made or the service mirrored for it.
func (s *ForgetUserSuite) TearDownTest() {
	for _, apps := range [][2]string{{s.app, person}, {s.app, "grace"}, {"other-" + s.app, person}} {
		ref := usageservice.SubjectRef{OrganizationID: s.organization, Subject: usageservice.SubjectUser, AppID: apps[0], UserID: apps[1]}
		s.keys = append(s.keys, ref.Key(usageservice.CounterPrefix))
	}
	for _, scope := range []usageservice.SubjectRef{
		{OrganizationID: s.organization, Subject: usageservice.SubjectOrganization},
		{OrganizationID: s.organization, Subject: usageservice.SubjectApp, AppID: s.app},
		{OrganizationID: s.organization, Subject: usageservice.SubjectApp, AppID: "other-" + s.app},
		{OrganizationID: s.organization, Subject: usageservice.SubjectAPIKey, AppID: s.app, APIKey: "key"},
		{OrganizationID: s.organization, Subject: usageservice.SubjectAPIKey, AppID: "other-" + s.app, APIKey: "key"},
	} {
		s.keys = append(s.keys, scope.Key(usageservice.CounterPrefix))
	}
	for _, client := range []rueidis.Client{s.routers, s.counters} {
		for _, key := range s.keys {
			s.Require().NoError(client.Do(s.ctx, client.B().Del().Key(key).Build()).Error())
		}
	}
}

// emptyDatabase is a database of this suite's own with an empty schema.
func (s *ForgetUserSuite) emptyDatabase(name string) string {
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

// recorded writes one billable row of app's user on a shard and ships its event, as the router
// would, and answers its usage id.
func (s *ForgetUserSuite) recorded(shard int, app, user string, cost int64) uuid.UUID {
	id := uuid.Must(uuid.NewV7())
	_, err := s.shards[shard].RecordRequests(s.ctx, []store.Request{{
		UsageID: id, Modality: "llm", CustomerID: app, OrganizationID: s.organization, OrganizationTier: usageservice.DefaultTier,
		APIKey: "key", UserID: user, Provider: "openai", Model: "gpt", StartedAt: time.Now().UTC(),
		CostMicros: cost, BaseCostMicros: cost, Billable: true, Success: true,
	}})
	s.Require().NoError(err)
	s.service.Spend(s.T(), s.organization, cost, func(e *usageservice.Event) { e.UsageID, e.AppID, e.UserID = id, app, user })
	return id
}

// forget runs the command with --settle 0 and the args, and answers what it printed.
func (s *ForgetUserSuite) forget(app, user string, args ...string) string {
	var out bytes.Buffer
	command := append([]string{"forget-user", "--app", app, "--user", user, "--actor", "staff@example.com", "--settle", "0s"}, args...)
	s.Require().NoError(runUsage(command, s.settings, &out))
	return out.String()
}

// requests is every shard's rows, user by usage id.
func (s *ForgetUserSuite) requests() map[uuid.UUID]store.Request {
	rows := map[uuid.UUID]store.Request{}
	for _, shard := range s.shards {
		var found []store.Request
		s.Require().NoError(shard.ShardDB().NewSelect().Model(&found).Scan(s.ctx))
		for _, row := range found {
			rows[row.UsageID] = row
		}
	}
	return rows
}

// events is the usage service's events, by usage id.
func (s *ForgetUserSuite) events() map[uuid.UUID]ledger.UsageEvent {
	var found []ledger.UsageEvent
	s.Require().NoError(s.service.Ledger.DB().NewSelect().Model(&found).Scan(s.ctx))
	events := map[uuid.UUID]ledger.UsageEvent{}
	for _, event := range found {
		events[event.UsageID] = event
	}
	return events
}

// windows is how many user windows of app's user the usage service keeps.
func (s *ForgetUserSuite) windows(app, user string) int {
	count, err := s.service.Ledger.DB().NewSelect().Model((*ledger.UsageWindow)(nil)).
		Where("subject = ?", usageservice.SubjectUser).Where("app_id = ?", app).Where("user_id = ?", user).Count(s.ctx)
	s.Require().NoError(err)
	return count
}

// spendOf is the count and cost of app's rows on every shard, and of its events.
func (s *ForgetUserSuite) spendOf(app string) [4]int64 {
	var spend [4]int64
	for _, row := range s.requests() {
		if row.CustomerID == app {
			spend[0]++
			spend[1] += row.CostMicros
		}
	}
	for _, event := range s.events() {
		if event.AppID == app {
			spend[2]++
			spend[3] += event.CostMicros
		}
	}
	return spend
}

// userKeys are app's user's holds and every day's quota buckets in the routers' Redis, written
// here, and their counter in the usage service's, which shipping their rows mirrored.
func (s *ForgetUserSuite) userKeys(app, user string) (routers, counters []string) {
	ref := usageservice.SubjectRef{OrganizationID: s.organization, Subject: usageservice.SubjectUser, AppID: app, UserID: user}
	routers = append([]string{policy.HoldKey(ref)}, quota.UserKeys(app, user, time.Now())...)
	for _, key := range routers {
		s.Require().NoError(s.routers.Do(s.ctx, s.routers.B().Hset().Key(key).FieldValue().FieldValue("messages", "3").Build()).Error())
	}
	s.keys = append(s.keys, routers...)
	return routers, []string{ref.Key(usageservice.CounterPrefix)}
}

// present is how many of keys client has.
func (s *ForgetUserSuite) present(client rueidis.Client, keys []string) int64 {
	var found int64
	for _, key := range keys {
		exists, err := client.Do(s.ctx, client.B().Exists().Key(key).Build()).AsInt64()
		s.Require().NoError(err)
		found += exists
	}
	return found
}

// settingsOf writes app's user's own policy, a limit of theirs and one of the app's for every
// user, a reset, a grant and a campaign they scheduled.
func (s *ForgetUserSuite) settingsOf(app, user string) {
	shared := s.shards[0]
	audit := store.LimitAudit{CustomerID: app, Source: store.AuditSourceAPI, ActorID: "staff"}
	ten := int64(10)
	s.Require().NoError(shared.SavePolicy(s.ctx, store.ScopeUser, store.UserScopeID(app, user),
		store.PolicyDocument{Quota: &store.Quota{MessagesPerUserPerDay: &ten}}))
	for _, limit := range []store.Limit{
		{Subject: usageservice.SubjectUser, Level: store.LevelUser, UserID: user},
		{Subject: usageservice.SubjectUser, Level: store.LevelApp},
	} {
		limit.OrganizationID, limit.AppID, limit.Interval, limit.LimitMicros, limit.UpdatedBy = s.organization, app, usageservice.Daily, 1_000, "staff"
		s.Require().NoError(shared.CreateLimit(s.ctx, &limit, audit))
	}
	start, _ := usageservice.Window(usageservice.Daily, time.Now())
	s.Require().NoError(shared.AddResets(s.ctx, []store.UsageReset{{
		OrganizationID: s.organization, Subject: usageservice.SubjectUser, AppID: app, UserID: user,
		Interval: usageservice.Daily, WindowStart: start, BaselineMicros: 5, Reason: "support", Actor: "staff",
	}}, audit))
	s.Require().NoError(shared.AddGrant(s.ctx, &store.LimitGrant{
		OrganizationID: s.organization, Subject: usageservice.SubjectUser, AppID: app, UserID: user,
		AmountMicros: 500, Reason: "support", Actor: "staff",
	}, audit))
	s.Require().NoError(shared.CreateCampaign(s.ctx, &store.Campaign{CustomerID: app, Name: "renewals", APIKey: "key", UserID: user}))
}

// userRows is what the shared database holds of app's user: their own policy, the limits naming
// them and the app's for every user, and the user ids on their resets, grants and campaigns.
type userRows struct {
	policy       bool
	ownLimits    int
	appLimits    int
	resetUsers   []string
	grantUsers   []string
	campaignUser []string
}

func (s *ForgetUserSuite) userRows(app, user string) userRows {
	shared := s.shards[0].DB()
	var rows userRows
	policy, err := s.shards[0].Policy(s.ctx, store.ScopeUser, store.UserScopeID(app, user))
	s.Require().NoError(err)
	rows.policy = policy.Quota != nil
	rows.ownLimits, err = shared.NewSelect().Model((*store.Limit)(nil)).Where("app_id = ?", app).Where("level = ?", store.LevelUser).Count(s.ctx)
	s.Require().NoError(err)
	rows.appLimits, err = shared.NewSelect().Model((*store.Limit)(nil)).Where("app_id = ?", app).Where("level = ?", store.LevelApp).Count(s.ctx)
	s.Require().NoError(err)
	s.Require().NoError(shared.NewSelect().Model((*store.UsageReset)(nil)).Column("user_id").Where("app_id = ?", app).Scan(s.ctx, &rows.resetUsers))
	s.Require().NoError(shared.NewSelect().Model((*store.LimitGrant)(nil)).Column("user_id").Where("app_id = ?", app).Scan(s.ctx, &rows.grantUsers))
	s.Require().NoError(shared.NewSelect().Model((*store.Campaign)(nil)).Column("user_id").Where("customer_id = ?", app).Scan(s.ctx, &rows.campaignUser))
	return rows
}

// forgottenAs is the one replacement every row of the user now carries.
func (s *ForgetUserSuite) forgottenAs(out string) string {
	_, rest, found := strings.Cut(out, "replacing the user's id with ")
	s.Require().True(found, out)
	replacement, _, _ := strings.Cut(rest, "\n")
	s.Require().True(usageservice.IsReplacement(replacement), replacement)
	return replacement
}

func (s *ForgetUserSuite) TestForgettingAUserRewritesEveryShardAndTheUsageService() {
	ids := []uuid.UUID{s.recorded(0, s.app, person, 300), s.recorded(0, s.app, person, 200), s.recorded(1, s.app, person, 500)}
	grace := s.recorded(1, s.app, "grace", 70)
	before := s.spendOf(s.app)

	out := s.forget(s.app, person)

	replacement := s.forgottenAs(out)
	s.Contains(out, "requests: 3 rows rewritten")
	s.Contains(out, "usage service: 3 events, 4 windows rewritten")
	requests, events := s.requests(), s.events()
	for _, id := range ids {
		s.Equal(replacement, requests[id].UserID, "on both shards")
		s.Equal(replacement, events[id].UserID)
	}
	s.Equal("grace", requests[grace].UserID)
	s.Equal("grace", events[grace].UserID)
	s.Zero(s.windows(s.app, person))
	s.Equal(len(usageservice.Intervals), s.windows(s.app, replacement))
	s.Equal(before, s.spendOf(s.app), "count and cost per app are unchanged")
	s.Equal([4]int64{4, 1070, 4, 1070}, before)
}

func (s *ForgetUserSuite) TestForgettingAUserLeavesOtherAppsAlone() {
	other := "other-" + s.app
	s.recorded(0, s.app, person, 100)
	theirs := s.recorded(1, other, person, 40)
	routers, counters := s.userKeys(other, person)
	s.settingsOf(other, person)
	before := s.userRows(other, person)

	s.forget(s.app, person)

	s.Equal(person, s.requests()[theirs].UserID)
	s.Equal(person, s.events()[theirs].UserID)
	s.Equal(len(usageservice.Intervals), s.windows(other, person))
	s.Equal(int64(len(routers)), s.present(s.routers, routers))
	s.Equal(int64(1), s.present(s.counters, counters))
	s.Equal(before, s.userRows(other, person))
	s.Equal(userRows{policy: true, ownLimits: 1, appLimits: 1, resetUsers: []string{person}, grantUsers: []string{person},
		campaignUser: []string{person}}, before)
}

func (s *ForgetUserSuite) TestForgettingAUserClearsTheirCountersPolicyAndLimits() {
	s.recorded(0, s.app, person, 100)
	routers, counters := s.userKeys(s.app, person)
	s.settingsOf(s.app, person)
	s.Require().Equal(int64(1), s.present(s.counters, counters), "shipping their row mirrored their counter")

	out := s.forget(s.app, person)

	replacement := s.forgottenAs(out)
	s.Zero(s.present(s.routers, routers), "their holds and quota buckets")
	s.Zero(s.present(s.counters, counters), "their spend counter")
	s.Contains(out, "redis: 8 keys deleted")
	s.Equal(userRows{appLimits: 1, resetUsers: []string{replacement}, grantUsers: []string{replacement},
		campaignUser: []string{replacement}}, s.userRows(s.app, person), "the app's limit for every user is what applies now")
	s.Contains(out, "shared database: 1 policies and 1 limits deleted; 1 resets, 1 grants and 1 campaigns rewritten")
}

func (s *ForgetUserSuite) TestTheAuditRowNeverNamesTheUser() {
	s.recorded(0, s.app, person, 100)
	s.settingsOf(s.app, person)

	out := s.forget(s.app, person)

	replacement := s.forgottenAs(out)
	s.NotContains(out, person)
	entries, err := s.shards[0].AuditEntries(s.ctx, s.app, store.AuditEntryFilter{ResourceType: store.AuditEndUser})
	s.Require().NoError(err)
	s.Require().Len(entries, 1)
	entry := entries[0]
	s.Equal([]string{replacement, store.AuditSourceCLI, "staff@example.com"}, []string{entry.ResourceID, entry.Source, entry.ActorID})
	s.Contains(out, "audit: "+entry.ID)
	recorded, err := json.Marshal(entry)
	s.Require().NoError(err)
	s.NotContains(string(recorded), person)
	counts := map[string]float64{}
	for _, change := range entry.Changes {
		counts[change.Field] = change.After.(float64)
	}
	s.Equal(map[string]float64{
		"requests": 1, "events": 1, "windows": 4, "redis_keys": 1,
		"policies_deleted": 1, "limits_deleted": 1, "resets": 1, "grants": 1, "campaigns": 1,
	}, counts)
	var audits []ledger.ManageAudit
	s.Require().NoError(s.service.Ledger.DB().NewSelect().Model(&audits).Where("action = ?", ledger.ActionForgetUser).Scan(s.ctx))
	s.Require().Len(audits, 1, "one organization rewritten in the usage service, audited there too")
	s.NotContains(string(audits[0].After), person)
}

func (s *ForgetUserSuite) TestADryRunChangesNothing() {
	id := s.recorded(0, s.app, person, 100)
	s.recorded(1, s.app, person, 100)
	routers, counters := s.userKeys(s.app, person)
	s.settingsOf(s.app, person)
	before := s.userRows(s.app, person)

	out := s.forget(s.app, person, "--dry-run")

	s.Equal("dry run, nothing is changed\nrequests: 2 rows\nusage service: 2 events, 4 windows\nredis: 8 keys\n"+
		"shared database: 1 policies and 1 limits to delete; 1 resets, 1 grants and 1 campaigns to rewrite\n", out)
	s.Equal(person, s.requests()[id].UserID)
	s.Equal(person, s.events()[id].UserID)
	s.Equal(len(usageservice.Intervals), s.windows(s.app, person))
	s.Equal(int64(len(routers)), s.present(s.routers, routers))
	s.Equal(int64(1), s.present(s.counters, counters))
	s.Equal(before, s.userRows(s.app, person))
	entries, err := s.shards[0].AuditEntries(s.ctx, s.app, store.AuditEntryFilter{ResourceType: store.AuditEndUser})
	s.Require().NoError(err)
	s.Empty(entries)
}

func (s *ForgetUserSuite) TestForgettingAUserNeedsTheManageKey() {
	s.settings.Usage.ManageKey = ""

	err := runUsage([]string{"forget-user", "--app", s.app, "--user", person, "--actor", "staff@example.com"}, s.settings, &bytes.Buffer{})

	s.ErrorContains(err, "ROUTER_USAGE_MANAGE_KEY")
}
