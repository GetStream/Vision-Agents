package main

import (
	"bufio"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/url"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	usageservice "github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

const usageCommands = `usage: router usage <command>

  replay --from path         write the usage rows a log of this shard's routers says were lost
                             (msg="usage row lost"), once each: replaying a file again writes
                             nothing twice. With a usage service (ROUTER_USAGE_URL) the sweeper
                             ships them. Writes the database
  reship --since time [--rejected]
                             queue this shard's rows started since then (RFC 3339, or a duration
                             back from now such as 24h) that the usage service has to be sent to
                             it again, which records a usage id once; --rejected queues the rows
                             it refused instead, once the producer is fixed. A row it never had
                             nor refused is never queued: one written before the router shipped,
                             or carried in by a data move. Writes the database
  forget-user --app id --user id --actor who [--dry-run] [--settle 2m]
                             replace an app's end user id in the usage records with forgotten:
                             and 32 random hex characters, in order: the requests of every shard;
                             the usage service's events and windows, and again after --settle
                             for a batch shipped in between; the user's spend counters, holds
                             and quota buckets in Redis, by name; and in the shared database
                             their own policy and user-level limits, deleted, so the app's
                             defaults apply, and the id on their resets, grants and campaigns.
                             Usage and cost stay where they were, under the app. Prints each
                             step's count and writes one audit row (end_user, source cli) that
                             names the replacement and --actor, never the id. --dry-run prints
                             the counts and changes nothing. It covers usage records only:
                             sessions, conversation history and agent_logs have their own
                             retention. Needs ROUTER_USAGE_URL and ROUTER_USAGE_MANAGE_KEY.
                             Writes the database
  storage [--hour time]      record an hour of every knowledge base's size: the last hour to have
                             ended, or the RFC 3339 hour given. One storage row per knowledge base
                             of an app this deployment knows, dated to the hour and owned by the
                             app under job:knowledge_storage; an empty one writes none. The size
                             is the one the base holds now, whatever the hour: turbopuffer keeps
                             no past sizes. A rerun for an hour writes nothing twice. One it cannot
                             read is logged, alerted on as storage_missed naming the hour, and
                             fails the run, so a retry within the next hour, or --hour, fills it
                             in. Run it hourly, once per turbopuffer key and region: it reads every
                             namespace the key holds, whichever shard or router wrote it, so one
                             run per shard would read each base once per shard (writing it once).
                             Needs TURBOPUFFER_API_KEY. Writes the database
  numbers --from time        bill each month of a held number after the month it was bought in,
                             on the day of the month it was bought, at the price its vendor quoted,
                             owned by its app under job:number_rental: the months that began since
                             --from (RFC 3339, required) and in the last 35 days and have no row.
                             A month that began before --from is never billed. A rerun writes
                             nothing twice. A number with no price is billed as work with no price
                             and alerted on as number_unpriced; a run that cannot read the numbers
                             is alerted on as number_rental_missed and fails. Run it hourly, once
                             per deployment. Writes the database
  dlc --from time            bill each calendar month of every 10DLC campaign a vendor holds,
                             after the three charged when it was registered, owned by its app
                             under job:dlc_fees: this month and the last, if they began since
                             --from (RFC 3339, required) and have no row. A run that cannot read
                             the campaigns is alerted on as dlc_fees_missed and fails. Run it
                             daily, once per deployment. Writes the database
  sandboxes [--reconcile [--day YYYY-MM-DD]]
                             delete this shard's sandboxes (labelled stream.env=SENTRY_ENVIRONMENT
                             and stream.router=its shard) whose session has closed, or that have no
                             session, once 10 minutes old. Run it hourly, once per shard. With
                             --reconcile, first check a day of each sandbox's billed vCPU, memory
                             and disk seconds against Daytona's records (the day before yesterday:
                             they lag up to 48 hours): what was billed short is recorded from
                             Daytona's figures, owned by the sandbox's labels' app under
                             job:sandbox_reconcile; more than 5% (or 60 units) apart is logged and
                             alerted on as sandbox_mismatch naming the day. Run that daily. A
                             rerun records nothing twice. Needs DAYTONA_API_KEY, and
                             DAYTONA_ORGANIZATION_ID unless the key can list a sandbox. Writes the
                             database
`

const (
	// usageAlertEvery is how often the usage conditions are checked, and usageAlertLimit the
	// least time between two alerts of one.
	usageAlertEvery = 15 * time.Second
	usageAlertLimit = time.Minute
	// replayBatch is how many lost rows replay writes in one transaction, as the writer does.
	replayBatch = 200
	// maxLogLine is the longest log line replay reads: a lost row is logged whole.
	maxLogLine = 4 << 20
	// lostRowMessage is the message the writer logs a lost row under.
	lostRowMessage = `msg="usage row lost"`
)

// newUsageClient is the usage service the router ships its rows to, or nil when it ships them
// nowhere.
func newUsageClient(settings config.Config, pgStore *store.Store, logger *slog.Logger) (*usageservice.Client, error) {
	if settings.Usage.URL == "" {
		if settings.Usage.Token != "" {
			// What stopping shipping by unsetting the URL alone leaves behind.
			logger.Warn("usage is not shipped: a usage token is set without the usage service's URL",
				"setting", "usage.url", "token_setting", "usage.token")
			return nil, nil
		}
		logger.Info("usage is not shipped: no usage service", "setting", "usage.url")
		return nil, nil
	}
	if pgStore == nil {
		logger.Warn("usage is not shipped: there is no database to write it to first", "setting", "postgres.dsn")
		return nil, nil
	}
	client, err := usageservice.NewClient(usageservice.ClientOptions{URL: settings.Usage.URL, Token: settings.Usage.Token})
	if err != nil {
		return nil, err
	}
	shown := settings.Usage.URL
	if parsed, err := url.Parse(shown); err == nil {
		shown = parsed.Redacted()
	}
	logger.Info("shipping usage to the usage service", "url", shown, "setting", "usage.url")
	return client, nil
}

// startUsageAlerts runs the process's usage alerts, tagged with its shard, until the returned
// function stops them.
func startUsageAlerts(ctx context.Context, source, shard string, logger *slog.Logger) (stop func()) {
	return startAlerts(ctx, alerts.New(alerts.Options{
		Logger: logger, Source: source, Tags: map[string]string{"shard": shard}, Every: usageAlertEvery, Limit: usageAlertLimit,
	}, routing.Alerts(usageAlertLimit)...))
}

// startUsageSweeper runs the shard's sweeper until the returned function stops it and it has
// given its lock back.
func startUsageSweeper(ctx context.Context, options routing.SweeperOptions) (stop func()) {
	background, cancel := context.WithCancel(context.WithoutCancel(ctx))
	sweeper := routing.NewSweeper(options)
	done := make(chan struct{})
	go func() {
		defer close(done)
		sweeper.Run(background)
	}()
	return func() {
		cancel()
		<-done
	}
}

func runUsage(args []string, settings config.Config, out io.Writer) error {
	if len(args) == 0 {
		return errors.New(usageCommands)
	}
	ctx := context.Background()
	switch args[0] {
	case "replay":
		return runUsageReplay(ctx, args[1:], settings, out)
	case "reship":
		return runUsageReship(ctx, args[1:], settings, out)
	case "forget-user":
		return runUsageForgetUser(ctx, args[1:], settings, out)
	case "storage":
		return runUsageStorage(ctx, args[1:], settings, out)
	case "numbers":
		return runUsageNumbers(ctx, args[1:], settings, out)
	case "dlc":
		return runUsageDLC(ctx, args[1:], settings, out)
	case "sandboxes":
		return runUsageSandboxes(ctx, args[1:], settings, out)
	default:
		return fmt.Errorf("unknown usage command %q\n\n%s", args[0], usageCommands)
	}
}

// runUsageReplay writes the rows a log says were lost. Each keeps its usage id, so a row
// written since, or replayed before, is not written twice.
func runUsageReplay(ctx context.Context, args []string, settings config.Config, out io.Writer) error {
	flags := flag.NewFlagSet("usage replay", flag.ContinueOnError)
	from := flags.String("from", "", "the log file to read the lost rows from")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *from == "" {
		return errors.New("usage replay needs --from path, a log of the routers that lost the rows")
	}
	file, err := os.Open(*from)
	if err != nil {
		return fmt.Errorf("usage replay: %w", err)
	}
	defer file.Close() //nolint:errcheck // read-only
	rows, unreadable, err := lostRows(file, settings.Usage.URL != "")
	if err != nil {
		return err
	}
	pgStore, err := openPriceStore(settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()

	var written, refused int
	for start := 0; start < len(rows); start += replayBatch {
		batch := rows[start:min(start+replayBatch, len(rows))]
		wrote, failed, err := replayRows(ctx, pgStore, batch, out)
		if err != nil {
			return err
		}
		written, refused = written+wrote, refused+failed
	}
	_, err = fmt.Fprintf(out, "%d lost rows: %d written, %d already there, %d refused; %d lost-row lines could not be read\n",
		len(rows), written, len(rows)-written-refused, refused, unreadable)
	return err
}

// replayRows writes one batch of lost rows, and one at a time when the database refuses the
// batch for a row in it, naming each row it still refuses.
func replayRows(ctx context.Context, pgStore *store.Store, rows []store.Request, out io.Writer) (written, refused int, err error) {
	ids, err := pgStore.RecordRequests(ctx, rows)
	if err == nil || errors.Is(err, store.ErrRequestLogsMissing) {
		return len(ids), 0, nil
	}
	if !store.Unwritable(err) {
		return 0, 0, err
	}
	if len(rows) == 1 {
		_, err := fmt.Fprintf(out, "refused %s: %v\n", rows[0].UsageID, err)
		return 0, 1, err
	}
	for i := range rows {
		wrote, failed, err := replayRows(ctx, pgStore, rows[i:i+1], out)
		if err != nil {
			return written, refused, err
		}
		written, refused = written+wrote, refused+failed
	}
	return written, refused, nil
}

// lostRows reads the rows of every lost-row line of a text log, as the writer would have
// written them: queued to ship when there is a usage service, with its defaults for a key and
// a tier the row was lost before it got, and without a user id the usage service would refuse
// the event for, which a router from before the rule may have logged. It counts the lost-row lines it could not read: a row
// that could not be encoded, or one with no usage id, which could not be written once.
func lostRows(log io.Reader, toShip bool) ([]store.Request, int, error) {
	var rows []store.Request
	unreadable := 0
	scanner := bufio.NewScanner(log)
	scanner.Buffer(make([]byte, 0, 64<<10), maxLogLine)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.Contains(line, lostRowMessage) {
			continue
		}
		row, ok := lostRow(line)
		if !ok {
			unreadable++
			continue
		}
		row.ID, row.ToShip, row.ShippedAt, row.ShipError = 0, toShip, nil, ""
		row.APIKey = cmp.Or(row.APIKey, routing.Unattributed)
		row.OrganizationTier = cmp.Or(row.OrganizationTier, usageservice.DefaultTier)
		row.UserID = routing.RecordedUser(row.UserID)
		rows = append(rows, row)
	}
	if err := scanner.Err(); err != nil {
		return nil, 0, fmt.Errorf("usage replay: read the log: %w", err)
	}
	return rows, unreadable, nil
}

// lostRow is the row a lost-row line carries in its row field, quoted as slog's text handler
// quotes it.
func lostRow(line string) (store.Request, bool) {
	_, value, found := strings.Cut(line, " row=")
	if !found {
		return store.Request{}, false
	}
	quoted, err := strconv.QuotedPrefix(value)
	if err != nil {
		return store.Request{}, false
	}
	raw, err := strconv.Unquote(quoted)
	if err != nil {
		return store.Request{}, false
	}
	var row store.Request
	if err := json.Unmarshal([]byte(raw), &row); err != nil || row.UsageID == uuid.Nil {
		return store.Request{}, false
	}
	return row, true
}

// runUsageReship queues rows to be sent to the usage service again.
func runUsageReship(ctx context.Context, args []string, settings config.Config, out io.Writer) error {
	flags := flag.NewFlagSet("usage reship", flag.ContinueOnError)
	sinceText := flags.String("since", "", "the earliest start of a row to send again: RFC 3339, or a duration back from now")
	rejected := flags.Bool("rejected", false, "the rows the usage service refused, rather than the ones it has")
	if err := flags.Parse(args); err != nil {
		return err
	}
	since, err := sinceTime(*sinceText, time.Now())
	if err != nil {
		return err
	}
	if settings.Usage.URL == "" {
		return errors.New("usage reship: no usage service is set (usage.url, ROUTER_USAGE_URL), so nothing would ship the rows")
	}
	pgStore, err := openPriceStore(settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()
	queued, err := pgStore.Reship(ctx, since, *rejected)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(out, "queued %d rows started since %s to ship again\n", queued, since.UTC().Format(time.RFC3339))
	return err
}

// sinceTime reads --since: an RFC 3339 time, or a duration back from now.
func sinceTime(text string, now time.Time) (time.Time, error) {
	if text == "" {
		return time.Time{}, errors.New("usage reship needs --since: an RFC 3339 time, or a duration back from now such as 24h")
	}
	if at, err := time.Parse(time.RFC3339, text); err == nil {
		return at, nil
	}
	back, err := time.ParseDuration(text)
	if err != nil || back <= 0 {
		return time.Time{}, fmt.Errorf("usage reship: --since %q is neither an RFC 3339 time nor a positive duration", text)
	}
	return now.Add(-back), nil
}

// forgetSettle is how long forget-user waits to rewrite the usage service's events again: a
// batch a shipper read before the shards were rewritten can land after the first rewrite,
// still carrying the id.
const forgetSettle = 2 * time.Minute

// forgetting is one run of forget-user: an app's end user and the id that replaces theirs.
type forgetting struct {
	store       *store.Store
	usage       *usageservice.Client
	redis       []rueidis.Client
	app, user   string
	actor       string
	replacement string
	out         io.Writer
}

// runUsageForgetUser replaces an end user's id in the usage records, step by step, or counts
// what it would replace.
func runUsageForgetUser(ctx context.Context, args []string, settings config.Config, out io.Writer) error {
	flags := flag.NewFlagSet("usage forget-user", flag.ContinueOnError)
	app := flags.String("app", "", "the app the user is an end user of")
	user := flags.String("user", "", "the end user id to forget, as it was recorded")
	actor := flags.String("actor", "", "who asked for it, recorded in the audit row: a staff member's email")
	dryRun := flags.Bool("dry-run", false, "print the counts and change nothing")
	settle := flags.Duration("settle", forgetSettle, "how long to wait before rewriting the usage service's events again")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *app == "" || *user == "" || *actor == "" {
		return errors.New("usage forget-user needs --app, --user and --actor")
	}
	if settings.Usage.URL == "" || settings.Usage.ManageKey == "" {
		return errors.New("usage forget-user needs the usage service (usage.url, ROUTER_USAGE_URL) and its manage key " +
			"(usage.manage_key, ROUTER_USAGE_MANAGE_KEY)")
	}
	client, err := usageservice.NewClient(usageservice.ClientOptions{URL: settings.Usage.URL, ManageKey: settings.Usage.ManageKey})
	if err != nil {
		return err
	}
	pgStore, err := openPriceStore(settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()
	redis, err := forgetRedis(settings)
	if err != nil {
		return err
	}
	defer func() {
		for _, client := range redis {
			client.Close()
		}
	}()
	run := forgetting{
		store: pgStore, usage: client, redis: redis, app: *app, user: *user, actor: *actor,
		replacement: usageservice.NewReplacement(), out: out,
	}
	if *dryRun {
		return run.count(ctx)
	}
	return run.forget(ctx, *settle)
}

// forgetRedis connects to every Redis holding a user's keys: the routers' own, which keeps holds
// and quota buckets, and the usage service's, which keeps spend counters, when that is another.
func forgetRedis(settings config.Config) ([]rueidis.Client, error) {
	var all []rueidis.ClientOption
	if settings.Redis.Addr != "" {
		all = append(all, rueidis.ClientOption{
			InitAddress: []string{settings.Redis.Addr}, Username: settings.Redis.Username, Password: settings.Redis.Password,
			DisableCache: true,
		})
	}
	if settings.Usage.RedisURL != "" {
		options, err := usageRedisOptions(settings.Usage.RedisURL)
		if err != nil {
			return nil, err
		}
		if len(all) == 0 || !slices.Equal(all[0].InitAddress, options.InitAddress) || all[0].SelectDB != options.SelectDB {
			all = append(all, options)
		}
	}
	clients := make([]rueidis.Client, 0, len(all))
	for _, options := range all {
		client, err := rueidis.NewClient(options)
		if err != nil {
			for _, opened := range clients {
				opened.Close()
			}
			return nil, fmt.Errorf("usage forget-user: redis: %w", err)
		}
		clients = append(clients, client)
	}
	return clients, nil
}

// count prints what forget would change, and changes nothing.
func (f forgetting) count(ctx context.Context) error {
	organizations, err := f.store.UserOrganizations(ctx, f.app, f.user)
	if err != nil {
		return err
	}
	requests, err := f.store.UserRequests(ctx, f.app, f.user)
	if err != nil {
		return err
	}
	service, err := f.usage.ForgetUser(ctx, f.actor, usageservice.ForgetUser{AppID: f.app, UserID: f.user, Replacement: f.replacement, DryRun: true})
	if err != nil {
		return err
	}
	keys := 0
	for _, client := range f.redis {
		for _, response := range client.DoMulti(ctx, f.commands(client, true, organizations, service.Organizations)...) {
			found, err := response.AsInt64()
			if err != nil {
				return fmt.Errorf("usage forget-user: redis: %w", err)
			}
			keys += int(found)
		}
	}
	settings, err := f.store.CountUserSettings(ctx, f.app, f.user)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(f.out, "dry run, nothing is changed\nrequests: %d rows\nusage service: %d events, %d windows\nredis: %d keys\n"+
		"shared database: %d policies and %d limits to delete; %d resets, %d grants and %d campaigns to rewrite\n",
		requests, service.Events, service.Windows, keys,
		settings.Policies, settings.Limits, settings.Resets, settings.Grants, settings.Campaigns)
	return err
}

// forget replaces the id, in the order each step needs: the shards first, so nothing ships the
// id again; the usage service twice, the second after settle, for a batch read before the
// shards were rewritten; Redis once the service writes no counter of the id any more; and the
// shared database last, with the audit row in the same transaction.
func (f forgetting) forget(ctx context.Context, settle time.Duration) error {
	if _, err := fmt.Fprintf(f.out, "replacing the user's id with %s\n", f.replacement); err != nil {
		return err
	}
	organizations, err := f.store.UserOrganizations(ctx, f.app, f.user)
	if err != nil {
		return err
	}
	requests, err := f.store.RenameUserRequests(ctx, f.app, f.user, f.replacement)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(f.out, "requests: %d rows rewritten\n", requests); err != nil {
		return err
	}
	forget := usageservice.ForgetUser{AppID: f.app, UserID: f.user, Replacement: f.replacement}
	first, err := f.usage.ForgetUser(ctx, f.actor, forget)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(f.out, "usage service: %d events, %d windows rewritten; again in %s\n", first.Events, first.Windows, settle); err != nil {
		return err
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(settle):
	}
	second, err := f.usage.ForgetUser(ctx, f.actor, forget)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(f.out, "usage service after %s: %d events, %d windows rewritten\n", settle, second.Events, second.Windows); err != nil {
		return err
	}
	var keys int64
	for _, client := range f.redis {
		for _, response := range client.DoMulti(ctx, f.commands(client, false, organizations, first.Organizations, second.Organizations)...) {
			deleted, err := response.AsInt64()
			if err != nil {
				return fmt.Errorf("usage forget-user: redis: %w", err)
			}
			keys += deleted
		}
	}
	if _, err := fmt.Fprintf(f.out, "redis: %d keys deleted\n", keys); err != nil {
		return err
	}
	entry := &store.AuditEntry{
		CustomerID: f.app, ResourceType: store.AuditEndUser, ResourceID: f.replacement,
		Action: store.AuditUpdated, Source: store.AuditSourceCLI, ActorID: f.actor,
		Changes: []store.AuditChange{
			{Field: "requests", After: requests}, {Field: "events", After: first.Events + second.Events},
			{Field: "windows", After: first.Windows + second.Windows}, {Field: "redis_keys", After: keys},
		},
	}
	settings, err := f.store.ForgetUserSettings(ctx, f.app, f.user, f.replacement, entry)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(f.out, "shared database: %d policies and %d limits deleted; %d resets, %d grants and %d campaigns rewritten\naudit: %s\n",
		settings.Policies, settings.Limits, settings.Resets, settings.Grants, settings.Campaigns, entry.ID)
	return err
}

// commands are, for one client, an EXISTS (exists) or a DEL of each key the user may have: their
// spend counter and holds in each organization, and their quota buckets for every day not yet
// expired. One command per key, since a cluster refuses keys of two slots in one.
func (f forgetting) commands(client rueidis.Client, exists bool, organizations ...[]string) []rueidis.Completed {
	keys := quota.UserKeys(f.app, f.user, time.Now())
	every := slices.Concat(organizations...)
	slices.Sort(every)
	for _, organization := range slices.Compact(every) {
		user := usageservice.SubjectRef{OrganizationID: organization, Subject: usageservice.SubjectUser, AppID: f.app, UserID: f.user}
		keys = append(keys, user.Key(usageservice.CounterPrefix), policy.HoldKey(user))
	}
	commands := make([]rueidis.Completed, 0, len(keys))
	for _, key := range keys {
		if exists {
			commands = append(commands, client.B().Exists().Key(key).Build())
		} else {
			commands = append(commands, client.B().Del().Key(key).Build())
		}
	}
	return commands
}
