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
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
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
	case "storage":
		return runUsageStorage(ctx, args[1:], settings, out)
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
