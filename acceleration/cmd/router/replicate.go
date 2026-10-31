package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

const replicateUsage = `usage: router replicate --from <url> --api-key <key> --api-secret <secret> [flags]
       router replicate --resume --as <app id>

  --from        the deployment to copy from, for example https://accelerate-benchmark.gcp.stream-io-api.com
  --api-key     a server-side key there, vak_live_...
  --api-secret  its secret, vas_live_...
  --as          the app id the rows land under here. Defaults to the one they had
  --source-auth jwt for a router behind the Stream gateway (the default), server for one
                reached directly
  --poll        how often to ask for changes once the copy is done
  --once        stop as soon as there is nothing left to catch up on
  --resume      the cut-over, once the deployment copied from no longer delivers: resume
                the scheduled tasks the copy paused because they were active there, each
                at its first occurrence after now, and name the campaigns that were
                running there, to start here

Until the last --once the deployment copied from is the truth: a change it makes to a
row overwrites what was done to that row here, a delete included. Once SDKs point here,
stop writing there.
`

// tokenLifetime is how long the token this command signs for itself is good for. It is
// re-signed per request, so it only has to outlive one.
const tokenLifetime = 5 * time.Minute

// runReplicate copies another deployment's data here and then follows it.
//
// A move is a copy plus everything that happens after the copy, and doing the second part
// is what makes the switchover free of a window where writes are lost. The copy is read
// at one moment; the changes carry on from exactly where it stopped; both are applied the
// same way, by upserting, so anything applied twice lands where applying it once would.
func runReplicate(args []string, settings config.Config, logger *slog.Logger) error {
	options, err := parseReplicate(args)
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	pgStore, err := openStore(ctx, settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()
	if options.resume {
		return resumeMove(ctx, pgStore, options.as, os.Stdout)
	}
	// An imported row may be pinned only to the app its customer acts in here, and one
	// pinned to the deployment's own app can be placed only once that app is known.
	secrets, err := newSecretSealer(ctx, settings)
	if err != nil {
		return err
	}
	streamClients, err := newStreamClients(settings, pgStore, secrets, logger)
	if err != nil {
		return err
	}
	learning, cancel := context.WithTimeout(ctx, learnTimeout)
	if _, err := streamClients.LearnDeploymentApp(learning); err != nil && !errors.Is(err, streamapp.ErrNoIdentity) {
		logger.Warn("stream: could not learn which Stream app is this deployment's own, so a row "+
			"pinned to one will be refused", "error", err)
	}
	cancel()
	pgStore.SetStreamPins(streamPins(streamClients))

	reservation, err := pgStore.ReserveImportedIDs(ctx)
	if err != nil {
		return err
	}
	customerID, cursor, err := copyEverything(ctx, options.source, pgStore, reservation, options.as, logger)
	if err != nil {
		return err
	}

	return follow(ctx, options.source, pgStore, reservation, customerID, cursor, options.poll, options.once, logger)
}

// replicateOptions is what the command line asked replicate for.
type replicateOptions struct {
	source *deployment
	as     string
	poll   time.Duration
	once   bool
	// resume is the cut-over, which reads no source.
	resume bool
}

// parseReplicate reads replicate's flags into the source it copies from and how.
func parseReplicate(args []string) (replicateOptions, error) {
	flags := flag.NewFlagSet("replicate", flag.ContinueOnError)
	from := flags.String("from", "", "the deployment to copy from")
	key := flags.String("api-key", "", "a server-side key there")
	secret := flags.String("api-secret", "", "its secret")
	// The Stream gateway in front of a hosted router accepts only a jwt Stream-Auth-Type and
	// takes a token signed with the app secret that names no user for a server-side one.
	// A router reached directly makes a caller server-side only when it says server.
	sourceAuth := flags.String("source-auth", auth.AuthTypeJWT, "the Stream-Auth-Type the source accepts: jwt or server")
	as := flags.String("as", "", "the app id the rows land under here")
	poll := flags.Duration("poll", 2*time.Second, "how often to ask for changes")
	once := flags.Bool("once", false, "stop once there is nothing left to catch up on")
	resume := flags.Bool("resume", false, "resume what the copy paused")
	if err := flags.Parse(args); err != nil {
		return replicateOptions{}, err
	}
	if *resume {
		// The cut-over reads only this deployment: one that also caught up first would read
		// the source's own last writes, a pause made there to stop it delivering among them.
		if *as == "" || *from != "" {
			return replicateOptions{}, errors.New(replicateUsage)
		}
		return replicateOptions{as: *as, resume: true}, nil
	}
	if *from == "" || *key == "" || *secret == "" {
		return replicateOptions{}, errors.New(replicateUsage)
	}
	if *sourceAuth != auth.AuthTypeJWT && *sourceAuth != auth.AuthTypeServer {
		return replicateOptions{}, fmt.Errorf("--source-auth is jwt or server, not %q", *sourceAuth)
	}
	return replicateOptions{
		source: &deployment{
			url:      strings.TrimSuffix(*from, "/"),
			key:      *key,
			secret:   *secret,
			authType: *sourceAuth,
			client:   &http.Client{Timeout: time.Hour},
		},
		as:   *as,
		poll: *poll,
		once: *once,
	}, nil
}

// copyEverything reads the export and writes it here, returning who it turned out to be
// and the cursor its changes carry on from.
func copyEverything(ctx context.Context, source *deployment, pgStore *store.Store, reservation *store.IDReservation, as string, logger *slog.Logger) (string, int64, error) {
	body, err := source.get(ctx, "/v1/data/export", nil)
	if err != nil {
		return "", 0, err
	}
	defer body.Close()

	customerID := as
	var cursor *int64
	rows := int64(0)
	decoder := json.NewDecoder(body)
	for {
		var line exportLine
		if err := decoder.Decode(&line); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return "", 0, fmt.Errorf("reading the export: %w", err)
		}
		switch {
		case line.Table != "":
			if customerID == "" {
				// Before the last line says who this is, the rows themselves do. An app
				// keeping its id is the ordinary case, since that is what every row's
				// customer_id holds and what --app-id on `keys create` is for.
				customerID = customerOf(line.Row)
			}
			if err := reservation.Cover(ctx, line.Table, line.Row); err != nil {
				return "", 0, err
			}
			if err := pgStore.ImportRow(ctx, customerID, line.Table, line.Row); err != nil {
				return "", 0, err
			}
			rows++
		case line.Cursor != nil:
			cursor = line.Cursor
			if as == "" && line.Customer != "" {
				customerID = line.Customer
			}
		}
	}
	if cursor == nil {
		return "", 0, errors.New("the export stopped before it finished: nothing was written that could be caught up from, so run this again")
	}
	if customerID == "" {
		return "", 0, errors.New("the export named no customer: pass --as to say which app these rows are for")
	}

	logger.Info("copied a customer", "customer", customerID, "rows", rows, "cursor", *cursor)
	return customerID, *cursor, nil
}

// follow replays what happens at the source from the cursor onwards, until there is
// nothing left and the caller asked to stop, or until the process is interrupted.
func follow(ctx context.Context, source *deployment, pgStore *store.Store, reservation *store.IDReservation, customerID string, cursor int64, poll time.Duration, once bool, logger *slog.Logger) error {
	ticker := time.NewTicker(poll)
	defer ticker.Stop()

	for {
		body, err := source.get(ctx, "/v1/data/changes", map[string]string{
			"after": fmt.Sprint(cursor),
		})
		if err != nil {
			return err
		}
		var page changePage
		err = json.NewDecoder(body).Decode(&page)
		body.Close()
		if err != nil {
			return fmt.Errorf("reading changes: %w", err)
		}

		if len(page.Changes) > 0 {
			for _, change := range page.Changes {
				if change.Op == store.ChangeDelete {
					continue
				}
				if err := reservation.Cover(ctx, change.Table, change.Payload); err != nil {
					return err
				}
			}
			if err := pgStore.ApplyChanges(ctx, customerID, page.Changes); err != nil {
				return err
			}
			cursor = page.Cursor
			logger.Info("caught up on changes", "changes", len(page.Changes), "cursor", cursor)
		}

		if page.CaughtUp {
			if once {
				fmt.Printf("caught up at cursor %d. Point your SDKs here, then run this once more to pick up the last writes. "+
					"Once the deployment copied from stops delivering, run it with --resume --as %s.\n", cursor, customerID)
				return nil
			}
			fmt.Printf("caught up at cursor %d\n", cursor)
		}

		select {
		case <-ctx.Done():
			logger.Info("stopped following", "cursor", cursor)
			return nil
		case <-ticker.C:
		}
	}
}

// resumeMove is the cut-over: it resumes what the copy paused and names what is left to start.
func resumeMove(ctx context.Context, pgStore *store.Store, customerID string, out io.Writer) error {
	cutOver := time.Now()
	if err := pgStore.SkipOccurrencesBeforeCutOver(ctx, customerID, cutOver, nextAfter(cutOver)); err != nil {
		return err
	}
	resumed, skipped, overLimit, campaigns, err := pgStore.ResumeMovedRows(ctx, customerID)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "resumed %d scheduled tasks\n", resumed)
	if skipped > 0 {
		fmt.Fprintf(out, "left %d scheduled tasks paused: their agent config is deleted or missing here\n", skipped)
	}
	if overLimit > 0 {
		fmt.Fprintf(out, "left %d scheduled tasks paused: their user already holds the most active tasks the agent config allows. "+
			"Run --resume again once the user has made room; the oldest are resumed first\n", overLimit)
	}
	for _, id := range campaigns {
		fmt.Fprintf(out, "campaign %s was running where it came from: start it here with POST /v1/agents/campaigns/%s/start\n", id, id)
	}
	return nil
}

// nextAfter is a moved task's next occurrence after the cut-over, as a resume counts it
// (resumeScheduledTask); with none left, a one-shot's is the missed run of the time it slept
// through, and a repeat has none.
func nextAfter(cutOver time.Time) func(store.ScheduledTask) (*time.Time, *store.ScheduledTaskRun, error) {
	return func(task store.ScheduledTask) (*time.Time, *store.ScheduledTaskRun, error) {
		schedule, err := scheduling.FromStored(task)
		if err != nil {
			return nil, nil, fmt.Errorf("scheduled task %s: %w", task.ID, err)
		}
		if next, ok := schedule.Next(cutOver); ok {
			return &next, nil, nil
		}
		return nil, scheduling.SleptThrough(task), nil
	}
}

// exportLine is a line of the export, which is either a row or the cursor that ends it.
type exportLine struct {
	Cursor   *int64          `json:"cursor"`
	Customer string          `json:"customer"`
	Table    string          `json:"table"`
	Row      json.RawMessage `json:"row"`
}

type changePage struct {
	Changes  []store.DataChange `json:"changes"`
	Cursor   int64              `json:"cursor"`
	CaughtUp bool               `json:"caught_up"`
}

// deployment is the router being copied from.
type deployment struct {
	url    string
	key    string
	secret string
	// authType is the Stream-Auth-Type sent with the server token: AuthTypeJWT or AuthTypeServer.
	authType string
	client   *http.Client
}

// get asks the source for something, as the backend it is.
func (d *deployment) get(ctx context.Context, path string, query map[string]string) (io.ReadCloser, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, d.url+path, nil)
	if err != nil {
		return nil, err
	}
	values := request.URL.Query()
	for name, value := range query {
		values.Set(name, value)
	}
	// The Stream gateway in front of a hosted router reads the key from the query string,
	// not from X-Api-Key; a router reached directly reads it from either.
	values.Set(auth.APIKeyParam, d.key)
	request.URL.RawQuery = values.Encode()

	token, err := auth.ServerToken(d.secret, tokenLifetime)
	if err != nil {
		return nil, err
	}
	request.Header.Set(auth.APIKeyHeader, d.key)
	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set(auth.AuthTypeHeader, d.authType)

	response, err := d.client.Do(request)
	if err != nil {
		return nil, err
	}
	if response.StatusCode == http.StatusGone {
		response.Body.Close()
		return nil, errors.New("the source no longer keeps the changes since that cursor: start again with a fresh export")
	}
	if response.StatusCode == http.StatusUnauthorized || response.StatusCode == http.StatusForbidden {
		message, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
		response.Body.Close()
		other := auth.AuthTypeServer
		if d.authType == auth.AuthTypeServer {
			other = auth.AuthTypeJWT
		}
		return nil, fmt.Errorf("%s %s: %s: %s (sent with --source-auth %s; a source that wants the other kind takes --source-auth %s)",
			request.Method, path, response.Status, strings.TrimSpace(string(message)), d.authType, other)
	}
	if response.StatusCode != http.StatusOK {
		message, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
		response.Body.Close()
		return nil, fmt.Errorf("%s %s: %s: %s", request.Method, path, response.Status, strings.TrimSpace(string(message)))
	}
	return response.Body, nil
}

// customerOf reads whose row this is, which is how the copy learns the app id when the
// caller did not name one.
func customerOf(row json.RawMessage) string {
	var fields struct {
		CustomerID string `json:"customer_id"`
	}
	if err := json.Unmarshal(row, &fields); err != nil {
		return ""
	}
	return fields.CustomerID
}
