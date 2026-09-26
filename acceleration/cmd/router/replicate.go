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
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const replicateUsage = `usage: router replicate --from <url> --api-key <key> --api-secret <secret> [flags]

  --from        the deployment to copy from, for example https://accelerate.gcp.stream-io-api.com
  --api-key     a server-side key there, vak_live_...
  --api-secret  its secret, vas_live_...
  --as          the app id the rows land under here. Defaults to the one they had
  --poll        how often to ask for changes once the copy is done
  --once        stop as soon as there is nothing left to catch up on
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
	flags := flag.NewFlagSet("replicate", flag.ContinueOnError)
	from := flags.String("from", "", "the deployment to copy from")
	key := flags.String("api-key", "", "a server-side key there")
	secret := flags.String("api-secret", "", "its secret")
	as := flags.String("as", "", "the app id the rows land under here")
	poll := flags.Duration("poll", 2*time.Second, "how often to ask for changes")
	once := flags.Bool("once", false, "stop once there is nothing left to catch up on")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *from == "" || *key == "" || *secret == "" {
		return errors.New(replicateUsage)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	pgStore, err := openStore(ctx, settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()

	source := &deployment{
		url:    strings.TrimSuffix(*from, "/"),
		key:    *key,
		secret: *secret,
		client: &http.Client{Timeout: time.Hour},
	}

	customerID, cursor, err := copyEverything(ctx, source, pgStore, *as, logger)
	if err != nil {
		return err
	}

	return follow(ctx, source, pgStore, customerID, cursor, *poll, *once, logger)
}

// copyEverything reads the export and writes it here, returning who it turned out to be
// and the cursor its changes carry on from.
func copyEverything(ctx context.Context, source *deployment, pgStore *store.Store, as string, logger *slog.Logger) (string, int64, error) {
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
func follow(ctx context.Context, source *deployment, pgStore *store.Store, customerID string, cursor int64, poll time.Duration, once bool, logger *slog.Logger) error {
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
			if err := pgStore.ApplyChanges(ctx, customerID, page.Changes); err != nil {
				return err
			}
			cursor = page.Cursor
			logger.Info("caught up on changes", "changes", len(page.Changes), "cursor", cursor)
		}

		if page.CaughtUp {
			if once {
				fmt.Printf("caught up at cursor %d. Point your SDKs here, then run this once more to pick up the last writes.\n", cursor)
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
	client *http.Client
}

// get asks the source for something, as the backend it is.
func (d *deployment) get(ctx context.Context, path string, query map[string]string) (io.ReadCloser, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, d.url+path, nil)
	if err != nil {
		return nil, err
	}
	if len(query) > 0 {
		values := request.URL.Query()
		for name, value := range query {
			values.Set(name, value)
		}
		request.URL.RawQuery = values.Encode()
	}

	token, err := auth.ServerToken(d.secret, tokenLifetime)
	if err != nil {
		return nil, err
	}
	request.Header.Set(auth.APIKeyHeader, d.key)
	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set(auth.AuthTypeHeader, auth.AuthTypeServer)

	response, err := d.client.Do(request)
	if err != nil {
		return nil, err
	}
	if response.StatusCode == http.StatusGone {
		response.Body.Close()
		return nil, errors.New("the source no longer keeps the changes since that cursor: start again with a fresh export")
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
