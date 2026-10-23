package main

import (
	"cmp"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/turbopuffer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// storageMissed is the alert a storage run sends, once, when it could not read a knowledge
// base: that hour of it is not billed.
const storageMissed = "storage_missed"

// storageOrigin is the key every storage row is recorded under: the router's own hourly job.
var storageOrigin = routing.JobOrigin(metered.KnowledgeStorageKind)

// storageReadTimeout bounds one knowledge base's metadata read. Nobody is waiting on it, so
// it is longer than a lookup's.
const storageReadTimeout = 30 * time.Second

// storageMissedListed caps how many missed knowledge bases the alert names; the log names
// every one.
const storageMissedListed = 20

// runUsageStorage records this hour of every knowledge base's size, through a writer of its
// own that ships to the usage service when one is set. What it has not shipped when it exits,
// the routers' sweeper ships.
func runUsageStorage(ctx context.Context, args []string, settings config.Config, out io.Writer) error {
	flags := flag.NewFlagSet("usage storage", flag.ContinueOnError)
	hourText := flags.String("hour", "", "the hour to record, RFC 3339; empty is the hour now")
	if err := flags.Parse(args); err != nil {
		return err
	}
	hour := time.Now()
	if *hourText != "" {
		at, err := time.Parse(time.RFC3339, *hourText)
		if err != nil {
			return fmt.Errorf("usage storage: --hour %q is not an RFC 3339 time", *hourText)
		}
		hour = at
	}
	logger := slog.Default()

	backend, err := turbopuffer.New(turbopuffer.Options{
		Cost: knowledgeCost(settings.Knowledge), Timeout: storageReadTimeout, Logger: logger,
	})
	if err != nil {
		return fmt.Errorf("usage storage: %w", err)
	}
	pgStore, err := openPriceStore(settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()

	recording, stop, err := storageRecorder(ctx, settings, pgStore, logger)
	if err != nil {
		return err
	}
	defer stop()
	return recordStorage(ctx, metered.NewKnowledge(backend, recording), hour, logger, out)
}

// storageRecorder is the recorder storage rows are written through, priced by the list in
// force, with the policies saying whose app each row's organization is and its tier. stop
// drains the writer.
func storageRecorder(ctx context.Context, settings config.Config, pgStore *store.Store, logger *slog.Logger) (*routing.Recorder, func(), error) {
	usageClient, err := newUsageClient(settings, pgStore, logger)
	if err != nil {
		return nil, nil, err
	}
	configs, err := appconfig.New(appconfig.Options{
		Store: pgStore, Address: settings.Redis.Addr, Username: settings.Redis.Username, Password: settings.Redis.Password, Logger: logger,
	})
	if err != nil {
		return nil, nil, err
	}
	policies, err := policy.New(configs, logger)
	if err != nil {
		configs.Close()
		return nil, nil, err
	}
	fallback, err := prices.Embedded(nil, release, logger)
	if err != nil {
		configs.Close()
		return nil, nil, err
	}
	reading, stopReading := context.WithCancel(ctx)
	active := pricing.NewActive(reading, pricing.ActiveOptions{
		Source: pgStore.PriceSource(), Fallback: fallback, Interval: time.Hour, Logger: logger,
	})
	host, _ := os.Hostname()
	writer := routing.NewWriter(routing.WriterOptions{
		Store: pgStore, Logger: logger, Gate: policies, Usage: usageClient,
		Source: routing.Source(cmp.Or(settings.Shard, routing.Unsharded), host),
	})
	return routing.NewRecorderOn(routing.Knowledge, writer, active, logger), func() {
		drained, cancel := context.WithTimeout(context.WithoutCancel(ctx), usageDrainTimeout)
		defer cancel()
		writer.Close(drained)
		stopReading()
		configs.Close()
	}, nil
}

// recordStorage writes one storage row per customer's knowledge base for the hour, owned by
// its app under the storage job's origin. A rerun within the hour writes nothing twice: each
// row's usage id is the namespace and the hour. A knowledge base it cannot read is logged at
// ERROR, and the run sends storage_missed once and fails, so a retry fills the hour in.
func recordStorage(ctx context.Context, base *metered.Knowledge, hour time.Time, logger *slog.Logger, out io.Writer) error {
	hour = hour.UTC().Truncate(time.Hour)
	namespaces, err := base.Namespaces(ctx, "")
	if err != nil {
		logger.Error("could not list the knowledge bases, so no storage is recorded this hour",
			"hour", hour.Format(time.RFC3339), "error", err)
		alerts.Send(storageMissed, alerts.Warn, map[string]any{"hour": hour.Format(time.RFC3339), "listing": err.Error()})
		return fmt.Errorf("usage storage: list the knowledge bases: %w", err)
	}

	var recorded, notCustomers, gone int
	var missed []string
	for _, namespace := range namespaces {
		customerID, _, scoped := strings.Cut(namespace, "__")
		if !scoped || customerID == "" {
			notCustomers++
			continue
		}
		owner := routing.Owner{CustomerID: customerID, Attribution: routing.Attribution{APIKey: storageOrigin}}
		err := base.Storage(ctx, owner, namespace, hour)
		switch {
		case err == nil:
			recorded++
		case errors.Is(err, knowledge.ErrNoNamespace):
			// Deleted since it was listed, so there is nothing of it to bill.
			gone++
		default:
			logger.Error("could not read a knowledge base's size, so this hour of it is not billed",
				"namespace", namespace, "customer_id", customerID, "hour", hour.Format(time.RFC3339), "error", err)
			missed = append(missed, namespace)
		}
	}

	if _, err := fmt.Fprintf(out, "%s: %d knowledge bases recorded, %d missed, %d gone since listed, %d not a customer's\n",
		hour.Format(time.RFC3339), recorded, len(missed), gone, notCustomers); err != nil {
		return err
	}
	if len(missed) == 0 {
		return nil
	}
	alerts.Send(storageMissed, alerts.Warn, map[string]any{
		"hour": hour.Format(time.RFC3339), "missed": len(missed), "namespaces": missed[:min(len(missed), storageMissedListed)],
	})
	return fmt.Errorf("usage storage: %d of %d knowledge bases could not be read", len(missed), len(namespaces))
}
