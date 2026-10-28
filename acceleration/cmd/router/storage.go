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
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/providerkeys"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// storageMissed is the alert a storage run sends, once, naming the hour, when it could not
// read a knowledge base: that hour of it is not billed until a run for the hour reads it.
const storageMissed = "storage_missed"

// storageOrigin is the key every storage row is recorded under: the router's own hourly job.
var storageOrigin = routing.JobOrigin(metered.KnowledgeStorageKind)

// storageReadTimeout bounds one knowledge base's metadata read. Nobody is waiting on it, so
// it is longer than a lookup's.
const storageReadTimeout = 30 * time.Second

// storageMissedListed caps how many missed knowledge bases the alert names; the log names
// every one.
const storageMissedListed = 20

// runUsageStorage records an hour of every knowledge base's size, through a writer of its own
// that ships to the usage service when one is set. What it has not shipped when it exits, the
// routers' sweeper ships.
func runUsageStorage(ctx context.Context, args []string, settings config.Config, out io.Writer) error {
	flags := flag.NewFlagSet("usage storage", flag.ContinueOnError)
	hourText := flags.String("hour", "", "the hour to record, RFC 3339; empty is the last hour that has ended")
	if err := flags.Parse(args); err != nil {
		return err
	}
	hour, err := storageHourOf(*hourText, time.Now())
	if err != nil {
		return err
	}
	logger := slog.Default()

	backends, err := knowledgeBackends(settings.Knowledge, storageReadTimeout, logger)
	if err != nil {
		return fmt.Errorf("usage storage: %w", err)
	}
	pgStore, err := openPriceStore(settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()

	recording, stop, err := jobRecorder(ctx, settings, pgStore, routing.Knowledge, logger)
	if err != nil {
		return err
	}
	defer stop()
	base := metered.NewRegionalKnowledge(backends, recording, nil, pgStore)
	// The bases kept in customers' own accounts are read on their keys, which only the keyring
	// they were sealed under opens.
	secrets, err := newSecretSealer(ctx, settings)
	if err != nil {
		return fmt.Errorf("usage storage: %w", err)
	}
	if secrets != nil {
		base.WithCustomerAccounts(providerkeys.NewKeyring(pgStore, secrets), knowledgeAccounts(settings.Knowledge, storageReadTimeout, logger))
	}
	return recordStorage(ctx, base, pgStore, hour, logger, out)
}

// storageHourOf is the hour a run records: the one --hour names, or else the last one to have
// ended, so a run, and its retries within the next hour, bill the same hour.
func storageHourOf(text string, now time.Time) (time.Time, error) {
	if text == "" {
		return now.UTC().Truncate(time.Hour).Add(-time.Hour), nil
	}
	at, err := time.Parse(time.RFC3339, text)
	if err != nil {
		return time.Time{}, fmt.Errorf("usage storage: --hour %q is not an RFC 3339 time", text)
	}
	return at.UTC().Truncate(time.Hour), nil
}

// jobRecorder is the recorder a job's rows of modality are written through, by a writer of the
// job's own that ships to the usage service when one is set, priced by the list in force, with
// the policies saying whose app each row's organization is and its tier. stop drains the
// writer; what it has not shipped by then, the routers' sweeper ships.
func jobRecorder(ctx context.Context, settings config.Config, pgStore *store.Store, modality routing.Modality,
	logger *slog.Logger) (*routing.Recorder, func(), error) {
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
	// The job admits nothing, so it reads no money and holds none: its policies only attribute
	// each row.
	policies, err := policy.New(configs, policy.Settings{Mode: config.LimitsOff}, logger)
	if err != nil {
		configs.Close()
		return nil, nil, err
	}
	fallback, err := prices.Embedded(nil, release, logger)
	if err != nil {
		policies.Close()
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
	return routing.NewRecorderOn(modality, writer, active, logger), func() {
		drained, cancel := context.WithTimeout(context.WithoutCancel(ctx), usageDrainTimeout)
		defer cancel()
		writer.Close(drained)
		stopReading()
		policies.Close()
		configs.Close()
	}, nil
}

// recordStorage writes one storage row for the hour per knowledge base of an app this
// deployment knows, in every region it keeps knowledge in, owned by the app under the storage
// job's origin, at the size the base holds now, priced in its region. A rerun for the hour
// writes nothing twice: each row's usage id is the namespace and the hour. A base that holds
// nothing writes no row. A base of an app this deployment has never seen is another
// deployment's on the same key, and is left to that one's job. The bases kept in a customer's
// own account, which the deployment's listing does not see, are the ones recorded so, each
// read on the key its owner keeps there and priced at the vendor's BYOK entry; one whose key is
// gone is not billed. A region or a base it cannot read is logged at ERROR, and the run sends
// storage_missed for the hour once and fails, so a retry, or a run with --hour, fills the hour in.
func recordStorage(ctx context.Context, base *metered.Knowledge, pgStore *store.Store, hour time.Time, logger *slog.Logger, out io.Writer) error {
	at := hour.UTC().Truncate(time.Hour).Format(time.RFC3339)
	type located struct {
		place     store.KnowledgePlace
		namespace string
	}
	var namespaces []located
	for _, region := range base.Regions() {
		listed, err := base.Namespaces(ctx, region, "")
		if err != nil {
			logger.Error("could not list the knowledge bases, so no storage is recorded for the hour", "hour", at, "region", region, "error", err)
			alerts.SendItem(storageMissed, at, alerts.Warn, map[string]any{"hour": at, "region": region, "listing": err.Error()})
			return fmt.Errorf("usage storage: list the knowledge bases in %s: %w", region, err)
		}
		for _, namespace := range listed {
			namespaces = append(namespaces, located{place: store.KnowledgePlace{Region: region, Account: store.KnowledgeAccountStream}, namespace: namespace})
		}
	}
	theirs, err := pgStore.KnowledgeInCustomerAccounts(ctx)
	if err != nil {
		logger.Error("could not list the knowledge bases in customers' own accounts, so no storage is recorded for the hour", "hour", at, "error", err)
		alerts.SendItem(storageMissed, at, alerts.Warn, map[string]any{"hour": at, "listing": err.Error()})
		return fmt.Errorf("usage storage: %w", err)
	}
	for _, kept := range theirs {
		namespaces = append(namespaces, located{place: kept.Place(), namespace: knowledge.Scoped(kept.CustomerID, kept.Namespace)})
	}
	customers := map[string]string{}
	var ids []string
	for _, found := range namespaces {
		customerID, _, scoped := strings.Cut(found.namespace, "__")
		if scoped && customerID != "" {
			customers[found.namespace] = customerID
			ids = append(ids, customerID)
		}
	}
	known, err := pgStore.KnownApps(ctx, ids)
	if err != nil {
		alerts.SendItem(storageMissed, at, alerts.Warn, map[string]any{"hour": at, "apps": err.Error()})
		return fmt.Errorf("usage storage: which apps are this deployment's: %w", err)
	}

	var recorded, empty, gone, notOurs, keyless int
	var missed []string
	for _, found := range namespaces {
		customerID, scoped := customers[found.namespace]
		if !scoped || !known[customerID] {
			notOurs++
			continue
		}
		owner := routing.Owner{CustomerID: customerID, Attribution: routing.Attribution{APIKey: storageOrigin}}
		stored, err := base.StorageAt(ctx, owner, found.place, found.namespace, hour)
		switch {
		case err == nil && stored.ByteHours == 0:
			empty++
		case err == nil:
			recorded++
		case errors.Is(err, knowledge.ErrNoNamespace):
			// Deleted since it was listed, so there is nothing of it to bill.
			gone++
		case errors.Is(err, metered.ErrNoAccountKey):
			// In the customer's own account, which nothing reaches once they delete its key.
			keyless++
		default:
			logger.Error("could not read a knowledge base's size, so this hour of it is not billed",
				"namespace", found.namespace, "region", found.place.Region, "account", found.place.Account,
				"customer_id", customerID, "hour", at, "error", err)
			missed = append(missed, found.namespace)
		}
	}

	if _, err := fmt.Fprintf(out, "%s: %d knowledge bases recorded, %d empty, %d missed, %d gone since listed, %d not this deployment's, "+
		"%d in a customer's own account whose key is gone\n",
		at, recorded, empty, len(missed), gone, notOurs, keyless); err != nil {
		return err
	}
	if len(missed) == 0 {
		return nil
	}
	alerts.SendItem(storageMissed, at, alerts.Warn, map[string]any{
		"hour": at, "missed": len(missed), "namespaces": missed[:min(len(missed), storageMissedListed)],
		"rerun": "router usage storage --hour " + at,
	})
	return fmt.Errorf("usage storage: %d of %d knowledge bases could not be read for %s; run router usage storage --hour %s once they can",
		len(missed), len(namespaces), at, at)
}
