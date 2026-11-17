package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The alerts the telephony jobs send: a run that could not bill, naming what to run again, and
// numbers billed with no price, which bill nothing until they have one.
const (
	rentalsMissed   = "number_rental_missed"
	numbersUnpriced = "number_unpriced"
	dlcFeesMissed   = "dlc_fees_missed"
)

// unpricedListed caps how many numbers with no price an alert names; the log names every one.
const unpricedListed = 20

// jobFrom is a job's --from: the time before which nothing it bills began, required so a first
// run bills nothing from before the job was switched on.
func jobFrom(command, text string) (time.Time, error) {
	if text == "" {
		return time.Time{}, fmt.Errorf("%s: --from is required: the RFC 3339 time it bills from, such as when it was first scheduled", command)
	}
	from, err := time.Parse(time.RFC3339, text)
	if err != nil {
		return time.Time{}, fmt.Errorf("%s: --from %q is not an RFC 3339 time", command, text)
	}
	return from.UTC(), nil
}

// runUsageNumbers bills the months of held numbers after the one each was bought in, through
// a writer of its own that ships to the usage service when one is set.
func runUsageNumbers(ctx context.Context, args []string, settings config.Config, out io.Writer) error {
	flags := flag.NewFlagSet("usage numbers", flag.ContinueOnError)
	fromText := flags.String("from", "", "bill no month that began before this RFC 3339 time (required)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	from, err := jobFrom("usage numbers", *fromText)
	if err != nil {
		return err
	}
	logger := slog.Default()
	pgStore, err := openPriceStore(settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()
	recorder, stop, err := jobRecorder(ctx, settings, pgStore, routing.Phone, logger)
	if err != nil {
		return err
	}
	defer stop()
	return billRentals(ctx, pgStore, recorder, from, time.Now(), logger, out)
}

// billRentals runs one rental pass and says what it billed. A pass that cannot read the
// numbers is logged at ERROR, alerted on as number_rental_missed and fails, so the next run, or
// a rerun, bills it: the months it would have billed are named by their usage ids, so nothing
// is billed twice. A number billed with no price is alerted on as number_unpriced.
func billRentals(ctx context.Context, pgStore *store.Store, recorder *routing.Recorder, from, now time.Time,
	logger *slog.Logger, out io.Writer) error {
	at := now.UTC().Format(time.RFC3339)
	rented, err := phone.Rentals{Store: pgStore, Recorder: recorder, Logger: logger}.Bill(ctx, from, now)
	if err != nil {
		logger.Error("could not bill the months of held numbers", "error", err)
		alerts.SendItem(rentalsMissed, at, alerts.Warn, map[string]any{"at": at, "error": err.Error(),
			"rerun": "router usage numbers --from " + from.Format(time.RFC3339)})
		return fmt.Errorf("usage numbers: %w", err)
	}
	if len(rented.Unpriced) > 0 {
		logger.Error("billed months of numbers whose vendor named no price; they cost nothing until priced",
			"numbers", strings.Join(rented.Unpriced, ", "))
		alerts.SendItem(numbersUnpriced, at, alerts.Warn, map[string]any{
			"months": len(rented.Unpriced), "numbers": rented.Unpriced[:min(len(rented.Unpriced), unpricedListed)],
		})
	}
	_, err = fmt.Fprintf(out, "%s: %d months of numbers billed, %d with no price\n", at, rented.Months, len(rented.Unpriced))
	return err
}

// runUsageDLC bills the months of the 10DLC campaigns a vendor holds after the three charged
// when each was registered, through a writer of its own.
func runUsageDLC(ctx context.Context, args []string, settings config.Config, out io.Writer) error {
	flags := flag.NewFlagSet("usage dlc", flag.ContinueOnError)
	fromText := flags.String("from", "", "bill no month that began before this RFC 3339 time (required)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	from, err := jobFrom("usage dlc", *fromText)
	if err != nil {
		return err
	}
	logger := slog.Default()
	pgStore, err := openPriceStore(settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()
	recorder, stop, err := jobRecorder(ctx, settings, pgStore, routing.Phone, logger)
	if err != nil {
		return err
	}
	defer stop()
	return billCampaigns(ctx, pgStore, recorder, from, nil, logger, out)
}

// billCampaigns runs one pass over the campaigns' months and says what it billed, failing and
// alerting as dlc_fees_missed when it cannot read them. now, nil for the wall clock, is the
// clock the current month is read off.
func billCampaigns(ctx context.Context, pgStore *store.Store, recorder *routing.Recorder, from time.Time,
	now func() time.Time, logger *slog.Logger, out io.Writer) error {
	fees, err := dlc.NewService(dlc.Options{Store: pgStore, Recorder: recorder, Now: now, Logger: logger})
	if err != nil {
		return err
	}
	at := time.Now().UTC().Format(time.RFC3339)
	billed, err := fees.BillCampaigns(ctx, from)
	if err != nil {
		logger.Error("could not bill the months of registered campaigns", "error", err)
		alerts.SendItem(dlcFeesMissed, at, alerts.Warn, map[string]any{"at": at, "error": err.Error(),
			"rerun": "router usage dlc --from " + from.Format(time.RFC3339)})
		return fmt.Errorf("usage dlc: %w", err)
	}
	_, err = fmt.Fprintf(out, "%s: %d months of 10DLC campaigns billed\n", at, billed)
	return err
}
