package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"os"
	"slices"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

const streamAppsUsage = `usage: router stream-apps <command> [flags]

  fallbacks       who app mode still writes into the deployment's own Stream app
    --since       how far back to look, as 14d or 36h (default 14d)
    --by-customer list each customer rather than counting them

  backfill-pins   pin every unpinned session, call and number to the deployment's
                  own Stream app, which is what an unpinned row meant. App mode only.
`

// runStreamApps looks after the Stream apps customers registered in app mode.
func runStreamApps(args []string, settings config.Config, logger *slog.Logger) error {
	if len(args) == 0 {
		return errors.New(streamAppsUsage)
	}
	ctx := context.Background()
	switch args[0] {
	case "fallbacks":
		return runFallbacks(ctx, args[1:], settings, os.Stdout)
	case "backfill-pins":
		return runBackfillPins(ctx, settings, logger, os.Stdout)
	}
	return fmt.Errorf("unknown stream-apps command %q\n\n%s", args[0], streamAppsUsage)
}

func runFallbacks(ctx context.Context, args []string, settings config.Config, out io.Writer) error {
	flags := flag.NewFlagSet("stream-apps fallbacks", flag.ContinueOnError)
	since := flags.String("since", "14d", "how far back to look")
	byCustomer := flags.Bool("by-customer", false, "list each customer")
	if err := flags.Parse(args); err != nil {
		return err
	}
	window, err := lookBack(*since)
	if err != nil {
		return err
	}
	pgStore, err := openStore(ctx, settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()

	uses, err := pgStore.StreamFallbackUses(ctx, time.Now().Add(-window))
	if err != nil {
		return err
	}
	return printFallbacks(out, uses, *since, *byCustomer)
}

// printFallbacks says how many customers used the fallback, and with byCustomer, which.
// Without it no customer is named, so the count can be passed on as it is.
func printFallbacks(out io.Writer, uses []store.StreamFallbackUse, since string, byCustomer bool) error {
	var total int64
	for _, use := range uses {
		total += use.Uses
	}
	if _, err := fmt.Fprintf(out, "%d customers wrote into the deployment's app %d times in the last %s\n",
		len(uses), total, since); err != nil {
		return err
	}
	if !byCustomer || len(uses) == 0 {
		return nil
	}
	table := tabwriter.NewWriter(out, 0, 4, 2, ' ', 0)
	fmt.Fprintln(table, "CUSTOMER\tUSES\tFIRST\tLAST")
	sorted := slices.Clone(uses)
	slices.SortFunc(sorted, func(a, b store.StreamFallbackUse) int { return b.LastAt.Compare(a.LastAt) })
	for _, use := range sorted {
		fmt.Fprintf(table, "%s\t%d\t%s\t%s\n", use.CustomerID, use.Uses,
			use.FirstAt.Format(time.RFC3339), use.LastAt.Format(time.RFC3339))
	}
	return table.Flush()
}

// lookBack reads a window as Go writes durations, or as a number of days.
func lookBack(window string) (time.Duration, error) {
	if days, ok := strings.CutSuffix(window, "d"); ok {
		n, err := strconv.Atoi(days)
		if err != nil || n < 1 {
			return 0, fmt.Errorf("--since is a number of days such as 14d, or a duration such as 36h, not %q", window)
		}
		return time.Duration(n) * 24 * time.Hour, nil
	}
	duration, err := time.ParseDuration(window)
	if err != nil || duration <= 0 {
		return 0, fmt.Errorf("--since is a number of days such as 14d, or a duration such as 36h, not %q", window)
	}
	return duration, nil
}

func runBackfillPins(ctx context.Context, settings config.Config, logger *slog.Logger, out io.Writer) error {
	if settings.Stream.Tenancy != config.TenancyApp {
		return fmt.Errorf("backfill-pins is for stream.tenancy=%s, where every row names its app; "+
			"deployment mode reads an unpinned row as the deployment's own already", config.TenancyApp)
	}
	pgStore, err := openStore(ctx, settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()
	secrets, err := newSecretSealer(settings)
	if err != nil {
		return err
	}
	clients, err := newStreamClients(settings, pgStore, secrets, logger)
	if err != nil {
		return err
	}
	learning, cancel := context.WithTimeout(ctx, learnTimeout)
	defer cancel()
	deployment, err := clients.LearnDeploymentApp(learning)
	if err != nil {
		return fmt.Errorf("backfill-pins needs the deployment's own Stream app id, which could not be read: %w", err)
	}

	pinned, err := pgStore.BackfillStreamPins(ctx, deployment)
	for _, table := range slices.Sorted(maps.Keys(pinned)) {
		fmt.Fprintf(out, "%s: %d rows pinned to app %d\n", table, pinned[table], deployment)
	}
	return err
}
