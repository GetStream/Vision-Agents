package main

import (
	"cmp"
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
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

const streamAppsUsage = `usage: router stream-apps <command> [flags]

  require         keep every app of an organization out of the deployment's own Stream
                  app (--org), or let them back in (--off)
  legacy          how much other customers still have in the deployment's own Stream
                  app, by kind
    --by-customer list each customer rather than counting them

  fallbacks       who app mode still writes into the deployment's own Stream app
    --since       how far back to look, as 14d or 36h (default 14d)
    --by-customer list each customer rather than counting them

  backfill-pins   pin every unpinned session, call and number to the deployment's
                  own Stream app, which is what an unpinned row meant. App mode only.
`

// runStreamApps looks after where customers act in Stream in app mode.
func runStreamApps(args []string, settings config.Config, logger *slog.Logger) error {
	if len(args) == 0 {
		return errors.New(streamAppsUsage)
	}
	ctx := context.Background()
	switch args[0] {
	case "require":
		return runRequire(ctx, args[1:], settings, os.Stdout)
	case "legacy":
		return runLegacy(ctx, args[1:], settings, logger, os.Stdout)
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
	return withStored(ctx, settings, logger, func(clients *streamapp.Clients, pgStore *store.Store) error {
		deployment := clients.DeploymentApp()
		pinned, err := pgStore.BackfillStreamPins(ctx, deployment)
		for _, table := range slices.Sorted(maps.Keys(pinned)) {
			fmt.Fprintf(out, "%s: %d rows pinned to app %d\n", table, pinned[table], deployment)
		}
		return err
	})
}

// withStored runs a command against app mode's source, with the deployment's own app
// learned first so nothing can be bound to it by mistake.
func withStored(ctx context.Context, settings config.Config, logger *slog.Logger,
	run func(*streamapp.Clients, *store.Store) error) error {
	if settings.Stream.Tenancy != config.TenancyApp {
		return fmt.Errorf("stream-apps looks after customers' own apps, which only stream.tenancy=%s has", config.TenancyApp)
	}
	pgStore, err := openStore(ctx, settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()
	secrets, err := newSecretSealer(ctx, settings)
	if err != nil {
		return err
	}
	clients, err := newStreamClients(settings, pgStore, secrets, logger)
	if err != nil {
		return err
	}
	learning, cancel := context.WithTimeout(ctx, learnTimeout)
	defer cancel()
	if _, err := clients.LearnDeploymentApp(learning); err != nil {
		return fmt.Errorf("the deployment's own Stream app id could not be read, and stream-apps acts only "+
			"once it is known: %w", err)
	}
	return run(clients, pgStore)
}

// runRequire sets an organization's require_own_stream_app, which only the operator may:
// any app's backend can write its organization's policy, and this is the one setting that
// would let one app keep all its siblings out of the shared app.
func runRequire(ctx context.Context, args []string, settings config.Config, out io.Writer) error {
	flags := flag.NewFlagSet("stream-apps require", flag.ContinueOnError)
	organization := flags.String("org", "", "the organization whose apps it applies to")
	off := flags.Bool("off", false, "let the organization's apps fall back again")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *organization == "" {
		return errors.New(streamAppsUsage)
	}
	pgStore, err := openStore(ctx, settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()

	document, err := pgStore.Policy(ctx, store.ScopeOrganization, *organization)
	if err != nil {
		return err
	}
	required := !*off
	document.RequireOwnStreamApp = &required
	if err := pgStore.SavePolicy(ctx, store.ScopeOrganization, *organization, document); err != nil {
		return err
	}
	if required {
		fmt.Fprintf(out, "every app of organization %s now acts only in a Stream app of its own\n", *organization)
	} else {
		fmt.Fprintf(out, "apps of organization %s may fall back to the deployment's own Stream app again\n", *organization)
	}
	return nil
}

// runLegacy counts what customers other than the deployment's own still have in the
// deployment's app, which is what turning the fallback off would leave read-only.
func runLegacy(ctx context.Context, args []string, settings config.Config, logger *slog.Logger, out io.Writer) error {
	flags := flag.NewFlagSet("stream-apps legacy", flag.ContinueOnError)
	byCustomer := flags.Bool("by-customer", false, "list each customer")
	if err := flags.Parse(args); err != nil {
		return err
	}
	return withStored(ctx, settings, logger, func(clients *streamapp.Clients, pgStore *store.Store) error {
		deployment := clients.DeploymentApp()
		counted, err := pgStore.LegacyStreamPins(ctx, deployment)
		if err != nil {
			return err
		}
		return printLegacy(out, counted, *byCustomer)
	})
}

// printLegacy says how much of each kind is left, and with byCustomer, whose. Without it no
// customer is named.
func printLegacy(out io.Writer, counted []store.LegacyCount, byCustomer bool) error {
	kinds := map[string][2]int64{}
	for _, one := range counted {
		total := kinds[one.Kind]
		kinds[one.Kind] = [2]int64{total[0] + one.Rows, total[1] + 1}
	}
	for _, kind := range []string{"session", "call", "number"} {
		fmt.Fprintf(out, "%s: %d rows of %d customers\n", kind, kinds[kind][0], kinds[kind][1])
	}
	if !byCustomer || len(counted) == 0 {
		return nil
	}
	sorted := slices.Clone(counted)
	slices.SortFunc(sorted, func(a, b store.LegacyCount) int {
		return cmp.Or(strings.Compare(a.Kind, b.Kind), strings.Compare(a.CustomerID, b.CustomerID))
	})
	table := tabwriter.NewWriter(out, 0, 4, 2, ' ', 0)
	fmt.Fprintln(table, "KIND\tCUSTOMER\tROWS")
	for _, one := range sorted {
		fmt.Fprintf(table, "%s\t%s\t%d\n", one.Kind, one.CustomerID, one.Rows)
	}
	return table.Flush()
}
