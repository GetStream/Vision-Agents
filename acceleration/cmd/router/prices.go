package main

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"math"
	"os"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

const pricesUsage = `usage: router prices <command>

  show [--at time] [--json] [--unpriced]
                             the price list in force: each entry's vendor rates, its markup
                             and what it charges; --unpriced lists the routed items no entry
                             prices. Reads the database
  sync --dry-run             what the next start would load from ROUTER_PRICES_FILE and
                             vendor.json, writing nothing. Reads the database
  check [--file path]        parse a price file (default: the embedded vendor.json) and
                             build a price list from it, with no database
  drift [--since 168h] [--json]
                             per model, how far the cost of rows whose vendor said what it
                             charged is from that figure, over every shard. Reads the database
  export [--from path] [--phone] [--out path] [--strip]
                             write the price: blocks of a routing config as a price file

No command writes a price: a router loads its price files when it starts, and adds only the
items its database has never priced.
`

// priceRetry is how long a router waits to load its price files into the database again
// after a load failed.
const priceRetry = 30 * time.Second

// priceAttempt bounds one load, so the first, made before the router serves, cannot keep it
// from serving: a database that answers slowly fails the attempt, and the retry takes over.
const priceAttempt = 15 * time.Second

// driftThreshold is the median gap past which a model's rows are said to drift from what its
// vendor charged: more than a rounding difference, so a wrong rate or a meter the list misses.
const driftThreshold = 0.01

// driftExamples is how many rows a drifting model is shown with.
const driftExamples = 10

func runPrices(args []string, settings config.Config, logger *slog.Logger) error {
	if len(args) == 0 {
		return errors.New(pricesUsage)
	}
	ctx := context.Background()
	switch args[0] {
	case "show":
		return runPricesShow(ctx, args[1:], settings, os.Stdout)
	case "sync":
		return runPricesSync(ctx, args[1:], settings, os.Stdout)
	case "check":
		return runPricesCheck(args[1:], os.Stdout)
	case "drift":
		return runPricesDrift(ctx, args[1:], settings, os.Stdout)
	case "export":
		return runPricesExport(args[1:], settings, os.Stdout)
	default:
		return fmt.Errorf("unknown prices command %q\n\n%s", args[0], pricesUsage)
	}
}

// releaseName is the release a price load is written under: this binary's, or dev for one
// built without -X main.release.
func releaseName() string {
	return cmp.Or(release, "dev")
}

// pricesVersion names the list built from the price files on every quote: the embedded
// vendor.json's version, or files:<release> when a deployment's own file is laid over it.
func pricesVersion(settings config.Config) string {
	if settings.PricesFile == "" {
		return prices.EmbeddedVersion(releaseName())
	}
	return "files:" + releaseName()
}

// loadPriceFiles reads the price files a router loads into its database: the deployment's own
// first, then the embedded vendor.json, the first winning for an item and When. A file that
// does not parse, or two that do not make a list together, stop the router at start, naming
// the file and the entry.
func loadPriceFiles(settings config.Config) (store.PriceLoad, error) {
	var files []prices.File
	var names []string
	if settings.PricesFile != "" {
		own, err := prices.Load(settings.PricesFile)
		if err != nil {
			return store.PriceLoad{}, err
		}
		files, names = append(files, own), append(names, settings.PricesFile)
	}
	files, names = append(files, prices.Vendor()), append(names, prices.VendorFile)
	merged := prices.Merge(files...)
	if _, err := pricing.NewList("merged", merged, slog.New(slog.DiscardHandler)); err != nil {
		return store.PriceLoad{}, fmt.Errorf("prices: %s: %w", strings.Join(names, " over "), err)
	}
	return store.PriceLoad{Entries: merged, MarkupBps: files[0].MarkupBps, Source: strings.Join(names, ", "), Release: releaseName()}, nil
}

// startPriceSync loads the price files into the database once, and when that fails keeps
// trying every interval in the background until it succeeds or ctx ends. A database that
// cannot be written now is no reason to refuse calls; the router prices from the files it
// embeds meanwhile.
func startPriceSync(ctx context.Context, pgStore *store.Store, load store.PriceLoad, every time.Duration, logger *slog.Logger) {
	if syncPricesOnce(ctx, pgStore, load, every, logger) {
		return
	}
	go func() {
		ticker := time.NewTicker(every)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}
			if syncPricesOnce(ctx, pgStore, load, every, logger) {
				return
			}
		}
	}()
}

// syncPricesOnce loads the price files once and says whether it succeeded.
func syncPricesOnce(ctx context.Context, pgStore *store.Store, load store.PriceLoad, every time.Duration, logger *slog.Logger) bool {
	ctx, cancel := context.WithTimeout(ctx, priceAttempt)
	defer cancel()
	change, added, wrote, err := pgStore.SyncPrices(ctx, load)
	if err != nil {
		logger.Error("could not load the price files into the database; trying again", "in", every, "error", err)
		return false
	}
	if !wrote {
		logger.Debug("the price list already has every item the price files have")
		return true
	}
	items := make([]string, 0, len(added))
	for _, entry := range added {
		items = append(items, prices.Name(entry))
	}
	logger.Info("loaded the price files into the price list", "change", change.Name, "kind", change.Kind, "items", items)
	return true
}

// openPriceStore opens the database to read the price list. It does not migrate: these
// commands run beside a router, often in its pod, and only the router moves the schema.
func openPriceStore(settings config.Config) (*store.Store, error) {
	if settings.Postgres.DSN == "" {
		return nil, errors.New("this needs a database: set postgres.dsn")
	}
	return store.OpenShards(store.Options{
		DSN:      settings.Postgres.DSN,
		ShardDSN: settings.Postgres.ShardDSN,
		Shards:   settings.Postgres.Shards,
	})
}

// shownEntry is an entry of the list in force as show writes it: the vendor's rates, its own
// markup or none, and what customers are charged.
type shownEntry struct {
	pricing.Entry
	Charged       map[pricing.Meter]pricing.Rate `json:"charged,omitempty"`
	ChargedTiers  []pricing.Tier                 `json:"charged_tiers,omitempty"`
	EffectiveAt   time.Time                      `json:"effective_at"`
	ChargedMarkup int64                          `json:"charged_markup_bps"`
}

// shownList is the list in force as show --json writes it.
type shownList struct {
	At        time.Time      `json:"at"`
	MarkupBps int64          `json:"markup_bps"`
	Entries   []shownEntry   `json:"entries"`
	Unpriced  []pricing.Item `json:"unpriced"`
}

func runPricesShow(ctx context.Context, args []string, settings config.Config, out io.Writer) error {
	flags := flag.NewFlagSet("prices show", flag.ContinueOnError)
	atText := flags.String("at", "", "the time to show the list at, RFC 3339; empty is now")
	asJSON := flags.Bool("json", false, "write JSON")
	unpricedOnly := flags.Bool("unpriced", false, "list only the routed items no entry prices")
	if err := flags.Parse(args); err != nil {
		return err
	}
	at := time.Now()
	if *atText != "" {
		parsed, err := time.Parse(time.RFC3339, *atText)
		if err != nil {
			return fmt.Errorf("--at: %w", err)
		}
		at = parsed
	}
	routingConfig, err := routing.LoadConfig(settings.RoutingConfig)
	if err != nil {
		return err
	}
	pgStore, err := openPriceStore(settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()
	rows, err := pgStore.ActivePriceEntries(ctx, at)
	if err != nil {
		return err
	}
	markup, err := pgStore.ActiveMarkup(ctx, at)
	if err != nil {
		return err
	}
	shown := shownList{At: at.UTC(), MarkupBps: markup, Entries: []shownEntry{}, Unpriced: []pricing.Item{}}
	entries := make([]pricing.Entry, 0, len(rows))
	for _, row := range rows {
		entry := row.Entry()
		entries = append(entries, entry)
		charged := pricing.Charged(entry, markup)
		chargedMarkup := markup
		if entry.MarkupBps != nil {
			chargedMarkup = *entry.MarkupBps
		}
		shown.Entries = append(shown.Entries, shownEntry{Entry: entry, Charged: charged.Rates, ChargedTiers: charged.Tiers,
			EffectiveAt: row.EffectiveAt.UTC(), ChargedMarkup: chargedMarkup})
	}
	list, err := pricing.NewList("database", prices.WithLocal(entries, routing.LocalItems(routingConfig)), slog.New(slog.DiscardHandler))
	if err != nil {
		return fmt.Errorf("prices: the list in force does not build: %w", err)
	}
	shown.Unpriced = append(shown.Unpriced, list.Unpriced(routing.Items(routingConfig))...)

	switch {
	case *asJSON:
		encoder := json.NewEncoder(out)
		encoder.SetIndent("", "  ")
		if *unpricedOnly {
			return encoder.Encode(shown.Unpriced)
		}
		return encoder.Encode(shown)
	case *unpricedOnly:
		for _, item := range shown.Unpriced {
			if _, err := fmt.Fprintf(out, "%s/%s/%s\n", item.Modality, item.Provider, item.Model); err != nil {
				return err
			}
		}
		return nil
	}
	if _, err := fmt.Fprintf(out, "at %s, markup %d basis points, %d entries\n\n", shown.At.Format(time.RFC3339), markup, len(shown.Entries)); err != nil {
		return err
	}
	for _, entry := range shown.Entries {
		if _, err := fmt.Fprintf(out, "%s  [%s, markup %d]\n%s", prices.Name(entry.Entry), entry.Version, entry.ChargedMarkup,
			describeRates(entry.Entry, entry.Charged)); err != nil {
			return err
		}
	}
	if len(shown.Unpriced) > 0 {
		if _, err := fmt.Fprintf(out, "\n%d routed items have no price: router prices show --unpriced\n", len(shown.Unpriced)); err != nil {
			return err
		}
	}
	return nil
}

// describeRates writes an entry's rates, one meter a line, the vendor's beside the charged.
func describeRates(entry pricing.Entry, charged map[pricing.Meter]pricing.Rate) string {
	var b strings.Builder
	if entry.Free {
		b.WriteString("    free\n")
	}
	for _, meter := range slices.Sorted(maps.Keys(entry.Rates)) {
		rate := entry.Rates[meter]
		fmt.Fprintf(&b, "    %-26s $%s per %d, charged $%s\n", meter, strconv.FormatFloat(rate.USD, 'g', 12, 64), rate.Per,
			strconv.FormatFloat(charged[meter].USD, 'g', 12, 64))
	}
	for _, tier := range entry.Tiers {
		fmt.Fprintf(&b, "    above %d input tokens: %d rates\n", tier.AboveInputTokens, len(tier.Rates))
	}
	return b.String()
}

func runPricesSync(ctx context.Context, args []string, settings config.Config, out io.Writer) error {
	flags := flag.NewFlagSet("prices sync", flag.ContinueOnError)
	dryRun := flags.Bool("dry-run", false, "say what the next start would load, writing nothing")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if !*dryRun {
		return errors.New("router prices sync only takes --dry-run: a router loads its price files when it starts")
	}
	load, err := loadPriceFiles(settings)
	if err != nil {
		return err
	}
	pgStore, err := openPriceStore(settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()
	seed, added, err := pgStore.PricesToLoad(ctx, load.Entries)
	if err != nil {
		return err
	}
	switch {
	case len(added) == 0:
		_, err = fmt.Fprintf(out, "nothing to add: the price list has every item in %s\n", load.Source)
		return err
	case seed:
		markup := "no markup"
		if load.MarkupBps != nil {
			markup = fmt.Sprintf("a markup of %d basis points", *load.MarkupBps)
		}
		_, err = fmt.Fprintf(out, "the next start seeds the price list from %s: %d entries, %s\n", load.Source, len(added), markup)
	default:
		_, err = fmt.Fprintf(out, "the next start adds %d items new in %s, from %s:\n", len(added), load.Release, load.Source)
	}
	if err != nil {
		return err
	}
	for _, entry := range added {
		if _, err := fmt.Fprintf(out, "  %s\n", prices.Name(entry)); err != nil {
			return err
		}
	}
	return nil
}

// driftedModel is one model's rows against what its vendor said it charged.
type driftedModel struct {
	Modality string `json:"modality"`
	Provider string `json:"provider"`
	Model    string `json:"model"`
	Rows     int    `json:"rows"`
	// MedianGap is the median of each row's (list cost - vendor's figure) / vendor's figure:
	// negative where the list billed less than the vendor charged.
	MedianGap float64 `json:"median_gap"`
	// Status is drift past driftThreshold either way, else ok.
	Status   string                `json:"status"`
	Examples []store.VendorCostRow `json:"examples"`
}

func runPricesDrift(ctx context.Context, args []string, settings config.Config, out io.Writer) error {
	flags := flag.NewFlagSet("prices drift", flag.ContinueOnError)
	since := flags.Duration("since", 168*time.Hour, "how far back to read rows")
	asJSON := flags.Bool("json", false, "write JSON")
	timeout := flags.Duration("timeout", time.Minute, "how long each shard's read may run before it is given up")
	if err := flags.Parse(args); err != nil {
		return err
	}
	pgStore, err := openPriceStore(settings)
	if err != nil {
		return err
	}
	defer pgStore.Close()
	models, err := pgStore.VendorCostGaps(ctx, time.Now().Add(-*since), driftExamples, *timeout)
	if err != nil {
		return err
	}
	drifted := make([]driftedModel, 0, len(models))
	for _, model := range models {
		median := medianOf(model.Gaps)
		status := "ok"
		if math.Abs(median) > driftThreshold {
			status = "drift"
		}
		drifted = append(drifted, driftedModel{Modality: model.Modality, Provider: model.Provider, Model: model.Model,
			Rows: len(model.Gaps), MedianGap: median, Status: status, Examples: model.Widest})
	}
	if *asJSON {
		encoder := json.NewEncoder(out)
		encoder.SetIndent("", "  ")
		return encoder.Encode(drifted)
	}
	if len(drifted) == 0 {
		_, err := fmt.Fprintf(out, "no row since %s says what its vendor charged\n", since)
		return err
	}
	for _, model := range drifted {
		if _, err := fmt.Fprintf(out, "%-5s  %s/%s/%s  %d rows, median gap %+.2f%%\n", model.Status, model.Modality, model.Provider,
			model.Model, model.Rows, model.MedianGap*100); err != nil {
			return err
		}
		if model.Status != "drift" {
			continue
		}
		for _, row := range model.Examples {
			name := fmt.Sprintf("row %d", row.ID)
			if row.UsageID != "" {
				name += " (usage " + row.UsageID + ")"
			}
			if _, err := fmt.Fprintf(out, "         %s at %s: vendor %d, list %d µ$\n", name, row.StartedAt.UTC().Format(time.RFC3339),
				row.VendorCostMicros, row.BaseCostMicros); err != nil {
				return err
			}
		}
	}
	return nil
}

func medianOf(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := slices.Sorted(slices.Values(values))
	middle := len(sorted) / 2
	if len(sorted)%2 == 1 {
		return sorted[middle]
	}
	return (sorted[middle-1] + sorted[middle]) / 2
}

// runPricesCheck parses a price file and builds a list from it, so a malformed file fails in
// CI rather than at a router's start.
func runPricesCheck(args []string, out io.Writer) error {
	flags := flag.NewFlagSet("prices check", flag.ContinueOnError)
	path := flags.String("file", "", "the price file to check; empty checks the embedded vendor.json")
	if err := flags.Parse(args); err != nil {
		return err
	}
	file := prices.Vendor()
	name := prices.VendorFile
	if *path != "" {
		var err error
		if file, err = prices.Load(*path); err != nil {
			return err
		}
		name = *path
	}
	markup := "no markup"
	if file.MarkupBps != nil {
		markup = fmt.Sprintf("a markup of %d basis points", *file.MarkupBps)
	}
	_, err := fmt.Fprintf(out, "%s: %d entries, %s\n", name, len(file.Entries), markup)
	return err
}

// runPricesExport writes the price: blocks of a routing config as a price file. It is how
// vendor.json is made from a router.yaml that still has them, and how a deployment's own
// routing config moves its prices into a file of its own.
func runPricesExport(args []string, settings config.Config, stdout io.Writer) error {
	flags := flag.NewFlagSet("prices export", flag.ContinueOnError)
	from := flags.String("from", settings.RoutingConfig, "the routing config to read; empty reads the embedded one")
	withPhone := flags.Bool("phone", false, "add the phone vendors' free calls, transfers and numbers, as vendor.json has them")
	outPath := flags.String("out", "", "where to write the price file; empty writes it to standard output")
	strip := flags.Bool("strip", false, "then rewrite the routing config at --from without its price: blocks")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *strip && *from == "" {
		return errors.New("--strip rewrites the routing config at --from, and the embedded one is not a file")
	}
	routingConfig, err := routing.LoadConfig(*from)
	if err != nil {
		return err
	}
	var phoneConfig *phone.Config
	if *withPhone {
		loaded, err := phone.LoadConfig(settings.PhoneConfig)
		if err != nil {
			return err
		}
		phoneConfig = &loaded
	}
	file := exportPrices(routingConfig, phoneConfig)
	if len(file.Entries) == 0 {
		return errors.New("the routing config prices nothing: it has no price: block")
	}
	encoded, err := encodePrices(file)
	if err != nil {
		return err
	}
	if _, err := prices.Parse("the exported file", encoded); err != nil {
		return err
	}
	if *outPath == "" {
		if _, err := stdout.Write(encoded); err != nil {
			return err
		}
	} else if err := os.WriteFile(*outPath, encoded, 0o644); err != nil {
		return err
	}
	if *strip {
		return stripPrices(*from)
	}
	return nil
}

// exportPrices is the price list a routing config's price: blocks declare, as a price file.
// Local is left out, since it is the routing config's to say, and so is an entry with
// neither a rate nor free, which says nothing a price file can hold.
func exportPrices(routingConfig routing.Config, phoneConfig *phone.Config) prices.File {
	var entries []pricing.Entry
	for _, entry := range routing.PriceEntries(routingConfig) {
		entry.Local = false
		if !entry.Free && len(entry.Rates) == 0 && len(entry.Tiers) == 0 {
			continue
		}
		entry.Rates = roundedRates(entry.Rates)
		for i := range entry.Tiers {
			entry.Tiers[i].Rates = roundedRates(entry.Tiers[i].Rates)
		}
		entries = append(entries, entry)
	}
	if phoneConfig != nil {
		entries = append(entries, phone.PriceEntries(*phoneConfig)...)
	}
	return prices.File{Entries: entries}
}

// pricedProviders names the models whose routing config still sets a price.
func pricedProviders(routingConfig routing.Config) []string {
	var names []string
	for _, entry := range exportPrices(routingConfig, nil).Entries {
		if len(entry.When) == 0 {
			names = append(names, prices.Name(entry))
		}
	}
	return names
}

// roundedRates keeps twelve significant digits of each rate, so a rate a tier multiplied
// reads as the vendor publishes it (0.065, never 0.06500000000000002). Twelve digits is
// far below a millionth of a dollar on any usage a row records.
func roundedRates(rates map[pricing.Meter]pricing.Rate) map[pricing.Meter]pricing.Rate {
	if rates == nil {
		return nil
	}
	rounded := make(map[pricing.Meter]pricing.Rate, len(rates))
	for meter, rate := range rates {
		rate.USD, _ = strconv.ParseFloat(strconv.FormatFloat(rate.USD, 'g', 12, 64), 64)
		rounded[meter] = rate
	}
	return rounded
}

// encodePrices writes a price file with one entry a line, so a diff names the item whose
// price moved.
func encodePrices(file prices.File) ([]byte, error) {
	var b bytes.Buffer
	b.WriteString("{\n")
	if file.MarkupBps != nil {
		fmt.Fprintf(&b, "  \"markup_bps\": %d,\n", *file.MarkupBps)
	}
	b.WriteString("  \"entries\": [\n")
	for i, entry := range file.Entries {
		var line bytes.Buffer
		encoder := json.NewEncoder(&line)
		encoder.SetEscapeHTML(false)
		if err := encoder.Encode(entry); err != nil {
			return nil, err
		}
		b.WriteString("    ")
		b.Write(bytes.TrimRight(line.Bytes(), "\n"))
		if i < len(file.Entries)-1 {
			b.WriteString(",")
		}
		b.WriteString("\n")
	}
	b.WriteString("  ]\n}\n")
	return b.Bytes(), nil
}

// priceKey is a provider's price: key, the line its block starts on.
var priceKey = regexp.MustCompile(`^( *)price:(\s|$)`)

// stripPrices rewrites a routing config without its price: blocks, every other line as it
// was. It refuses to write a config that reads differently from the one it read in anything
// but the prices.
func stripPrices(path string) error {
	text, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	before, err := routing.LoadConfig(path)
	if err != nil {
		return err
	}
	// Written beside it and renamed over it, so a check that fails leaves the config as it was.
	draft := path + ".stripped"
	if err := os.WriteFile(draft, withoutPriceBlocks(text), 0o644); err != nil {
		return err
	}
	after, err := routing.LoadConfig(draft)
	if err != nil {
		os.Remove(draft) //nolint:errcheck // the error that matters is the one returned
		return fmt.Errorf("prices: %s without its price: blocks does not load: %w", path, err)
	}
	for _, section := range before {
		for i := range section.Providers {
			section.Providers[i].Price = routing.Price{}
		}
	}
	if !reflect.DeepEqual(before, after) {
		os.Remove(draft) //nolint:errcheck // the error that matters is the one returned
		return fmt.Errorf("prices: %s without its price: blocks reads differently in more than its prices", path)
	}
	return os.Rename(draft, path)
}

// withoutPriceBlocks is the YAML with every price: key and the lines indented under it taken
// out. A blank line inside a block goes with it; one after it stays.
func withoutPriceBlocks(text []byte) []byte {
	lines := strings.SplitAfter(string(text), "\n")
	var kept strings.Builder
	for i := 0; i < len(lines); {
		match := priceKey.FindStringSubmatch(lines[i])
		if match == nil {
			kept.WriteString(lines[i])
			i++
			continue
		}
		indent := len(match[1])
		i++
		for i < len(lines) {
			next := i
			for next < len(lines) && strings.TrimSpace(lines[next]) == "" {
				next++
			}
			if next == len(lines) || indentOf(lines[next]) <= indent {
				break
			}
			i = next + 1
		}
	}
	return []byte(kept.String())
}

func indentOf(line string) int {
	return len(line) - len(strings.TrimLeft(line, " "))
}
