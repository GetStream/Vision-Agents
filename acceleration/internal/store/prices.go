package store

import (
	"cmp"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strings"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/prices"
)

// What wrote a price change (PriceChange.Kind).
const (
	// PriceChangeSeed is a database's first load, from the router's price files.
	PriceChangeSeed = "seed"
	// PriceChangeRelease is the items a release's price files had that the database lacked.
	PriceChangeRelease = "release"
	// PriceChangeAPI is a change made through the management API.
	PriceChangeAPI = "api"
)

// ErrNoPriceChange is a price change nobody wrote.
var ErrNoPriceChange = errors.New("store: there is no such price change")

// PriceEntry is one item's price, as a change wrote it. Rows are only ever added: the entry
// in force for an item is the newest whose EffectiveAt has passed under a change that was not
// cancelled, and a newer row replaces an older one whole.
type PriceEntry struct {
	bun.BaseModel `bun:"table:price_entries,alias:pe"`

	ID     int64  `bun:"id,pk,autoincrement"`
	Change string `bun:"change,notnull"`
	// SKU names the item across every change of its price (AssignSKUs).
	SKU string `bun:"sku,notnull"`
	// Modality, Provider and Model are the item; Attributes is the entry's When: the item
	// attributes it applies to, such as {"service_tier": "flex"}. Empty is the default entry.
	Modality   string            `bun:"modality,notnull"`
	Provider   string            `bun:"provider,notnull"`
	Model      string            `bun:"model,notnull"`
	Attributes map[string]string `bun:"attributes,type:jsonb,nullzero"`
	Free       bool              `bun:"free,notnull"`
	// Rates are the vendor's, for reference: nothing is charged from them.
	Rates map[pricing.Meter]pricing.Rate `bun:"rates,type:jsonb,nullzero"`
	// Tiers are the vendor's context tiers: a longer prompt costed at other rates, whole.
	Tiers []pricing.Tier `bun:"tiers,type:jsonb,nullzero"`
	// Prices are what the work is charged, by the meters of Rates; PriceTiers are its context
	// tiers, at the thresholds of Tiers.
	Prices     map[pricing.Meter]pricing.Rate `bun:"prices,type:jsonb,nullzero"`
	PriceTiers []pricing.Tier                 `bun:"price_tiers,type:jsonb,nullzero"`
	// Withdrawn stops pricing the item under these Attributes from EffectiveAt on; its
	// entries under other Attributes are untouched.
	Withdrawn bool `bun:"withdrawn,notnull"`
	// EffectiveAt is stamped from the change by the price_entries_guard trigger, so an insert
	// leaves it zero and no entry can backdate itself.
	EffectiveAt time.Time `bun:"effective_at,nullzero"`
	CreatedAt   time.Time `bun:"created_at,nullzero,notnull,default:current_timestamp"`
}

// priceEntryColumns are PriceEntry's columns, read and returned by name rather than as *, so
// a column a later release adds or drops does not break the read.
const priceEntryColumns = "id, change, sku, modality, provider, model, attributes, free, rates, tiers, prices, " +
	"price_tiers, withdrawn, effective_at, created_at"

// Entry is the row as a price list prices with it, its Version the change that wrote it.
func (e PriceEntry) Entry() pricing.Entry {
	entry := pricing.Entry{
		Modality:   e.Modality,
		Provider:   e.Provider,
		Model:      e.Model,
		SKU:        e.SKU,
		Free:       e.Free,
		Rates:      e.Rates,
		Tiers:      e.Tiers,
		Prices:     e.Prices,
		PriceTiers: e.PriceTiers,
		Version:    e.Change,
	}
	if len(e.Attributes) > 0 {
		entry.When = e.Attributes
	}
	if len(e.Rates) == 0 {
		entry.Rates = nil
	}
	if len(e.Tiers) == 0 {
		entry.Tiers = nil
	}
	if len(e.Prices) == 0 {
		entry.Prices = nil
	}
	if len(e.PriceTiers) == 0 {
		entry.PriceTiers = nil
	}
	return entry
}

// priceEntryOf is the row a change writes for an entry. Local and Version have no column:
// local is the routing config's to say, and the version is the change itself.
func priceEntryOf(change string, entry pricing.Entry) PriceEntry {
	return PriceEntry{
		Change:     change,
		SKU:        entry.SKU,
		Modality:   entry.Modality,
		Provider:   entry.Provider,
		Model:      entry.Model,
		Attributes: entry.When,
		Free:       entry.Free,
		Rates:      entry.Rates,
		Tiers:      entry.Tiers,
		Prices:     entry.Prices,
		PriceTiers: entry.PriceTiers,
	}
}

// PriceChange is one write to the price list. It is never deleted; one not yet in force can
// be cancelled, once.
type PriceChange struct {
	bun.BaseModel `bun:"table:price_changes,alias:pc"`

	// Name is <UTC yyyymmddhhmmss>_<snake_case>.
	Name string `bun:"name,pk"`
	// Kind is PriceChangeSeed, PriceChangeRelease or PriceChangeAPI.
	Kind   string `bun:"kind,notnull"`
	Reason string `bun:"reason,notnull"`
	// Source is the vendor's page and the date it was read, the operator's decision, or the
	// files and release a load came from.
	Source         string     `bun:"source,notnull"`
	EffectiveAt    time.Time  `bun:"effective_at,notnull"`
	CreatedBy      string     `bun:"created_by,notnull"`
	IdempotencyKey string     `bun:"idempotency_key,nullzero"`
	CreatedAt      time.Time  `bun:"created_at,nullzero,notnull,default:current_timestamp"`
	CancelledAt    *time.Time `bun:"cancelled_at"`
	CancelledBy    string     `bun:"cancelled_by,nullzero"`
	CancelReason   string     `bun:"cancel_reason,nullzero"`
}

// priceChangeColumns are PriceChange's columns, returned by name as priceEntryColumns are.
const priceChangeColumns = "name, kind, reason, source, effective_at, created_by, idempotency_key, created_at, " +
	"cancelled_at, cancelled_by, cancel_reason"

// PriceLoad is what a router's start loads into the price list: its price files, merged.
type PriceLoad struct {
	// Entries are every file's, the first file's entry winning for an item and When. One
	// with no price is priced at its vendor rates times pricing.PriceFactorBps as it is
	// written, and one with no SKU is given one (AssignSKUs).
	Entries []pricing.Entry
	// Source names the files, for the change's source.
	Source string
	// Release is the router's, for the change's name and author.
	Release string
}

// pricesLock serialises loads of the price list, so routers starting at once write it once.
// pricesLockTimeout bounds the wait for it: a load runs before a router serves, and one held
// up by a stuck session fails and is tried again later rather than keeping the router down.
const (
	pricesLock        = "SELECT pg_advisory_xact_lock(hashtextextended('router.prices', 0))"
	pricesLockTimeout = "SET LOCAL lock_timeout = '5s'"
)

// PricesToLoad is what SyncPrices would write now, writing nothing: whether it would be the
// database's first load, and the entries it would add.
func (s *Store) PricesToLoad(ctx context.Context, entries []pricing.Entry) (bool, []pricing.Entry, error) {
	return pricesToLoad(ctx, s.db, entries)
}

// SyncPrices loads the router's price files into the price list, in one transaction. A
// database with no change yet gets them all as its seed. Otherwise only the entries for an
// item and When no change that stands has ever priced are added, as one release change: an
// item the database has priced keeps its price and its withdrawal, whatever the files say
// now. It reports the change it wrote, the entries in it as written, and whether it wrote one.
func (s *Store) SyncPrices(ctx context.Context, load PriceLoad) (PriceChange, []pricing.Entry, bool, error) {
	var change PriceChange
	var added []pricing.Entry
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if err := lockPrices(ctx, tx); err != nil {
			return err
		}
		var err error
		change, added, err = syncPrices(ctx, tx, load)
		return err
	})
	if err != nil {
		return PriceChange{}, nil, false, stack.Wrap(fmt.Errorf("store: sync prices: %w", err))
	}
	return change, added, added != nil, nil
}

// lockPrices takes the lock every write to the price list takes, giving up after
// pricesLockTimeout.
func lockPrices(ctx context.Context, tx bun.Tx) error {
	if _, err := tx.ExecContext(ctx, pricesLockTimeout); err != nil {
		return err
	}
	_, err := tx.ExecContext(ctx, pricesLock)
	return err
}

// syncPrices is SyncPrices in a transaction that holds the prices lock. It returns nil
// entries when it wrote nothing.
func syncPrices(ctx context.Context, tx bun.Tx, load PriceLoad) (PriceChange, []pricing.Entry, error) {
	seed, entries, err := pricesToLoad(ctx, tx, load.Entries)
	if err != nil || len(entries) == 0 {
		return PriceChange{}, nil, err
	}
	// The database's clock, so routers whose clocks disagree still name and time their
	// changes in one order.
	var now time.Time
	if err := tx.QueryRowContext(ctx, "SELECT now()").Scan(&now); err != nil {
		return PriceChange{}, nil, err
	}
	change := PriceChange{
		EffectiveAt: now,
		Source:      load.Source + ", router " + load.Release,
		CreatedBy:   "router " + load.Release,
	}
	stamp := now.UTC().Format("20060102150405")
	if seed {
		change.Name, change.Kind, change.Reason = stamp+"_seed", PriceChangeSeed, "The first price list"
	} else {
		change.Name, change.Kind = stamp+"_release_"+ReleaseName(load.Release), PriceChangeRelease
		change.Reason = "Items new in " + load.Release
	}
	history, err := priceSKUs(ctx, tx)
	if err != nil {
		return PriceChange{}, nil, err
	}
	writes := make([]PriceEntryWrite, 0, len(entries))
	for _, entry := range entries {
		writes = append(writes, PriceEntryWrite{Entry: pricing.Priced(entry)})
	}
	if writes, err = AssignSKUs(history, now, writes); err != nil {
		return PriceChange{}, nil, err
	}
	if _, err := tx.NewInsert().Model(&change).Exec(ctx); err != nil {
		return PriceChange{}, nil, err
	}
	rows := make([]PriceEntry, 0, len(writes))
	entries = make([]pricing.Entry, 0, len(writes))
	names := make([]string, 0, len(writes))
	for _, write := range writes {
		rows = append(rows, priceEntryOf(change.Name, write.Entry))
		entries = append(entries, write.Entry)
		names = append(names, prices.Name(write.Entry))
	}
	if _, err := tx.NewInsert().Model(&rows).Exec(ctx); err != nil {
		return PriceChange{}, nil, err
	}
	changes := []AuditChange{{Field: "entries", After: names}}
	err = recordAuditEntry(ctx, tx, &AuditEntry{
		ResourceType: AuditResourcePriceList,
		ResourceID:   change.Name,
		ResourceName: change.Reason,
		Action:       AuditCreated,
		Source:       AuditSourceCLI,
		ActorID:      change.CreatedBy,
		ActorName:    "router",
		Changes:      changes,
	})
	if err != nil {
		return PriceChange{}, nil, err
	}
	return change, entries, nil
}

// PriceReload is what a reload of the price list found in it, and the seed it wrote in its
// place.
type PriceReload struct {
	// At is the database's clock when the reload read the list.
	At time.Time
	// Changes are every change the list held, newest first: cancelled ones and those not yet
	// in force among them. Entries is how many entries they wrote.
	Changes []PriceChange
	Entries int
	// Active is the list in force at At, before the reload.
	Active []PriceEntry
	// Seed is the change the reload wrote and Loaded its entries; both are empty on a dry run.
	Seed   PriceChange
	Loaded []pricing.Entry
}

// ReloadPrices empties the price list and seeds it from load, as a router's first start on a
// fresh database does, in one transaction under the lock every load takes. Every change goes:
// those scheduled and cancelled with the rest, so the list holds what the files say and
// nothing else. TRUNCATE is what removes them, because the tables' triggers
// refuse a DELETE; no table references either one. The audit log keeps its rows, and gains
// one saying what the reload removed. dryRun reads the list under the lock and writes nothing.
//
// It is for a development database only; the caller decides that it is one.
func (s *Store) ReloadPrices(ctx context.Context, load PriceLoad, dryRun bool) (PriceReload, error) {
	if len(load.Entries) == 0 {
		return PriceReload{}, stack.Wrap(errors.New("store: reload prices: the price files have no entry, so the list would be empty"))
	}
	var reload PriceReload
	err := s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if err := lockPrices(ctx, tx); err != nil {
			return err
		}
		if err := tx.QueryRowContext(ctx, "SELECT now()").Scan(&reload.At); err != nil {
			return err
		}
		reload.Changes = []PriceChange{}
		if err := tx.NewSelect().Model(&reload.Changes).Order("created_at DESC", "name DESC").Scan(ctx); err != nil {
			return err
		}
		var err error
		if reload.Entries, err = tx.NewSelect().Model((*PriceEntry)(nil)).Count(ctx); err != nil {
			return err
		}
		if reload.Active, err = activePriceEntries(ctx, tx, reload.At); err != nil {
			return err
		}
		if dryRun {
			return nil
		}
		if _, err := tx.ExecContext(ctx, "TRUNCATE price_entries, price_changes"); err != nil {
			return err
		}
		// The removal is recorded before the seed, under a name of its own, so the log reads
		// in the order the reload did them.
		if len(reload.Changes) > 0 {
			names := make([]string, 0, len(reload.Changes))
			for _, change := range reload.Changes {
				names = append(names, change.Name)
			}
			err := recordAuditEntry(ctx, tx, &AuditEntry{
				ResourceType: AuditResourcePriceList,
				ResourceID:   reload.At.UTC().Format("20060102150405") + "_reload",
				ResourceName: "Every change, removed by a development reload",
				Action:       AuditDeleted,
				Source:       AuditSourceCLI,
				ActorID:      "router " + load.Release,
				ActorName:    "router prices reload",
				Changes: []AuditChange{
					{Field: "changes", Before: names},
					{Field: "entries", Before: reload.Entries},
				},
			})
			if err != nil {
				return err
			}
		}
		reload.Seed, reload.Loaded, err = syncPrices(ctx, tx, load)
		return err
	})
	if err != nil {
		return PriceReload{}, stack.Wrap(fmt.Errorf("store: reload prices: %w", busyOr(err)))
	}
	return reload, nil
}

// pricesToLoad is whether the database has no change yet, and the entries a load would add:
// all of them to an empty one, else those for an item and When no change that stands wrote.
func pricesToLoad(ctx context.Context, db bun.IDB, entries []pricing.Entry) (bool, []pricing.Entry, error) {
	var changes int
	if err := db.NewSelect().Model((*PriceChange)(nil)).ColumnExpr("count(*)").Scan(ctx, &changes); err != nil {
		return false, nil, stack.Wrap(fmt.Errorf("store: prices to load: %w", err))
	}
	if changes == 0 {
		return true, slices.Clone(entries), nil
	}
	var priced []PriceEntry
	err := db.NewSelect().Model(&priced).
		ColumnExpr("pe.modality, pe.provider, pe.model, pe.attributes").
		Join("JOIN price_changes AS pc ON pc.name = pe.change AND pc.cancelled_at IS NULL").
		Distinct().
		Scan(ctx)
	if err != nil {
		return false, nil, stack.Wrap(fmt.Errorf("store: prices to load: %w", err))
	}
	known := make(map[string]bool, len(priced))
	for _, row := range priced {
		known[prices.Name(pricing.Entry{Modality: row.Modality, Provider: row.Provider, Model: row.Model, When: row.Attributes})] = true
	}
	var added []pricing.Entry
	for _, entry := range entries {
		if !known[prices.Name(entry)] {
			added = append(added, entry)
		}
	}
	return false, added, nil
}

var notInName = regexp.MustCompile(`[^a-z0-9_]`)

// ReleaseName is a release as a change's name carries it: lowercased, with every character
// outside [a-z0-9_] an underscore, so v1.4.0 is v1_4_0.
func ReleaseName(release string) string {
	return notInName.ReplaceAllString(strings.ToLower(release), "_")
}

// ActivePriceEntries is the entry in force at at for every item and When: the newest that has
// taken effect under a change not cancelled, ties going to the later change. An item whose
// newest entry withdrew it is left out.
func (s *Store) ActivePriceEntries(ctx context.Context, at time.Time) ([]PriceEntry, error) {
	return activePriceEntries(ctx, s.db, at)
}

func activePriceEntries(ctx context.Context, db bun.IDB, at time.Time) ([]PriceEntry, error) {
	var newest []PriceEntry
	err := db.NewRaw(`
SELECT DISTINCT ON (pe.modality, pe.provider, pe.model, pe.attributes) `+qualified("pe", priceEntryColumns)+`
FROM price_entries AS pe
JOIN price_changes AS pc ON pc.name = pe.change AND pc.cancelled_at IS NULL
WHERE pe.effective_at <= ?
ORDER BY pe.modality, pe.provider, pe.model, pe.attributes, pe.effective_at DESC, pe.change DESC, pe.id DESC`, at).
		Scan(ctx, &newest)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: active price entries: %w", err))
	}
	var active []PriceEntry
	for _, entry := range newest {
		if !entry.Withdrawn {
			active = append(active, entry)
		}
	}
	return active, nil
}

// qualified is a list of columns each prefixed with the table's alias.
func qualified(alias, columns string) string {
	names := strings.Split(columns, ", ")
	for i, name := range names {
		names[i] = alias + "." + name
	}
	return strings.Join(names, ", ")
}

// NextPriceChangeAt is when the next change not cancelled takes effect after after, and
// whether one is scheduled, so a list in memory can switch at that moment.
func (s *Store) NextPriceChangeAt(ctx context.Context, after time.Time) (time.Time, bool, error) {
	return nextPriceChangeAt(ctx, s.db, after)
}

func nextPriceChangeAt(ctx context.Context, db bun.IDB, after time.Time) (time.Time, bool, error) {
	var next sql.NullTime
	err := db.NewSelect().Model((*PriceChange)(nil)).ColumnExpr("min(effective_at)").
		Where("cancelled_at IS NULL").
		Where("effective_at > ?", after).
		Scan(ctx, &next)
	if err != nil {
		return time.Time{}, false, stack.Wrap(fmt.Errorf("store: next price change: %w", err))
	}
	return next.Time, next.Valid, nil
}

// UpcomingPriceEntry is an entry a change not yet in force will set, with what the change
// says about it.
type UpcomingPriceEntry struct {
	PriceEntry
	Reason string
}

// UpcomingPriceEntries is every entry of a change not cancelled that takes effect after after,
// soonest first.
func (s *Store) UpcomingPriceEntries(ctx context.Context, after time.Time) ([]UpcomingPriceEntry, error) {
	return upcomingPriceEntries(ctx, s.db, after)
}

func upcomingPriceEntries(ctx context.Context, db bun.IDB, after time.Time) ([]UpcomingPriceEntry, error) {
	var changes []PriceChange
	err := db.NewSelect().Model(&changes).
		Where("cancelled_at IS NULL").
		Where("effective_at > ?", after).
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: upcoming price entries: %w", err))
	}
	if len(changes) == 0 {
		return nil, nil
	}
	reasons := make(map[string]string, len(changes))
	for _, change := range changes {
		reasons[change.Name] = change.Reason
	}
	var entries []PriceEntry
	err = db.NewSelect().Model(&entries).
		Where("change IN (?)", bun.In(slices.Collect(maps.Keys(reasons)))).
		Order("effective_at", "change", "id").
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: upcoming price entries: %w", err))
	}
	upcoming := make([]UpcomingPriceEntry, 0, len(entries))
	for _, entry := range entries {
		upcoming = append(upcoming, UpcomingPriceEntry{PriceEntry: entry, Reason: reasons[entry.Change]})
	}
	return upcoming, nil
}

// PriceChange is the change of that name and the entries it wrote, or ErrNoPriceChange.
func (s *Store) PriceChange(ctx context.Context, name string) (PriceChange, []PriceEntry, error) {
	var change PriceChange
	err := s.db.NewSelect().Model(&change).Where("name = ?", name).Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return PriceChange{}, nil, ErrNoPriceChange
	}
	if err != nil {
		return PriceChange{}, nil, stack.Wrap(fmt.Errorf("store: price change: %w", err))
	}
	entries, err := priceEntriesOf(ctx, s.db, name)
	return change, entries, err
}

// priceEntriesOf is the entries a change wrote, in the order it wrote them.
func priceEntriesOf(ctx context.Context, db bun.IDB, name string) ([]PriceEntry, error) {
	entries := []PriceEntry{}
	if err := db.NewSelect().Model(&entries).Where("change = ?", name).Order("id").Scan(ctx); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: price change: %w", err))
	}
	return entries, nil
}

// VendorCostAttribute is the row attribute a vendor's own figure for what the work cost is
// kept in, in micros, by the providers whose answers carry one.
const VendorCostAttribute = "vendor_cost_micros"

// VendorCostModel is, for one model, how far the cost the price list gave each row is from what
// the vendor said it charged: each row's gap as a fraction of the vendor's figure, negative
// where the list billed less, and the rows with the widest gaps.
type VendorCostModel struct {
	Modality string
	Provider string
	Model    string
	Gaps     []float64
	Widest   []VendorCostRow
}

// VendorCostRow is one row with the vendor's figure beside the list's cost at the vendor's
// rates. UsageID is empty on a database whose requests have no usage id.
type VendorCostRow struct {
	ID               int64     `json:"id"`
	UsageID          string    `json:"usage_id,omitempty"`
	StartedAt        time.Time `json:"started_at"`
	VendorCostMicros int64     `json:"vendor_cost_micros"`
	BaseCostMicros   int64     `json:"base_cost_micros"`
	Gap              float64   `json:"gap"`
}

// vendorCosts is every row since a time whose vendor reported what it charged, with the list's
// cost at the vendor's rates: base_cost_micros once something writes it, else cost_micros,
// which is the same figure for a row priced before either was written. A malformed figure is skipped,
// never cast, so one bad row cannot fail the read. %s is the usage id's expression.
const vendorCosts = `
SELECT id, %s, modality, provider, model, started_at, vendor, base
FROM (
    SELECT id, %s, modality, provider, model, started_at,
           CASE WHEN attributes->>'vendor_cost_micros' ~ '^[0-9]{1,18}$'
                THEN (attributes->>'vendor_cost_micros')::bigint END AS vendor,
           CASE WHEN price_version = '' THEN cost_micros ELSE base_cost_micros END AS base
    FROM requests
    WHERE started_at >= ? AND attributes->>'vendor_cost_micros' IS NOT NULL
) AS reported
WHERE vendor > 0`

// VendorCostGaps reads, over every shard, the rows since a time whose vendor reported what it
// charged, per model, with up to widest of the rows whose gap is widest. Each shard is read
// once, in a read-only transaction that gives up after timeout: no index leads with
// started_at, so the read scans, and a large shard must not hold a connection for long.
func (s *Store) VendorCostGaps(ctx context.Context, since time.Time, widest int, timeout time.Duration) ([]VendorCostModel, error) {
	shards, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]VendorCostModel, error) {
		return vendorCostGaps(ctx, db, since, widest, timeout)
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: vendor cost gaps: %w", err))
	}
	byModel := map[string]*VendorCostModel{}
	var order []string
	for _, models := range shards {
		for _, model := range models {
			key := model.Modality + "/" + model.Provider + "/" + model.Model
			found, ok := byModel[key]
			if !ok {
				found = &VendorCostModel{Modality: model.Modality, Provider: model.Provider, Model: model.Model}
				byModel[key] = found
				order = append(order, key)
			}
			found.Gaps = append(found.Gaps, model.Gaps...)
			found.Widest = append(found.Widest, model.Widest...)
		}
	}
	slices.Sort(order)
	merged := make([]VendorCostModel, 0, len(order))
	for _, key := range order {
		model := byModel[key]
		model.Widest = widestOf(model.Widest, widest)
		merged = append(merged, *model)
	}
	return merged, nil
}

// widestOf is the rows with the widest gaps, widest first, at most n of them; ties go to the
// lower id, so the same rows are picked every time.
func widestOf(rows []VendorCostRow, n int) []VendorCostRow {
	slices.SortFunc(rows, func(a, b VendorCostRow) int {
		if gapA, gapB := max(a.Gap, -a.Gap), max(b.Gap, -b.Gap); gapA != gapB {
			if gapA > gapB {
				return -1
			}
			return 1
		}
		return cmp.Compare(a.ID, b.ID)
	})
	if len(rows) > n {
		rows = rows[:n:n]
	}
	return rows
}

// vendorCostGaps is VendorCostGaps on one shard's database: one pass over its rows, the gaps
// and the widest rows kept per model as they stream by.
func vendorCostGaps(ctx context.Context, db bun.IDB, since time.Time, widest int, timeout time.Duration) ([]VendorCostModel, error) {
	var models []VendorCostModel
	err := db.RunInTx(ctx, &sql.TxOptions{ReadOnly: true}, func(ctx context.Context, tx bun.Tx) error {
		// At least a millisecond: 0 would turn the timeout off.
		if _, err := tx.ExecContext(ctx, fmt.Sprintf("SET LOCAL statement_timeout = %d", max(timeout.Milliseconds(), 1))); err != nil {
			return err
		}
		usageID := "''"
		var hasUsageID bool
		if err := tx.QueryRowContext(ctx, `SELECT EXISTS (SELECT 1 FROM information_schema.columns
WHERE table_schema = current_schema() AND table_name = 'requests' AND column_name = 'usage_id')`).Scan(&hasUsageID); err != nil {
			return err
		}
		if hasUsageID {
			usageID = "coalesce(usage_id::text, '')"
		}
		rows, err := tx.QueryContext(ctx, fmt.Sprintf(vendorCosts, usageID+" AS usage_id", usageID+" AS usage_id"), since)
		if err != nil {
			return err
		}
		defer rows.Close()
		byModel := map[string]int{}
		for rows.Next() {
			var modality, provider, model string
			var row VendorCostRow
			if err := rows.Scan(&row.ID, &row.UsageID, &modality, &provider, &model, &row.StartedAt, &row.VendorCostMicros, &row.BaseCostMicros); err != nil {
				return err
			}
			row.Gap = float64(row.BaseCostMicros-row.VendorCostMicros) / float64(row.VendorCostMicros)
			key := modality + "/" + provider + "/" + model
			i, ok := byModel[key]
			if !ok {
				i = len(models)
				byModel[key] = i
				models = append(models, VendorCostModel{Modality: modality, Provider: provider, Model: model})
			}
			found := &models[i]
			found.Gaps = append(found.Gaps, row.Gap)
			found.Widest = append(found.Widest, row)
			// Pruned as it grows, so a model with a million rows keeps a few dozen of them.
			if len(found.Widest) >= 4*widest {
				found.Widest = widestOf(found.Widest, widest)
			}
		}
		return rows.Err()
	})
	if err != nil {
		return nil, err
	}
	for i := range models {
		models[i].Widest = widestOf(models[i].Widest, widest)
	}
	return models, nil
}

// How many price changes are handed back at once. Neither is measured; they are the sizes
// every other list here uses.
const (
	defaultPriceChangeLimit = 25
	maxPriceChangeLimit     = 200
)

// PriceChangeLimit is the page size a list of price changes uses for the limit asked for.
// PriceChanges returns one row more than this.
func PriceChangeLimit(asked int) int {
	return clampLimit(asked, defaultPriceChangeLimit, maxPriceChangeLimit)
}

// PriceChangePosition is the last change of a page.
type PriceChangePosition struct {
	CreatedAt time.Time `json:"c"`
	Name      string    `json:"n"`
}

// PriceChangeFilter picks price changes, a page at a time.
type PriceChangeFilter struct {
	// Kind keeps only changes of that kind. Empty is every kind.
	Kind  string
	Limit int
	// After is the last change of the previous page.
	After *PriceChangePosition
}

// PriceChanges lists the price changes, newest first, one more than
// PriceChangeLimit(filter.Limit).
func (s *Store) PriceChanges(ctx context.Context, filter PriceChangeFilter) ([]PriceChange, error) {
	changes := []PriceChange{}
	query := s.db.NewSelect().Model(&changes)
	if filter.Kind != "" {
		query = query.Where("kind = ?", filter.Kind)
	}
	if after := filter.After; after != nil {
		query = query.Where("(created_at, name) < (?, ?)", after.CreatedAt, after.Name)
	}
	err := query.
		Order("created_at DESC", "name DESC").
		Limit(PriceChangeLimit(filter.Limit) + 1).
		Scan(ctx)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: price changes: %w", err))
	}
	return changes, nil
}
