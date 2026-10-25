package ledger

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"

	"github.com/uptrace/bun"
	"github.com/uptrace/bun/driver/pgdriver"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// TierFile is a list of tiers to seed: an operator's tier file. Amounts are micros; a setting
// left out is no limit.
type TierFile struct {
	Tiers []usage.Tier `json:"tiers"`
}

// ReadTierFile reads and checks the tier file at path. A file that does not parse, names a
// tier twice, or holds a tier SetTier would refuse is an error naming the file and the tier.
func ReadTierFile(path string) (*TierFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("ledger: tier file %s: %w", path, err)
	}
	file, err := ParseTierFile(data)
	if err != nil {
		return nil, fmt.Errorf("ledger: tier file %s: %w", path, err)
	}
	return file, nil
}

// ParseTierFile parses and checks a tier file. A field it does not know is an error: a
// setting misspelt would otherwise be no limit.
func ParseTierFile(data []byte) (*TierFile, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var file TierFile
	if err := decoder.Decode(&file); err != nil {
		return nil, err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, errors.New("the file holds more than one JSON value")
	}
	seen := make(map[string]bool, len(file.Tiers))
	for i, tier := range file.Tiers {
		if problem := tier.Problem(); problem != "" {
			return nil, fmt.Errorf("tier %q (tiers[%d]): %s", tier.Name, i, problem)
		}
		if seen[tier.Name] {
			return nil, fmt.Errorf("tier %q is named twice", tier.Name)
		}
		seen[tier.Name] = true
	}
	return &file, nil
}

// seedActor is who a seeded tier is written by: the release that seeded it.
func seedActor(release string) Actor {
	if release == "" {
		return Actor{ID: "usage"}
	}
	return Actor{ID: "usage " + release}
}

// SyncTiers adds the tiers of the file, which may be nil, that have no row yet, and then the
// built-in default tier if there is still no default row, each audited as seeded by the
// release. A tier the database has is never touched: it changes through SetTier alone. Every
// replica syncs as it starts, one at a time. It answers the tiers it added.
func (l *Ledger) SyncTiers(ctx context.Context, file *TierFile, release string) ([]string, error) {
	actor := seedActor(release)
	var tiers []usage.Tier
	if file != nil {
		tiers = file.Tiers
	}
	var added []string
	err := l.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		added = nil
		if err := lockTiers(ctx, tx); err != nil {
			return err
		}
		for _, tier := range slices.Concat(tiers, []usage.Tier{usage.BuiltInDefaultTier()}) {
			row := Tier{Name: tier.Name, FailMode: tier.FailMode, Limits: tier.Limits.Compact(), UpdatedBy: actor.ID}
			var inserted []Tier
			if err := tx.NewInsert().Model(&row).On("CONFLICT (name) DO NOTHING").Returning("*").Scan(ctx, &inserted); err != nil {
				return fmt.Errorf("ledger: sync tiers: %s: %w", tier.Name, err)
			}
			if len(inserted) == 0 {
				continue
			}
			if err := audit(ctx, tx, actor, "", ActionSeedTier, "", nil, inserted[0]); err != nil {
				return err
			}
			added = append(added, tier.Name)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return added, nil
}

// PlanTierSync is what SyncTiers would add now, writing nothing: the file's tiers with no
// row, and the built-in default when neither the file nor the database has one; and whether
// the database has its tiers table yet. One that has not been migrated to it has no tier,
// which is when an operator checks a release before its first start.
func (l *Ledger) PlanTierSync(ctx context.Context, file *TierFile) ([]string, bool, error) {
	existing, err := l.Tiers(ctx)
	var refused pgdriver.Error
	migrated := !(errors.As(err, &refused) && refused.Field('C') == undefinedTable)
	if err != nil && migrated {
		return nil, false, err
	}
	have := make(map[string]bool, len(existing))
	for _, tier := range existing {
		have[tier.Name] = true
	}
	var tiers []usage.Tier
	if file != nil {
		tiers = file.Tiers
	}
	var added []string
	for _, tier := range slices.Concat(tiers, []usage.Tier{usage.BuiltInDefaultTier()}) {
		if !have[tier.Name] {
			have[tier.Name] = true
			added = append(added, tier.Name)
		}
	}
	return added, migrated, nil
}

// undefinedTable is Postgres's SQLSTATE for a relation that does not exist.
const undefinedTable = "42P01"
