// Package prices reads price files: the vendors' list prices this repository publishes in
// vendor.json, embedded in every router, and any file of a deployment's own laid over them.
//
// A file seeds a database's price list. Once a database holds an item, editing a file
// never changes it; a release only adds the items no database has yet.
package prices

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"os"
	"slices"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
)

// vendorJSON is the vendors' published list prices and nothing else: no markup and no
// negotiated figure, since this repository is public.
//
//go:embed vendor.json
var vendorJSON []byte

// VendorFile is what vendor.json is called wherever a load or an error names it.
const VendorFile = "vendor.json"

// File is a price list as JSON: vendor.json, or a deployment's own file.
type File struct {
	// MarkupBps is the operator's markup on every entry, in basis points (1500 is 15%).
	// Only the first load of a database reads it; after that a markup changes in the
	// database.
	MarkupBps *int64 `json:"markup_bps,omitempty"`
	// Entries are vendor rates, one per item and When.
	Entries []pricing.Entry `json:"entries"`
}

// Vendor is the vendor list embedded in the binary. Its tests parse it, so a build whose
// file does not parse never ships.
func Vendor() File {
	file, err := Parse(VendorFile, vendorJSON)
	if err != nil {
		panic(err)
	}
	return file
}

// Load reads and checks the price file at path.
func Load(path string) (File, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return File{}, fmt.Errorf("prices: %w", err)
	}
	return Parse(path, data)
}

// Parse reads a price file and checks it builds a price list. name is what an error says
// the file is. A field nobody reads is refused rather than dropped, so a misspelt rate is
// an error and never a price of zero.
func Parse(name string, data []byte) (File, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var file File
	if err := decoder.Decode(&file); err != nil {
		return File{}, fmt.Errorf("prices: %s: %w", name, err)
	}
	if _, err := decoder.Token(); err != io.EOF {
		return File{}, fmt.Errorf("prices: %s: more than one JSON value", name)
	}
	if err := file.check(); err != nil {
		return File{}, fmt.Errorf("prices: %s: %w", name, err)
	}
	return file, nil
}

func (f File) check() error {
	if f.MarkupBps != nil && !pricing.ValidMarkup(*f.MarkupBps) {
		return fmt.Errorf("a markup of %d basis points, outside 0 to %d", *f.MarkupBps, pricing.MaxMarkupBps)
	}
	for _, entry := range f.Entries {
		// A database keeps neither: local says how a deployment runs a model, which is its
		// routing config's to say, and a version is the change that wrote the entry.
		if entry.Local {
			return fmt.Errorf("%s sets local: mark the model local in the routing config instead", Name(entry))
		}
		if entry.Version != "" {
			return fmt.Errorf("%s sets a version, which only the price list's changes have", Name(entry))
		}
	}
	var markup int64
	if f.MarkupBps != nil {
		markup = *f.MarkupBps
	}
	_, err := pricing.NewList("check", f.Entries, markup, slog.New(slog.DiscardHandler))
	return err
}

// Name is the item an entry prices, as logs and errors write it: modality/provider/model,
// and its When when it has one.
func Name(entry pricing.Entry) string {
	name := entry.Modality + "/" + entry.Provider + "/" + entry.Model
	if len(entry.When) > 0 {
		name += " when " + When(entry.When)
	}
	return name
}

// When writes an entry's conditions in one order, so two equal ones are equal strings.
func When(when map[string]string) string {
	pairs := make([]string, 0, len(when))
	for _, attribute := range slices.Sorted(maps.Keys(when)) {
		pairs = append(pairs, attribute+"="+when[attribute])
	}
	return strings.Join(pairs, ",")
}

// Merge lays the files over each other: for one item and When, the first file with an
// entry for it wins, so a deployment's own file goes before vendor.json.
func Merge(files ...File) []pricing.Entry {
	var merged []pricing.Entry
	seen := map[string]bool{}
	for _, file := range files {
		for _, entry := range file.Entries {
			if key := Name(entry); !seen[key] {
				seen[key] = true
				merged = append(merged, entry)
			}
		}
	}
	return merged
}

// WithLocal marks the items a deployment runs itself: each of their entries is local, and an
// item with none gets one with no rate, so its work is free rather than unpriced.
func WithLocal(entries []pricing.Entry, local []pricing.Item) []pricing.Entry {
	return pricing.WithLocal(entries, local)
}

// EmbeddedVersion is the version a list built from the embedded vendor.json names on every
// quote: embedded:<release>, or embedded for a binary that names no release.
func EmbeddedVersion(release string) string {
	if release == "" {
		return "embedded"
	}
	return "embedded:" + release
}

// Embedded is the price list of the vendor list embedded in the binary, at the vendors'
// rates with no markup, with the items the deployment runs itself marked local.
func Embedded(local []pricing.Item, release string, logger *slog.Logger) (*pricing.List, error) {
	return pricing.NewList(EmbeddedVersion(release), WithLocal(Vendor().Entries, local), 0, logger)
}
