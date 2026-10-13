package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"slices"

	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

// City is a place numbers are bought in.
type City struct {
	Name string
	// State is the US state, for vendors that search by it.
	State string
	// AreaCodes are the city's own, most established first. A search tries them in order.
	AreaCodes []string
}

// Cities are the ten most populous in the US, by the Census Bureau's 2024 estimates.
var Cities = []City{
	{Name: "New York", State: "NY", AreaCodes: []string{"212", "718", "646", "347", "917", "929", "332"}},
	{Name: "Los Angeles", State: "CA", AreaCodes: []string{"213", "323", "310", "818"}},
	{Name: "Chicago", State: "IL", AreaCodes: []string{"312", "773", "872"}},
	{Name: "Houston", State: "TX", AreaCodes: []string{"713", "281", "832", "346"}},
	{Name: "Phoenix", State: "AZ", AreaCodes: []string{"602", "480", "623"}},
	{Name: "Philadelphia", State: "PA", AreaCodes: []string{"215", "267", "445"}},
	{Name: "San Antonio", State: "TX", AreaCodes: []string{"210", "726"}},
	{Name: "San Diego", State: "CA", AreaCodes: []string{"619", "858"}},
	{Name: "Dallas", State: "TX", AreaCodes: []string{"214", "469", "972", "945"}},
	{Name: "Jacksonville", State: "FL", AreaCodes: []string{"904"}},
}

// Line is one number the bench owns.
type Line struct {
	Vendor string `json:"vendor"`
	City   string `json:"city"`
	E164   string `json:"e164"`
}

// Lines is the bench's numbers, kept in a file between provisioning, runs and release.
type Lines []Line

// LoadLines reads the numbers file. A file that does not exist yet is no numbers.
func LoadLines(path string) (Lines, error) {
	raw, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	var lines Lines
	if err := json.Unmarshal(raw, &lines); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return lines, nil
}

// Save writes the numbers file.
func (l Lines) Save(path string) error {
	raw, err := json.MarshalIndent(l, "", "  ")
	if err != nil {
		return fmt.Errorf("encode numbers: %w", err)
	}
	if err := os.WriteFile(path, append(raw, '\n'), 0o644); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}

// In is how many numbers a vendor has in a city.
func (l Lines) In(vendor, city string) int {
	count := 0
	for _, line := range l {
		if line.Vendor == vendor && line.City == city {
			count++
		}
	}
	return count
}

// Only keeps the numbers of the named vendors and cities. An empty list keeps them all.
func (l Lines) Only(vendors, cities []string) Lines {
	var kept Lines
	for _, line := range l {
		if (len(vendors) == 0 || slices.Contains(vendors, line.Vendor)) &&
			(len(cities) == 0 || slices.Contains(cities, line.City)) {
			kept = append(kept, line)
		}
	}
	return kept
}

// Provision buys numbers until every vendor has perCity of them in every city, saving the
// file after each purchase so a failure halfway does not lose track of what was bought.
func Provision(ctx context.Context, providers map[string]phone.Provider, cities []City, perCity int, path string, log *slog.Logger) (Lines, error) {
	lines, err := LoadLines(path)
	if err != nil {
		return nil, err
	}
	for _, vendor := range sortedKeys(providers) {
		provider := providers[vendor]
		for _, city := range cities {
			for lines.In(vendor, city.Name) < perCity {
				offer, err := findNumber(ctx, provider, city, lines)
				if err != nil {
					return lines, fmt.Errorf("%s in %s: %w", vendor, city.Name, err)
				}
				bought, err := provider.BuyNumber(ctx, phone.Order{E164: offer.E164, Country: "US"})
				if err != nil {
					return lines, fmt.Errorf("buy %s from %s: %w", offer.E164, vendor, err)
				}
				lines = append(lines, Line{Vendor: vendor, City: city.Name, E164: bought.E164})
				if err := lines.Save(path); err != nil {
					return lines, err
				}
				log.Info("telephony-latency: bought", "vendor", vendor, "city", city.Name, "number", bought.E164,
					"monthly_usd", float64(bought.MonthlyCostMicros)/1e6)
			}
		}
	}
	return lines, nil
}

// findNumber is a local voice number in the city that the bench does not already own,
// looked for by area code first and by name if none of them has one.
func findNumber(ctx context.Context, provider phone.Provider, city City, owned Lines) (phone.Available, error) {
	searches := make([]phone.Search, 0, len(city.AreaCodes)+1)
	for _, code := range city.AreaCodes {
		searches = append(searches, phone.Search{Country: "US", AreaCode: code})
	}
	searches = append(searches, phone.Search{Country: "US", Locality: city.Name, AdministrativeArea: city.State})

	for _, search := range searches {
		search.Type = phone.Local
		search.Capabilities = []phone.Capability{phone.Voice}
		search.Limit = 10
		if len(search.Unsupported(provider)) > 0 {
			continue
		}
		offers, err := provider.SearchNumbers(ctx, search)
		if err != nil {
			return phone.Available{}, err
		}
		for _, offer := range offers {
			if search.Matches(offer) && !slices.ContainsFunc(owned, func(line Line) bool { return line.E164 == offer.E164 }) {
				return offer, nil
			}
		}
	}
	return phone.Available{}, errors.New("no local voice number for sale")
}

// Release gives every number in the file back and empties it, keeping the ones a vendor
// refused so they can be released again.
func Release(ctx context.Context, providers map[string]phone.Provider, path string, log *slog.Logger) error {
	lines, err := LoadLines(path)
	if err != nil {
		return err
	}
	var kept Lines
	var failed []error
	for _, line := range lines {
		provider, ok := providers[line.Vendor]
		if !ok {
			kept = append(kept, line)
			continue
		}
		if err := provider.ReleaseNumber(ctx, line.E164); err != nil {
			kept = append(kept, line)
			failed = append(failed, fmt.Errorf("release %s at %s: %w", line.E164, line.Vendor, err))
			continue
		}
		log.Info("telephony-latency: released", "vendor", line.Vendor, "number", line.E164)
	}
	if err := kept.Save(path); err != nil {
		failed = append(failed, err)
	}
	return errors.Join(failed...)
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}
