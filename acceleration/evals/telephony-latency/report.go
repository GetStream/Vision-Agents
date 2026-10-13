package main

import (
	"fmt"
	"math"
	"net/netip"
	"slices"
	"strings"
)

// Direction is which way a stamp travelled.
type Direction int

const (
	// Forward is from the caller's side to the callee's: what the person called hears.
	Forward Direction = iota
	// Backward is from the callee's side to the caller's: what the person calling hears.
	Backward
)

// Cell is the calls from one vendor to another.
type Cell struct {
	Caller, Callee string
}

// Stats summarise the calls of a cell, one value per call: the median of its stamps, so a
// call with one stamp delayed by a packet burst counts once rather than skewing the cell.
type Stats struct {
	Median, P95 float64
	Calls       int
	Failed      int
}

// String is the stats as a matrix cell.
func (s Stats) String() string {
	if s.Calls == 0 {
		if s.Failed > 0 {
			return fmt.Sprintf("failed ×%d", s.Failed)
		}
		return "–"
	}
	text := fmt.Sprintf("%.0f ms (p95 %.0f, n=%d)", s.Median, s.P95, s.Calls)
	if s.Failed > 0 {
		text += fmt.Sprintf(", %d failed", s.Failed)
	}
	return text
}

// Summarise is a cell's stats in one direction.
func Summarise(results []Result, direction Direction) Stats {
	var perCall []float64
	failed := 0
	for _, result := range results {
		stamps := result.ForwardMs
		if direction == Backward {
			stamps = result.BackwardMs
		}
		if len(stamps) == 0 {
			failed++
			continue
		}
		perCall = append(perCall, percentile(stamps, 50))
	}
	if len(perCall) == 0 {
		return Stats{Failed: failed}
	}
	return Stats{Median: percentile(perCall, 50), P95: percentile(perCall, 95), Calls: len(perCall), Failed: failed}
}

// Shares are a matrix split into what the calling vendor and the called vendor each add.
//
// Every cell is fitted as Baseline + Caller[a] + Callee[b]. Only the differences between
// vendors in the same role can be told apart: a millisecond on every call could belong to
// either side, so it stays in the baseline, and each role's shares average to zero.
type Shares struct {
	Baseline float64
	Caller   map[string]float64
	Callee   map[string]float64
	// Residual is the root mean square of what the fit leaves unexplained. Large next to the
	// shares means a pair of vendors behaves in a way neither does alone, e.g. a slow
	// interconnect between those two only.
	Residual float64
	// Cells is how many cells the fit used.
	Cells int
}

// Split fits the shares to a matrix of cell medians. Missing cells are left out of the fit
// rather than guessed, which is why it is solved by turns rather than by row and column means.
func Split(cells map[Cell]float64) Shares {
	shares := Shares{Caller: map[string]float64{}, Callee: map[string]float64{}, Cells: len(cells)}
	if len(cells) == 0 {
		return shares
	}
	for cell, value := range cells {
		shares.Baseline += value / float64(len(cells))
		shares.Caller[cell.Caller] = 0
		shares.Callee[cell.Callee] = 0
	}

	for range 200 {
		fitRole(cells, shares.Caller, func(c Cell) string { return c.Caller }, func(c Cell, v float64) float64 {
			return v - shares.Baseline - shares.Callee[c.Callee]
		})
		shares.Baseline += centre(shares.Caller)
		fitRole(cells, shares.Callee, func(c Cell) string { return c.Callee }, func(c Cell, v float64) float64 {
			return v - shares.Baseline - shares.Caller[c.Caller]
		})
		shares.Baseline += centre(shares.Callee)
	}

	var squares float64
	for cell, value := range cells {
		miss := value - shares.Baseline - shares.Caller[cell.Caller] - shares.Callee[cell.Callee]
		squares += miss * miss
	}
	shares.Residual = math.Sqrt(squares / float64(len(cells)))
	return shares
}

// fitRole sets each vendor's share in one role to the mean of what is left of its cells.
func fitRole(cells map[Cell]float64, role map[string]float64, vendorOf func(Cell) string, left func(Cell, float64) float64) {
	sums := map[string]float64{}
	counts := map[string]int{}
	for cell, value := range cells {
		vendor := vendorOf(cell)
		sums[vendor] += left(cell, value)
		counts[vendor]++
	}
	for vendor := range role {
		role[vendor] = sums[vendor] / float64(counts[vendor])
	}
}

// centre shifts a role's shares to average zero and returns what was taken out.
func centre(role map[string]float64) float64 {
	var mean float64
	for _, share := range role {
		mean += share / float64(len(role))
	}
	for vendor := range role {
		role[vendor] -= mean
	}
	return mean
}

// Render writes the report: for each region the bench ran in, each city's matrix in both
// directions, how much each vendor adds as caller and as callee, and where each vendor's
// side of each cell was served from.
func Render(results []Result, places map[string]Place) string {
	var out strings.Builder
	out.WriteString("# Telephony latency\n\n")
	out.WriteString("Mouth-to-ear latency between phone numbers, measured by a timestamp sent as tones " +
		"from one leg of a call and decoded on the other. Both legs end on the bench, so it is one clock " +
		"and the figures include the bench's own trip to each vendor and back.\n\n")
	out.WriteString("A cell is the median over its calls of each call's median stamp; p95 is over calls. " +
		"Rows are the calling vendor, columns the called vendor.\n\n")

	for _, region := range distinct(results, func(r Result) string { return r.Region }) {
		inRegion := filter(results, func(r Result) bool { return r.Region == region })
		vendors := vendorsOf(inRegion)
		fmt.Fprintf(&out, "## %s\n\n", or(region, "unnamed region"))
		fmt.Fprintf(&out, "%d calls, %d failed.\n\n", len(inRegion),
			len(filter(inRegion, func(r Result) bool { return r.Error != "" })))

		writeMatrix(&out, "All calls, caller → callee", inRegion, vendors, Forward)
		writeMatrix(&out, "All calls, callee → caller", inRegion, vendors, Backward)
		writeShares(&out, inRegion, vendors)

		for _, city := range distinct(inRegion, func(r Result) string { return r.Caller.City }) {
			sameCity := filter(inRegion, func(r Result) bool { return r.Caller.City == city && r.Callee.City == city })
			if len(sameCity) == 0 {
				continue
			}
			fmt.Fprintf(&out, "### %s\n\n", city)
			writeMatrix(&out, "Caller → callee", sameCity, vendors, Forward)
			writeMatrix(&out, "Callee → caller", sameCity, vendors, Backward)
			writeNetwork(&out, sameCity, vendors, places)
		}
		writeSetup(&out, inRegion, vendors)
		writeFailures(&out, inRegion)
	}
	return out.String()
}

func writeMatrix(out *strings.Builder, title string, results []Result, vendors []string, direction Direction) {
	fmt.Fprintf(out, "**%s**\n\n| caller \\ callee | %s |\n|---|%s\n", title,
		strings.Join(vendors, " | "), strings.Repeat("---|", len(vendors)))
	for _, caller := range vendors {
		row := make([]string, 0, len(vendors))
		for _, callee := range vendors {
			row = append(row, Summarise(inCell(results, Cell{caller, callee}), direction).String())
		}
		fmt.Fprintf(out, "| %s | %s |\n", caller, strings.Join(row, " | "))
	}
	out.WriteString("\n")
}

// writeShares splits every city's matrix, and all of them together, into what each vendor
// adds as the caller and as the callee.
func writeShares(out *strings.Builder, results []Result, vendors []string) {
	out.WriteString("**What each vendor adds** (caller → callee)\n\n")
	out.WriteString("Each cell is fitted as a baseline plus the calling vendor's share plus the called " +
		"vendor's share. A share is against the average vendor in that role, so negative is faster.\n\n")

	header := []string{"scope", "baseline"}
	for _, vendor := range vendors {
		header = append(header, vendor+" as caller")
	}
	for _, vendor := range vendors {
		header = append(header, vendor+" as callee")
	}
	header = append(header, "unexplained")
	fmt.Fprintf(out, "| %s |\n|%s\n", strings.Join(header, " | "), strings.Repeat("---|", len(header)))

	scopes := []string{""}
	scopes = append(scopes, distinct(results, func(r Result) string { return r.Caller.City })...)
	for _, scope := range scopes {
		inScope := results
		if scope != "" {
			inScope = filter(results, func(r Result) bool { return r.Caller.City == scope && r.Callee.City == scope })
		}
		cells := map[Cell]float64{}
		for _, caller := range vendors {
			for _, callee := range vendors {
				if stats := Summarise(inCell(inScope, Cell{caller, callee}), Forward); stats.Calls > 0 {
					cells[Cell{caller, callee}] = stats.Median
				}
			}
		}
		if !identifiable(cells, vendors) {
			continue
		}
		shares := Split(cells)
		row := []string{or(scope, "all calls"), fmt.Sprintf("%.0f ms", shares.Baseline)}
		for _, vendor := range vendors {
			row = append(row, signed(shares.Caller[vendor]))
		}
		for _, vendor := range vendors {
			row = append(row, signed(shares.Callee[vendor]))
		}
		row = append(row, fmt.Sprintf("%.0f ms", shares.Residual))
		fmt.Fprintf(out, "| %s |\n", strings.Join(row, " | "))
	}
	out.WriteString("\n")
}

// identifiable is whether a matrix has the cells to split: with every vendor in both roles
// and more cells than shares, or a share could be any value at all.
func identifiable(cells map[Cell]float64, vendors []string) bool {
	callers, callees := map[string]bool{}, map[string]bool{}
	for cell := range cells {
		callers[cell.Caller] = true
		callees[cell.Callee] = true
	}
	return len(callers) == len(vendors) && len(callees) == len(vendors) && len(cells) > 2*len(vendors)-2
}

// writeNetwork lists the addresses each vendor's side of each cell came from.
func writeNetwork(out *strings.Builder, results []Result, vendors []string, places map[string]Place) {
	out.WriteString("**Where the media went**\n\n| caller → callee | caller's media | callee's media |\n|---|---|---|\n")
	for _, caller := range vendors {
		for _, callee := range vendors {
			cell := inCell(results, Cell{caller, callee})
			if len(cell) == 0 {
				continue
			}
			fmt.Fprintf(out, "| %s → %s | %s | %s |\n", caller, callee,
				describe(cell, func(r Result) string { return r.CallerMedia }, places),
				describe(cell, func(r Result) string { return r.CalleeMedia }, places))
		}
	}
	out.WriteString("\n")
}

func writeSetup(out *strings.Builder, results []Result, vendors []string) {
	out.WriteString("**Call setup**\n\nRing is from asking the calling vendor to dial until the called vendor " +
		"delivers the call; bridge is from answering it until the calling vendor connects its side.\n\n")
	out.WriteString("| caller → callee | ring | bridge |\n|---|---|---|\n")
	for _, caller := range vendors {
		for _, callee := range vendors {
			var ring, bridge []float64
			for _, result := range inCell(results, Cell{caller, callee}) {
				if result.RingMs > 0 {
					ring = append(ring, result.RingMs)
				}
				if result.BridgeMs > 0 {
					bridge = append(bridge, result.BridgeMs)
				}
			}
			if len(ring) == 0 {
				continue
			}
			bridgeText := "–"
			if len(bridge) > 0 {
				bridgeText = fmt.Sprintf("%.0f ms", percentile(bridge, 50))
			}
			fmt.Fprintf(out, "| %s → %s | %.0f ms | %s |\n", caller, callee, percentile(ring, 50), bridgeText)
		}
	}
	out.WriteString("\n")
}

func writeFailures(out *strings.Builder, results []Result) {
	counts := map[string]int{}
	for _, result := range results {
		if result.Error != "" {
			counts[fmt.Sprintf("%s → %s: %s", result.Caller.Vendor, result.Callee.Vendor, result.Error)]++
		}
	}
	if len(counts) == 0 {
		return
	}
	out.WriteString("**Failures**\n\n")
	for _, failure := range sortedKeys(counts) {
		fmt.Fprintf(out, "- %s (×%d)\n", failure, counts[failure])
	}
	out.WriteString("\n")
}

// describe is the distinct places one side of a cell's media came from, most used first.
func describe(results []Result, addressOf func(Result) string, places map[string]Place) string {
	counts := map[string]int{}
	for _, result := range results {
		address, err := netip.ParseAddrPort(addressOf(result))
		if err != nil || !address.IsValid() {
			continue
		}
		ip := address.Addr().String()
		place, ok := places[ip]
		if !ok {
			place = Place{IP: ip}
		}
		counts[place.String()]++
	}
	described := sortedKeys(counts)
	slices.SortStableFunc(described, func(a, b string) int { return counts[b] - counts[a] })
	for i, place := range described {
		if counts[place] > 1 {
			described[i] = fmt.Sprintf("%s (×%d)", place, counts[place])
		}
	}
	return or(strings.Join(described, "<br>"), "–")
}

func inCell(results []Result, cell Cell) []Result {
	return filter(results, func(r Result) bool { return r.Caller.Vendor == cell.Caller && r.Callee.Vendor == cell.Callee })
}

func vendorsOf(results []Result) []string {
	seen := map[string]bool{}
	for _, result := range results {
		seen[result.Caller.Vendor] = true
		seen[result.Callee.Vendor] = true
	}
	return sortedKeys(seen)
}

func distinct(results []Result, key func(Result) string) []string {
	seen := map[string]bool{}
	for _, result := range results {
		seen[key(result)] = true
	}
	return sortedKeys(seen)
}

func filter(results []Result, keep func(Result) bool) []Result {
	var kept []Result
	for _, result := range results {
		if keep(result) {
			kept = append(kept, result)
		}
	}
	return kept
}

// percentile is the nearest-rank percentile, which is a value that actually occurred.
func percentile(values []float64, p float64) float64 {
	sorted := slices.Clone(values)
	slices.Sort(sorted)
	rank := int(math.Ceil(p / 100 * float64(len(sorted))))
	return sorted[max(rank, 1)-1]
}

func signed(ms float64) string {
	return fmt.Sprintf("%+.0f ms", ms)
}

func or(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}
