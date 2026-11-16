package scheduling

import "time"

// ResolveLocal turns a wall-clock reading (held in UTC, see wallOf) into the
// instant it names in loc. Go's time.Date makes no promise for the two shapes
// a daylight-saving change creates, so the choice is made here:
//
//   - a reading inside the spring-forward gap moves forward by the length of
//     the gap (02:30 on a day the clocks jump 02:00 to 03:00 is 03:30);
//   - an ambiguous fall-back reading takes its first occurrence.
//
// The candidates are the zone offsets in force around the reading: for each,
// the instant the reading would be under that offset, kept when converting it
// back reads the same wall clock.
func ResolveLocal(wall time.Time, loc *time.Location) time.Time {
	target := wallOf(wall)
	offsets := offsetsAround(target, loc)

	var first time.Time
	for _, offset := range offsets {
		candidate := target.Add(-offset)
		if wallOf(candidate.In(loc)).Equal(target) {
			if first.IsZero() || candidate.Before(first) {
				first = candidate
			}
		}
	}
	if !first.IsZero() {
		return first
	}

	// No offset reads back as the target: the reading sits in a gap. The
	// offset in force before the jump is the one to count from, which lands
	// the instant past the gap reading target + the gap's length.
	before, after, found := transitionAround(target, loc)
	if found && after > before {
		return target.Add(-before)
	}
	return time.Date(target.Year(), target.Month(), target.Day(),
		target.Hour(), target.Minute(), target.Second(), 0, loc)
}

// offsetsAround lists the distinct offsets loc is under in the days around
// target, in the order they take effect.
func offsetsAround(target time.Time, loc *time.Location) []time.Duration {
	seen := map[time.Duration]bool{}
	offsets := []time.Duration{}
	for step := -32; step <= 32; step++ {
		at := target.Add(time.Duration(step) * 3 * time.Hour)
		_, seconds := at.In(loc).Zone()
		offset := time.Duration(seconds) * time.Second
		if !seen[offset] {
			seen[offset] = true
			offsets = append(offsets, offset)
		}
	}
	return offsets
}

// transitionAround finds the offset change nearest target and returns the
// offsets either side of it.
func transitionAround(target time.Time, loc *time.Location) (time.Duration, time.Duration, bool) {
	offsetAt := func(at time.Time) time.Duration {
		_, seconds := at.In(loc).Zone()
		return time.Duration(seconds) * time.Second
	}
	start := target.Add(-72 * time.Hour)
	previous := offsetAt(start)
	for step := 1; step <= 144; step++ {
		at := start.Add(time.Duration(step) * time.Hour)
		offset := offsetAt(at)
		if offset != previous {
			return previous, offset, true
		}
		previous = offset
	}
	return 0, 0, false
}
