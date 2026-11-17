package scheduling

import (
	"testing"
	"time"
)

func quietZone(t *testing.T) *time.Location {
	t.Helper()
	loc, err := LoadZone("Europe/Amsterdam")
	if err != nil {
		t.Fatal(err)
	}
	return loc
}

func TestTheNightIsQuietAndEndsInTheMorning(t *testing.T) {
	loc := quietZone(t)
	now := time.Date(2026, 10, 20, 23, 30, 0, 0, loc)

	until, held := QuietEnd("22:00", "07:00", loc, now)

	if !held {
		t.Fatal("23:30 is inside 22:00 to 07:00")
	}
	want := time.Date(2026, 10, 21, 7, 0, 0, 0, loc)
	if !until.Equal(want) {
		t.Errorf("the night ends at %v, want %v", until, want)
	}
}

func TestTheEarlyMorningIsTheSameNight(t *testing.T) {
	loc := quietZone(t)
	now := time.Date(2026, 10, 20, 3, 0, 0, 0, loc)

	until, held := QuietEnd("22:00", "07:00", loc, now)

	if !held {
		t.Fatal("03:00 is inside 22:00 to 07:00")
	}
	want := time.Date(2026, 10, 20, 7, 0, 0, 0, loc)
	if !until.Equal(want) {
		t.Errorf("the night ends at %v, want %v", until, want)
	}
}

func TestTheDayIsNotQuiet(t *testing.T) {
	loc := quietZone(t)
	for _, hour := range []int{7, 12, 21} {
		now := time.Date(2026, 10, 20, hour, 0, 0, 0, loc)
		if _, held := QuietEnd("22:00", "07:00", loc, now); held {
			t.Errorf("%02d:00 is not inside 22:00 to 07:00", hour)
		}
	}
}

func TestQuietHoursWithinOneDay(t *testing.T) {
	loc := quietZone(t)
	now := time.Date(2026, 10, 20, 13, 30, 0, 0, loc)

	until, held := QuietEnd("13:00", "14:00", loc, now)

	if !held {
		t.Fatal("13:30 is inside 13:00 to 14:00")
	}
	want := time.Date(2026, 10, 20, 14, 0, 0, 0, loc)
	if !until.Equal(want) {
		t.Errorf("the span ends at %v, want %v", until, want)
	}
	if _, held := QuietEnd("13:00", "14:00", loc, time.Date(2026, 10, 20, 14, 0, 0, 0, loc)); held {
		t.Error("14:00 is the end itself, not inside it")
	}
}

func TestAQuietSpanThatDoesNotReadIsNoQuietHours(t *testing.T) {
	loc := quietZone(t)
	now := time.Date(2026, 10, 20, 23, 30, 0, 0, loc)
	for _, span := range [][2]string{
		{"ten", "07:00"}, {"22:00", "7"}, {"22:00", "22:00"}, {"", ""}, {"24:00", "07:00"},
	} {
		if _, held := QuietEnd(span[0], span[1], loc, now); held {
			t.Errorf("%q to %q held, and it does not read", span[0], span[1])
		}
		if _, _, ok := QuietHours(span[0], span[1]); ok {
			t.Errorf("%q to %q read as quiet hours", span[0], span[1])
		}
	}
}

func TestQuietHoursEndWhenTheClockSaysAcrossDaylightSaving(t *testing.T) {
	loc := quietZone(t)
	// The night of 29 March 2026 the clocks jump 02:00 to 03:00 in Amsterdam:
	// the night is an hour shorter, and still ends at 07:00 as read.
	now := time.Date(2026, 3, 28, 23, 30, 0, 0, loc)

	until, held := QuietEnd("22:00", "07:00", loc, now)

	if !held {
		t.Fatal("23:30 is inside 22:00 to 07:00")
	}
	if read := until.In(loc); read.Hour() != 7 || read.Day() != 29 {
		t.Errorf("the night ends at %v, want 07:00 on the 29th as the clock reads", until)
	}
	if gap := until.Sub(now); gap != 6*time.Hour+30*time.Minute {
		t.Errorf("the night lasts %v across the jump, want 6h30m", gap)
	}
}
