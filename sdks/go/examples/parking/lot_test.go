package main

import (
	"image"
	"image/color"
	"testing"
)

func TestTheAnswerIsReadIntoLabelledBoxesInPixels(t *testing.T) {
	answer := "<ref>parked car</ref><box><0><0><500><500></box><box><500><500><1000><1000></box>" +
		"<ref>empty parking space</ref><box><100><200><300><400></box>"

	detections := parse(answer, image.Rect(0, 0, 200, 100))

	want := []Detection{
		{Label: carLabel, Box: image.Rect(0, 0, 100, 50)},
		{Label: carLabel, Box: image.Rect(100, 50, 200, 100)},
		{Label: freeLabel, Box: image.Rect(20, 20, 60, 40)},
	}
	if len(detections) != len(want) {
		t.Fatalf("got %d detections, want %d: %+v", len(detections), len(want), detections)
	}
	for i := range want {
		if detections[i] != want[i] {
			t.Errorf("detection %d is %+v, want %+v", i, detections[i], want[i])
		}
	}
}

func TestABoxWithNoAreaIsDropped(t *testing.T) {
	detections := parse("<ref>parked car</ref><box><500><500><500><600></box>", image.Rect(0, 0, 100, 100))
	if len(detections) != 0 {
		t.Errorf("a line should not count as a car: %+v", detections)
	}
}

func TestTheCapacityIsWhatTheModelSawWhenTheSpacesAreUnknown(t *testing.T) {
	detections := []Detection{{Label: carLabel}, {Label: carLabel}, {Label: carLabel}, {Label: freeLabel}}

	lot := occupancy(detections, 0)

	if lot != (Occupancy{Cars: 3, Free: 1, Spaces: 4}) {
		t.Errorf("got %+v", lot)
	}
	if lot.FullPercent() != 75 || lot.FreePercent() != 25 {
		t.Errorf("got %.0f%% full and %.0f%% free", lot.FullPercent(), lot.FreePercent())
	}
}

func TestAKnownCapacityDecidesWhatIsFree(t *testing.T) {
	detections := []Detection{{Label: carLabel}, {Label: carLabel}, {Label: freeLabel}}

	lot := occupancy(detections, 10)

	if lot != (Occupancy{Cars: 2, Free: 8, Spaces: 10}) {
		t.Errorf("got %+v", lot)
	}
	if lot.FullPercent() != 20 {
		t.Errorf("got %.0f%% full", lot.FullPercent())
	}
}

func TestMoreCarsThanSpacesIsAFullLotNotAnOverfullOne(t *testing.T) {
	lot := occupancy([]Detection{{Label: carLabel}, {Label: carLabel}, {Label: carLabel}}, 2)

	if lot.Free != 0 || lot.FullPercent() != 100 || lot.FreePercent() != 0 {
		t.Errorf("got %+v, %.0f%% full", lot, lot.FullPercent())
	}
}

func TestAnEmptyAnswerIsAnEmptyLot(t *testing.T) {
	lot := occupancy(parse("", image.Rect(0, 0, 10, 10)), 0)

	if lot.FullPercent() != 0 || lot.FreePercent() != 0 {
		t.Errorf("nothing seen should not divide by zero: %+v", lot)
	}
}

func TestTheBarIsAsRedAsTheLotIsFull(t *testing.T) {
	picture := image.NewRGBA(image.Rect(0, 0, 100, 100))
	lot := Occupancy{Cars: 3, Free: 1, Spaces: 4}

	canvas := annotate(picture, nil, lot)

	if got := canvas.RGBAAt(10, 2); got != taken {
		t.Errorf("the full part of the bar is %v", got)
	}
	if got := canvas.RGBAAt(90, 2); got != open {
		t.Errorf("the free part of the bar is %v", got)
	}
	if got := canvas.RGBAAt(50, 50); got != (color.RGBA{}) {
		t.Errorf("below the bar the picture is untouched, got %v", got)
	}
}
