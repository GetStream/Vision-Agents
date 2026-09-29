package main

import (
	"image"
	"image/color"
	"image/draw"
	"regexp"
	"slices"
	"strconv"
)

// What the lot is asked for.
//
// Each is asked on its own. Asked for both in one prompt, LocateAnything finds the cars
// and then repeats one empty space until it runs out of tokens; asked separately, each
// comes back in a few seconds.
const (
	carLabel  = "parked car"
	freeLabel = "empty parking space"
)

// query is the prompt that asks for every instance of one thing.
func query(label string) string {
	return "Locate all the instances that matches the following description: " + label + "."
}

// grid is the scale LocateAnything writes coordinates on, whatever the image's size.
const grid = 1000

// boxPattern is one box in the model's answer.
var boxPattern = regexp.MustCompile(`<box><(\d+)><(\d+)><(\d+)><(\d+)></box>`)

// Detection is one box the model drew, in the image's pixels.
type Detection struct {
	Label string
	Box   image.Rectangle
}

// parse reads the answer to the question about label into boxes over an image with
// those bounds.
func parse(answer, label string, bounds image.Rectangle) []Detection {
	var detections []Detection
	for _, match := range boxPattern.FindAllStringSubmatch(answer, -1) {
		scale := func(value string, size, offset int) int {
			n, _ := strconv.Atoi(value)
			return offset + n*size/grid
		}
		box := image.Rect(
			scale(match[1], bounds.Dx(), bounds.Min.X),
			scale(match[2], bounds.Dy(), bounds.Min.Y),
			scale(match[3], bounds.Dx(), bounds.Min.X),
			scale(match[4], bounds.Dy(), bounds.Min.Y),
		).Intersect(bounds)
		if box.Empty() {
			continue
		}
		detections = append(detections, Detection{Label: label, Box: box})
	}
	return detections
}

// oversized is how many times a typical car's area a box may be before it is taken for
// a group of cars or a stretch of open tarmac rather than one car or one space.
const oversized = 4

// duplicate is how much two boxes of the same label may overlap, as intersection over
// union, before the second is taken for the first one drawn again.
const duplicate = 0.7

// plausible drops what the model sometimes answers besides one box per thing: a box
// around a whole row or road, and the same box twice.
func plausible(detections []Detection) []Detection {
	var areas []int
	for _, detection := range detections {
		if detection.Label == carLabel {
			areas = append(areas, area(detection.Box))
		}
	}
	limit := 0
	if len(areas) > 0 {
		slices.Sort(areas)
		limit = oversized * areas[len(areas)/2]
	}

	kept := detections[:0:0]
	for _, detection := range detections {
		if limit > 0 && area(detection.Box) > limit {
			continue
		}
		if slices.ContainsFunc(kept, func(other Detection) bool {
			return other.Label == detection.Label && overlap(other.Box, detection.Box) > duplicate
		}) {
			continue
		}
		kept = append(kept, detection)
	}
	return kept
}

func area(box image.Rectangle) int { return box.Dx() * box.Dy() }

// overlap is the intersection over union of two boxes.
func overlap(a, b image.Rectangle) float64 {
	shared := area(a.Intersect(b))
	return float64(shared) / float64(area(a)+area(b)-shared)
}

// Occupancy is how full the lot is.
type Occupancy struct {
	Cars   int
	Free   int
	Spaces int
}

// occupancy counts the lot. When the number of spaces is known it is the capacity and
// whatever is not taken is free; otherwise the capacity is what the model could see,
// taken or not.
func occupancy(detections []Detection, spaces int) Occupancy {
	var o Occupancy
	for _, detection := range detections {
		switch detection.Label {
		case carLabel:
			o.Cars++
		case freeLabel:
			o.Free++
		}
	}
	if spaces > 0 {
		o.Spaces = spaces
		o.Free = max(spaces-o.Cars, 0)
		return o
	}
	o.Spaces = o.Cars + o.Free
	return o
}

// FullPercent is the share of the lot that is taken, from 0 to 100.
func (o Occupancy) FullPercent() float64 {
	if o.Spaces == 0 {
		return 0
	}
	return min(100, 100*float64(o.Cars)/float64(o.Spaces))
}

// FreePercent is the share of the lot that is open, from 0 to 100.
func (o Occupancy) FreePercent() float64 {
	if o.Spaces == 0 {
		return 0
	}
	return 100 - o.FullPercent()
}

var (
	taken = color.RGBA{R: 220, G: 38, B: 38, A: 255}
	open  = color.RGBA{R: 22, G: 163, B: 74, A: 255}
)

// annotate draws every box over the picture, red for a car and green for a free space,
// with a bar across the top that is as red as the lot is full.
func annotate(picture image.Image, detections []Detection, o Occupancy) *image.RGBA {
	bounds := picture.Bounds()
	canvas := image.NewRGBA(bounds)
	draw.Draw(canvas, bounds, picture, bounds.Min, draw.Src)

	for _, detection := range detections {
		shade := open
		if detection.Label == carLabel {
			shade = taken
		}
		tint := color.NRGBA{R: shade.R, G: shade.G, B: shade.B, A: 60}
		draw.Draw(canvas, detection.Box, image.NewUniform(tint), image.Point{}, draw.Over)
		outline(canvas, detection.Box, shade, 3)
	}

	bar := image.Rect(bounds.Min.X, bounds.Min.Y, bounds.Max.X, bounds.Min.Y+max(12, bounds.Dy()/40))
	split := bar.Min.X + int(float64(bar.Dx())*o.FullPercent()/100)
	draw.Draw(canvas, image.Rect(bar.Min.X, bar.Min.Y, split, bar.Max.Y), image.NewUniform(taken), image.Point{}, draw.Src)
	draw.Draw(canvas, image.Rect(split, bar.Min.Y, bar.Max.X, bar.Max.Y), image.NewUniform(open), image.Point{}, draw.Src)
	return canvas
}

// outline strokes the inside edge of a rectangle.
func outline(canvas *image.RGBA, box image.Rectangle, shade color.RGBA, width int) {
	fill := image.NewUniform(shade)
	for _, edge := range []image.Rectangle{
		{Min: box.Min, Max: image.Pt(box.Max.X, box.Min.Y+width)},
		{Min: image.Pt(box.Min.X, box.Max.Y-width), Max: box.Max},
		{Min: box.Min, Max: image.Pt(box.Min.X+width, box.Max.Y)},
		{Min: image.Pt(box.Max.X-width, box.Min.Y), Max: box.Max},
	} {
		draw.Draw(canvas, edge.Intersect(box), fill, image.Point{}, draw.Src)
	}
}
