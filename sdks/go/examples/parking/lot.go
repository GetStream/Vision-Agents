package main

import (
	"image"
	"image/color"
	"image/draw"
	"regexp"
	"strconv"
	"strings"
)

// What the lot is asked for. LocateAnything answers every category it was given under a
// <ref> naming it, so these are also the labels the boxes come back under.
const (
	carLabel  = "parked car"
	freeLabel = "empty parking space"
)

// query asks for both at once, which is one pass over the image rather than two.
const query = "Locate all the instances that matches the following description: " +
	carLabel + "</c>" + freeLabel + "."

// grid is the scale LocateAnything writes coordinates on, whatever the image's size.
const grid = 1000

// token is one <ref> naming a category or one <box> belonging to the last one named.
var token = regexp.MustCompile(`<ref>(.*?)</ref>|<box><(\d+)><(\d+)><(\d+)><(\d+)></box>`)

// Detection is one box the model drew, in the image's pixels.
type Detection struct {
	Label string
	Box   image.Rectangle
}

// parse reads the model's answer into boxes over an image with those bounds.
func parse(answer string, bounds image.Rectangle) []Detection {
	var detections []Detection
	label := ""
	for _, match := range token.FindAllStringSubmatch(answer, -1) {
		if match[1] != "" {
			label = strings.ToLower(strings.TrimSpace(match[1]))
			continue
		}
		if match[2] == "" {
			continue
		}
		scale := func(value string, size, offset int) int {
			n, _ := strconv.Atoi(value)
			return offset + n*size/grid
		}
		box := image.Rect(
			scale(match[2], bounds.Dx(), bounds.Min.X),
			scale(match[3], bounds.Dy(), bounds.Min.Y),
			scale(match[4], bounds.Dx(), bounds.Min.X),
			scale(match[5], bounds.Dy(), bounds.Min.Y),
		).Intersect(bounds)
		if box.Empty() {
			continue
		}
		detections = append(detections, Detection{Label: label, Box: box})
	}
	return detections
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
