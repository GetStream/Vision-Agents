package report

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"

	"golang.org/x/image/font"
	"golang.org/x/image/font/gofont/gobold"
	"golang.org/x/image/font/gofont/goregular"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
)

// The card is drawn at twice the size it is shown, so it stays sharp on a retina screen.
const (
	cardWidth  = 2000
	cardMargin = 72
	labelWidth = 520
	valueWidth = 300
	laneHeight = 46
)

var (
	cardBackground = color.RGBA{0xfd, 0xfd, 0xfc, 0xff}
	cardInk        = color.RGBA{0x12, 0x14, 0x19, 0xff}
	cardMuted      = color.RGBA{0x6b, 0x71, 0x7c, 0xff}
	cardRule       = color.RGBA{0xdd, 0xe0, 0xe5, 0xff}
	// cardRunColors go to the runs in order: ours first in blue, LiveKit after it in orange.
	cardRunColors = []color.RGBA{
		{0x2a, 0x78, 0xd6, 0xff},
		{0xeb, 0x68, 0x34, 0xff},
		{0x8a, 0x5c, 0xd6, 0xff},
		{0x1f, 0x8a, 0x5b, 0xff},
	}
)

// PNG draws the digest as a scorecard: one block per metric, one lane per run, a dot for
// the value and a bar for its 95% interval.
func (d Digest) PNG(title string) ([]byte, error) {
	faces, err := loadCardFaces()
	if err != nil {
		return nil, err
	}
	lanes := len(d.Runs)
	blockHeight := 56 + lanes*laneHeight + 28
	height := 250 + len(d.Rows)*blockHeight + 40
	img := image.NewRGBA(image.Rect(0, 0, cardWidth, height))
	draw.Draw(img, img.Bounds(), &image.Uniform{cardBackground}, image.Point{}, draw.Src)

	text(img, faces.title, cardInk, cardMargin, 100, title)
	text(img, faces.small, cardMuted, cardMargin, 152, d.Meta())
	x := cardMargin
	for i, run := range d.Runs {
		fill(img, x, 190, x+26, 216, runColor(i))
		x += 40
		text(img, faces.body, cardInk, x, 214, run)
		x += measure(faces.body, run) + 56
	}

	y := 250
	plotLeft := cardMargin + labelWidth
	plotRight := cardWidth - cardMargin - valueWidth
	for _, row := range d.Rows {
		fill(img, cardMargin, y, cardWidth-cardMargin, y+2, cardRule)
		text(img, faces.heading, cardInk, cardMargin, y+60, row.Name)
		if verdict := row.verdict(d.Runs); verdict != "" {
			text(img, faces.small, cardMuted, cardMargin, y+102, verdict[len(" → "):])
		}
		scale := row.scale()
		for i, cell := range row.Cells {
			laneY := y + 56 + i*laneHeight
			mid := laneY + laneHeight/2
			fill(img, plotLeft, mid-1, plotRight, mid+1, cardRule)
			if cell.Missing {
				text(img, faces.body, cardMuted, plotRight+32, mid+12, "—")
				continue
			}
			at := func(v float64) int { return plotLeft + int(v/scale*float64(plotRight-plotLeft)) }
			c := runColor(i)
			if cell.HasCI {
				band := color.NRGBA{R: c.R, G: c.G, B: c.B, A: 0x59}
				fillOver(img, at(cell.Lo), mid-8, max(at(cell.Hi), at(cell.Lo)+2), mid+8, band)
			}
			dot(img, at(cell.Value), mid, 13, c)
			label := cell.Text(row.Unit)
			if cell.Samples > 0 {
				label += fmt.Sprintf("  n=%d", cell.Samples)
			}
			face := faces.body
			if i == row.Best && row.Decided() {
				face = faces.bodyBold
			}
			text(img, face, cardInk, plotRight+32, mid+12, label)
		}
		y += blockHeight
	}

	var out bytes.Buffer
	if err := png.Encode(&out, img); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

// scale is the value the right edge of a row's plot stands for.
func (r DigestRow) scale() float64 {
	if r.Unit == "%" {
		return 100
	}
	top := 0.0
	for _, cell := range r.Cells {
		top = max(top, cell.Value, cell.Hi)
	}
	if top == 0 {
		return 1
	}
	return top * 1.08
}

type cardFaces struct {
	title, heading, body, bodyBold, small font.Face
}

func loadCardFaces() (cardFaces, error) {
	regular, err := opentype.Parse(goregular.TTF)
	if err != nil {
		return cardFaces{}, err
	}
	bold, err := opentype.Parse(gobold.TTF)
	if err != nil {
		return cardFaces{}, err
	}
	face := func(f *opentype.Font, size float64) (font.Face, error) {
		return opentype.NewFace(f, &opentype.FaceOptions{Size: size, DPI: 72, Hinting: font.HintingFull})
	}
	var faces cardFaces
	for _, want := range []struct {
		into *font.Face
		font *opentype.Font
		size float64
	}{
		{&faces.title, bold, 60},
		{&faces.heading, bold, 34},
		{&faces.body, regular, 30},
		{&faces.bodyBold, bold, 30},
		{&faces.small, regular, 28},
	} {
		if *want.into, err = face(want.font, want.size); err != nil {
			return cardFaces{}, err
		}
	}
	return faces, nil
}

func runColor(i int) color.RGBA { return cardRunColors[i%len(cardRunColors)] }

func text(img *image.RGBA, face font.Face, c color.Color, x, y int, s string) {
	drawer := font.Drawer{Dst: img, Src: image.NewUniform(c), Face: face, Dot: fixed.P(x, y)}
	drawer.DrawString(s)
}

func measure(face font.Face, s string) int {
	return font.MeasureString(face, s).Ceil()
}

func fill(img *image.RGBA, x0, y0, x1, y1 int, c color.Color) {
	draw.Draw(img, image.Rect(x0, y0, x1, y1), image.NewUniform(c), image.Point{}, draw.Src)
}

func fillOver(img *image.RGBA, x0, y0, x1, y1 int, c color.Color) {
	draw.Draw(img, image.Rect(x0, y0, x1, y1), image.NewUniform(c), image.Point{}, draw.Over)
}

func dot(img *image.RGBA, cx, cy, r int, c color.Color) {
	for y := -r; y <= r; y++ {
		for x := -r; x <= r; x++ {
			if x*x+y*y <= r*r {
				img.Set(cx+x, cy+y, c)
			}
		}
	}
}
