package extract

import (
	"math"

	"github.com/klippa-app/go-pdfium/structs"
)

// figureShare is how much of a page its pictures and drawings cover before the page is
// read from its picture too. A chart or a diagram worth asking about takes a tenth of a
// page or more; a logo, an icon or a signature takes less.
const figureShare = 0.10

// figureGrid is how many cells a page is divided into on each side to measure how much of
// it figures cover, so figures that overlap are not counted twice. A cell of an A4 page is
// about 15 by 21 points.
const figureGrid = 40

// minFigureSide is the narrowest a picture or a shape is and still part of a figure:
// anything thinner is a rule, an underline or a table's border.
const minFigureSide = 4.0

// backgroundShare is how much of a page a picture or a shape covers from which it is the
// page's background, its frame or its letterhead rather than a figure on it. A scan is a
// picture covering the page, which Scanned says already.
const backgroundShare = 0.8

// maxPageObjects is how many of a page's objects are looked at for figures. A page drawn
// from thousands of shapes, as a map or an engineering drawing is, has passed figureShare
// long before the last of them.
const maxPageObjects = 5000

// maxPathPoints is how many points of a shape are looked at for a curve or a slant.
const maxPathPoints = 64

// maxFormDepth is how deep forms within forms are followed. A form is a group of objects
// drawn as one, which some producers wrap a whole page in.
const maxFormDepth = 4

// slanted says a line that moves dx across and dy up is neither level nor upright, beyond
// what rounding leaves in a line drawn either way.
func slanted(dx, dy float64) bool {
	dx, dy = math.Abs(dx), math.Abs(dy)
	return min(dx, dy) > 0.02*max(dx, dy)
}

// box is a rectangle on a page, in points. PDF coordinates grow upwards.
type box struct {
	left, bottom, right, top float64
}

func (b box) width() float64  { return b.right - b.left }
func (b box) height() float64 { return b.top - b.bottom }

// overlaps says the two boxes share some area.
func (b box) overlaps(o box) bool {
	return b.left < o.right && o.left < b.right && b.bottom < o.top && o.bottom < b.top
}

// matrix maps a form's own coordinates onto the page, as PDF writes it: a point (x, y)
// lands at (a·x + c·y + e, b·x + d·y + f).
type matrix struct {
	a, b, c, d, e, f float64
}

var identity = matrix{a: 1, d: 1}

func matrixOf(m structs.FPDF_FS_MATRIX) matrix {
	return matrix{a: float64(m.A), b: float64(m.B), c: float64(m.C), d: float64(m.D), e: float64(m.E), f: float64(m.F)}
}

// then is m followed by outer: a point mapped by m, then by outer.
func (m matrix) then(outer matrix) matrix {
	return matrix{
		a: m.a*outer.a + m.b*outer.c,
		b: m.a*outer.b + m.b*outer.d,
		c: m.c*outer.a + m.d*outer.c,
		d: m.c*outer.b + m.d*outer.d,
		e: m.e*outer.a + m.f*outer.c + outer.e,
		f: m.e*outer.b + m.f*outer.d + outer.f,
	}
}

func (m matrix) point(x, y float64) (float64, float64) {
	return m.a*x + m.c*y + m.e, m.b*x + m.d*y + m.f
}

// box is b mapped by m, as the smallest upright box holding all four of its corners.
func (m matrix) box(b box) box {
	out := box{left: 1e18, bottom: 1e18, right: -1e18, top: -1e18}
	for _, corner := range [][2]float64{{b.left, b.bottom}, {b.left, b.top}, {b.right, b.bottom}, {b.right, b.top}} {
		x, y := m.point(corner[0], corner[1])
		out.left, out.right = min(out.left, x), max(out.right, x)
		out.bottom, out.top = min(out.bottom, y), max(out.top, y)
	}
	return out
}

// sized says a box is the size a figure is on this page: neither a thin rule nor the
// page's whole background.
func (p *pdfPage) sized(b box) bool {
	if b.width() < minFigureSide || b.height() < minFigureSide {
		return false
	}
	return p.width <= 0 || p.height <= 0 || b.width()*b.height() < backgroundShare*p.width*p.height
}

// written says some of the page's text is inside the box. A filled box with text in it is
// a table's shading, a highlighted note or a labelled box; one with none is a chart's bar.
func (p *pdfPage) written(b box) bool {
	for _, r := range p.runs {
		if b.overlaps(box{left: r.left, bottom: r.bottom, right: r.right, top: r.top}) {
			return true
		}
	}
	return false
}

// figured says the page's pictures and drawings cover figureShare of it or more, counting
// a part of the page covered twice once.
func (p *pdfPage) figured() bool {
	if len(p.drawn) == 0 || p.width <= 0 || p.height <= 0 {
		return false
	}
	var covered [figureGrid][figureGrid]bool
	count := 0
	cellWidth, cellHeight := p.width/figureGrid, p.height/figureGrid
	for _, b := range p.drawn {
		fromX, toX := cell(b.left, cellWidth), cell(b.right, cellWidth)
		fromY, toY := cell(b.bottom, cellHeight), cell(b.top, cellHeight)
		for x := fromX; x <= toX; x++ {
			for y := fromY; y <= toY; y++ {
				if !covered[x][y] {
					covered[x][y] = true
					count++
				}
			}
		}
	}
	return float64(count) >= figureShare*figureGrid*figureGrid
}

// cell is the grid cell a coordinate falls in, kept on the page.
func cell(at, size float64) int {
	return min(max(int(at/size), 0), figureGrid-1)
}
