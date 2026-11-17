package extract

import (
	"bytes"
	"crypto/sha256"
	"image"
	"image/gif"
	_ "image/jpeg"
	"image/png"
	"regexp"
	"strconv"
	"strings"
)

// Picture is a picture a document holds, as a chart or a photo pasted into a Word file is,
// which only reading the picture can say the content of.
type Picture struct {
	// MIME is image/png or image/jpeg, which is all a model is shown.
	MIME string
	Data []byte
	// Text is what reading the picture said, written where the picture is. Until it is
	// read the picture is left out of the document's text.
	Text string
}

// minPictureSide is the fewest pixels a side of a picture has for it to be read: an icon,
// a bullet or a rule drawn as a picture is smaller.
const minPictureSide = 100

// maxPictureSide and maxPictureBytes are the most a page read by a model may be, which is
// what ocr.Request.Validate allows.
const (
	maxPictureSide  = 10000
	maxPictureBytes = 10 << 20
)

// maxPictures is the most pictures of one document that are kept to be read.
const maxPictures = 500

const (
	pictureOpen  = "<!-- picture "
	pictureClose = " -->"
)

// PictureMarker is what stands in a document's text where its picture number n, from 1,
// is until the picture is read. It is an HTML comment, so it shows as nothing.
func PictureMarker(n int) string {
	return pictureOpen + strconv.Itoa(n) + pictureClose
}

var pictureMarker = regexp.MustCompile(`<!-- picture (\d+) -->`)

// pictures are the pictures of a document being read, in the order they are placed.
type pictures struct {
	kept []Picture
	// placed is the hash of every picture kept, so one placed again, as a logo in every
	// section is, is read once and written where it first is.
	placed map[[sha256.Size]byte]bool
}

// keep keeps a picture to be read and returns its marker. A picture that is too small to
// be a figure, in a format no model is shown, too large to send, placed before, or more
// than maxPictures is not kept, and its marker is empty.
func (p *pictures) keep(data []byte) string {
	if len(p.kept) >= maxPictures {
		return ""
	}
	hash := sha256.Sum256(data)
	if p.placed[hash] {
		return ""
	}
	mime, data, ok := readable(data)
	if !ok {
		return ""
	}
	if p.placed == nil {
		p.placed = map[[sha256.Size]byte]bool{}
	}
	p.placed[hash] = true
	p.kept = append(p.kept, Picture{MIME: mime, Data: data})
	return PictureMarker(len(p.kept))
}

// readable is a picture as a model is shown it, and whether it is one to read. A GIF is
// redrawn as a PNG. Word's own drawings, EMF and WMF, are in no format a model reads.
func readable(data []byte) (string, []byte, bool) {
	config, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || config.Width < minPictureSide || config.Height < minPictureSide ||
		config.Width > maxPictureSide || config.Height > maxPictureSide {
		return "", nil, false
	}
	switch format {
	case "jpeg":
		return "image/jpeg", data, len(data) <= maxPictureBytes
	case "png":
		return "image/png", data, len(data) <= maxPictureBytes
	case "gif":
		picture, err := gif.Decode(bytes.NewReader(data))
		if err != nil {
			return "", nil, false
		}
		var b bytes.Buffer
		if err := png.Encode(&b, picture); err != nil {
			return "", nil, false
		}
		return "image/png", b.Bytes(), b.Len() <= maxPictureBytes
	}
	return "", nil, false
}

// placePictures writes what each picture in text was read as where its marker is: in a
// table's cell as one line, elsewhere as it was read. A marker whose picture said nothing,
// or was never read, is taken out with the line it stood on.
func placePictures(text string, kept []Picture) string {
	if !strings.Contains(text, pictureOpen) {
		return text
	}
	lines := strings.Split(text, "\n")
	out := lines[:0]
	removed := false
	for _, line := range lines {
		inTable := strings.HasPrefix(strings.TrimSpace(line), "|")
		placed := pictureMarker.ReplaceAllStringFunc(line, func(marker string) string {
			n, err := strconv.Atoi(pictureMarker.FindStringSubmatch(marker)[1])
			if err != nil || n < 1 || n > len(kept) {
				return ""
			}
			said := strings.TrimSpace(kept[n-1].Text)
			if inTable {
				return tableCell(said)
			}
			return said
		})
		if placed != line && strings.TrimSpace(placed) == "" {
			removed = true
			continue
		}
		out = append(out, placed)
	}
	text = strings.Join(out, "\n")
	if removed {
		text = blankLines.ReplaceAllString(strings.TrimSpace(text), "\n\n")
	}
	return text
}

var blankLines = regexp.MustCompile(`\n{3,}`)
