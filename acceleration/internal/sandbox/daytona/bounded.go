package daytona

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

// maxAnswerBytes bounds the whole of a toolbox answer the router reads: a run hands back
// nothing larger than its largest file, and an output cut at sandbox.MaxOutputBytes is far
// smaller.
const maxAnswerBytes = sandbox.MaxFileBytes

// downloadSlack is what a download's answer may hold past sandbox.MaxFileBytes: the one byte more
// that download reads of its file, the multipart framing, and an error part, which the SDK reads
// whole (sdk-go v0.211.2 pkg/daytona/file_transfer.go:92, :139). An upload's answer is far smaller.
const downloadSlack = 1 << 20

// maxHeldBytes is the most of an output's JSON string held before it is cut. An escape such as
// \u0000 is six bytes on the wire for one byte of output, so an output that fits
// sandbox.MaxOutputBytes is never more than six times it; the rest is room for one escape cut in
// two at the end.
const maxHeldBytes = 6*sandbox.MaxOutputBytes + 16

// errAnswerTooLarge is a toolbox answer past what the router reads of it, and errBadOutput one
// whose output, cut, is not a JSON string.
var (
	errAnswerTooLarge = errors.New("daytona: the sandbox's answer is too large to read")
	errBadOutput      = errors.New("daytona: the sandbox's answer holds an output that is not a JSON string")
)

// bounded bounds what the router reads of the toolbox's answers that the SDK reads whole: a code
// run's and a command's (toolbox-api-client-go v0.211.2 api_process.go, io.ReadAll before the
// JSON decode), a download's and an upload's (api_file_system.go).
type bounded struct {
	next http.RoundTripper
}

func (b bounded) RoundTrip(r *http.Request) (*http.Response, error) {
	answer, err := b.next.RoundTrip(r)
	if err != nil {
		return answer, err
	}
	switch {
	case strings.HasSuffix(r.URL.Path, "/process/code-run"), strings.HasSuffix(r.URL.Path, "/process/execute"):
		answer.Body = &cutOutput{body: answer.Body, left: maxAnswerBytes}
	case strings.HasSuffix(r.URL.Path, "/files/bulk-download"), strings.HasSuffix(r.URL.Path, "/files/upload-v2"):
		answer.Body = &capped{body: answer.Body, left: sandbox.MaxFileBytes + downloadSlack}
	}
	return answer, nil
}

// capped is a body that fails once more than left bytes were read of it.
type capped struct {
	body io.ReadCloser
	left int64
}

func (c *capped) Read(p []byte) (int, error) {
	n, err := c.body.Read(p[:min(int64(len(p)), c.left+1)])
	c.left -= int64(n)
	if c.left < 0 {
		return 0, errAnswerTooLarge
	}
	return n, err
}

func (c *capped) Close() error { return c.body.Close() }

// cutOutput is a toolbox answer whose top-level "result", what the code printed, is cut at
// sandbox.MaxOutputBytes of output with sandbox.OutputCut after it, and the rest of it is
// dropped unread. An output that fits passes byte for byte. Every other byte passes as it came,
// and an answer giving more than left bytes fails.
type cutOutput struct {
	body io.ReadCloser
	left int64
	read [32 << 10]byte
	// out is what was passed on and not yet read, from off.
	out []byte
	off int
	err error

	// depth is how deep in objects and arrays the next byte is. inString is a JSON string being
	// read and escaped a backslash just read in it. key is the start of the last string, while
	// inKey reads it; value says a colon came since the last comma, so a string of the top-level
	// object is the value of the key before that colon.
	depth    int
	inString bool
	escaped  bool
	inKey    bool
	key      []byte
	value    bool
	// output is the string of "result" being read, held up to maxHeldBytes, and dropping says
	// it was cut and the rest of it is skipped.
	output   bool
	held     []byte
	dropping bool
}

func (c *cutOutput) Read(p []byte) (int, error) {
	for len(c.out) == 0 && c.err == nil {
		n, err := c.body.Read(c.read[:])
		for _, b := range c.read[:n] {
			if c.err = c.step(b); c.err != nil {
				break
			}
		}
		if c.err == nil {
			c.err = err
		}
	}
	n := copy(p, c.out[c.off:])
	c.off += n
	if c.off < len(c.out) {
		return n, nil
	}
	c.out, c.off = c.out[:0], 0
	return n, c.err
}

func (c *cutOutput) Close() error { return c.body.Close() }

// step reads one byte of the answer.
func (c *cutOutput) step(b byte) error {
	if c.output {
		return c.outputByte(b)
	}
	if c.inString {
		switch {
		case c.escaped:
			c.escaped = false
		case b == '\\':
			c.escaped = true
		case b == '"':
			c.inString, c.inKey = false, false
		}
		if c.inKey && len(c.key) <= len("result") {
			c.key = append(c.key, b)
		}
		return c.emit(b)
	}
	switch b {
	case '"':
		if c.depth == 1 && c.value && string(c.key) == "result" {
			c.output, c.held = true, c.held[:0]
			return nil
		}
		c.inString, c.inKey, c.key = true, true, c.key[:0]
	case '{', '[':
		c.depth++
	case '}', ']':
		c.depth--
	case ':':
		c.value = true
	case ',':
		c.value = false
	}
	return c.emit(b)
}

// outputByte reads one byte of the output's string, after its opening quote.
func (c *cutOutput) outputByte(b byte) error {
	switch {
	case c.escaped:
		c.escaped = false
	case b == '\\':
		c.escaped = true
	case b == '"':
		c.output = false
		if c.dropping {
			c.dropping = false
			return nil
		}
		return c.settle(false)
	}
	if c.dropping {
		return nil
	}
	c.held = append(c.held, b)
	if len(c.held) < maxHeldBytes {
		return nil
	}
	c.dropping = true
	return c.settle(true)
}

// settle passes the output held as it came when it fits, and cut otherwise. over says more of it
// came than is held, so its end may be an escape or a character cut in two.
func (c *cutOutput) settle(over bool) error {
	var text string
	quoted := append(append([]byte{'"'}, c.held...), '"')
	err := json.Unmarshal(quoted, &text)
	if !over && len(text) <= sandbox.MaxOutputBytes {
		// An output that does not decode leaves text empty and passes, to fail the SDK's own
		// decode as it would unbounded.
		return c.emit(quoted...)
	}
	// An escape is at most six bytes, \uXXXX, so the held output decodes once at most five of its
	// last bytes are dropped.
	for end := len(c.held); err != nil && end > len(c.held)-5; end-- {
		quoted = append(append(quoted[:1], c.held[:end-1]...), '"')
		err = json.Unmarshal(quoted, &text)
	}
	if err != nil {
		return errBadOutput
	}
	// A string always encodes.
	cut, _ := json.Marshal(cutText(text))
	return c.emit(cut...)
}

// cutText is text when it fits sandbox.MaxOutputBytes, and otherwise as much of it as fits with
// sandbox.OutputCut after it, never splitting a UTF-8 sequence. The reader cuts a "result" it
// finds with it, and so does Run for the output the SDK decoded, which Go's decode takes from a
// key the reader does not know as "result": "Result", or one written with escapes.
func cutText(text string) string {
	if len(text) <= sandbox.MaxOutputBytes {
		return text
	}
	kept := text[:sandbox.MaxOutputBytes-len(sandbox.OutputCut)]
	for !utf8.ValidString(kept) {
		kept = kept[:len(kept)-1]
	}
	return kept + sandbox.OutputCut
}

// emit passes bytes on, and fails once more than left would have passed.
func (c *cutOutput) emit(b ...byte) error {
	c.left -= int64(len(b))
	if c.left < 0 {
		return errAnswerTooLarge
	}
	c.out = append(c.out, b...)
	return nil
}
