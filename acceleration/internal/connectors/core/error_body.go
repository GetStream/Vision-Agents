package core

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"
)

// recordedBody is the body of a 4xx or a 5xx answer to a request whose context carries an
// Exchange. It reads nothing on its own: each read its reader makes is passed through as it
// came, and the first maxErrorExcerpt bytes of it are recorded on the Exchange
// (Exchange.ErrorBody), redacted. So the caller gets the answer when its headers come, as it
// would without an Exchange, and whatever reads the body (Scheme.Classify's read of a 401, a
// 403 or a 429, the MCP client's of a 400, the connection proxy's copy) records it as it goes.
//
// A reader that closes it before maxErrorExcerpt bytes or the end (the MCP client closes a
// 500's, 502's, 503's or 504's body unread, go-sdk v1.8.0 mcp/streamable.go checkResponse)
// has Close wait for the rest of that KiB, at most closeReadWait (Close).
type recordedBody struct {
	body     io.ReadCloser
	exchange *Exchange
	redact   redaction
	// deadline is the request's, zero when it has none.
	deadline time.Time
	// read is the start of the body read so far, at most maxErrorExcerpt bytes.
	read []byte
	// ended says the body returned an error, io.EOF included: nothing more comes. whole says
	// that error was io.EOF and read holds every byte before it; past says a byte came that
	// read has no room for.
	ended, whole, past bool
	closed             bool
}

// recordBody is body, of the answer to request, recording on exchange what is read of it.
func recordBody(request *http.Request, body io.ReadCloser, exchange *Exchange, redact redaction) io.ReadCloser {
	deadline, _ := request.Context().Deadline()
	return &recordedBody{body: body, exchange: exchange, redact: redact, deadline: deadline}
}

func (b *recordedBody) Read(p []byte) (int, error) {
	n, err := b.body.Read(p)
	b.keep(p[:n], err)
	return n, err
}

// keep records read, which the body returned with err, as far as maxErrorExcerpt.
func (b *recordedBody) keep(read []byte, err error) {
	took := min(maxErrorExcerpt-len(b.read), len(read))
	b.read = append(b.read, read[:took]...)
	b.past = b.past || took < len(read)
	if err != nil {
		b.ended, b.whole = true, err == io.EOF && !b.past
	}
	if took > 0 || err != nil {
		b.exchange.bodyRead(b.redact.apply(string(b.read), b.whole))
	}
}

// Close closes the body. One read short of maxErrorExcerpt bytes and of its end is first read
// on for at most closeReadWait, and never closer to the request's deadline than twice that.
//
// Why: the MCP client closes a 5xx's body unread, so without this read the log line of a 502
// whose page came at once would have no body; and a provider whose 5xx body stalls must not
// hold the call (a 500 that stalls was back at once, external_server, on 11102295). The margin
// before the deadline is net/http's own wait on a Close of a body read short
// (maxPostCloseReadTime, net/http/transport.go:2425 in go1.27.0) and as much again, so the call
// still ends before its deadline, which the session's dispatcher would read as outcome_unknown.
func (b *recordedBody) Close() error {
	if !b.closed && !b.ended && len(b.read) < maxErrorExcerpt {
		b.closed = true
		if closed, err := b.readRest(); closed {
			return err
		}
	}
	b.closed = true
	return b.body.Close()
}

// readRest reads on to maxErrorExcerpt bytes or the end of the body, for the wait Close allows.
// When that wait ends first it closes the body, which ends the read, and reports that it did,
// with Close's error.
//
// Why the read runs on a goroutine: a Read of a body that stalls returns only once the body is
// closed; it returns then, so the goroutine never outlives Close. net/http's HTTP/1 body gives
// up its connection on a Close before the end, which fails the Read in progress
// (bodyEOFSignal.Close's earlyCloseFn, persistConn.readLoop, net/http/transport.go:2565-2627 in
// go1.27.0); its HTTP/2 body breaks the pipe the Read waits on.
func (b *recordedBody) readRest() (bool, error) {
	wait := closeReadWait
	if !b.deadline.IsZero() {
		wait = min(wait, time.Until(b.deadline)-2*closeReadWait)
	}
	if wait <= 0 {
		return false, nil
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		buffer := make([]byte, maxErrorExcerpt-len(b.read))
		for !b.ended && len(b.read) < maxErrorExcerpt {
			n, err := b.body.Read(buffer[:maxErrorExcerpt-len(b.read)])
			b.keep(buffer[:n], err)
		}
	}()
	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case <-done:
		return false, nil
	case <-timer.C:
	}
	err := b.body.Close()
	<-done
	return true, err
}

// redacted is what a redaction puts in place of what it takes out.
const redacted = "[redacted]"

// redaction is what a provider's error body may echo of the request that must not be kept,
// in the order it is taken out: the path and query together before the path apart.
type redaction []string

// redactionOf is what of request and credential an error body must not keep: the request's
// path and query, together and apart, as sent and decoded, and each string in the
// credential's secret (AccessCredential.Secret). Example: Express's default 404 is
// "Cannot POST /primary/mcp", and a provider may answer "invalid token <the token>".
func redactionOf(request *http.Request, credential AccessCredential) redaction {
	found := []string{request.URL.RequestURI(), request.URL.EscapedPath(), request.URL.Path, request.URL.RawQuery}
	var secret any
	if json.Unmarshal(credential.Secret(), &secret) == nil {
		found = appendStrings(found, secret)
	}
	return redactionFrom(found)
}

// redactionFrom is the redaction of each of found longer than a byte, in order.
func redactionFrom(found []string) redaction {
	var r redaction
	for _, s := range found {
		// Why: a one-byte value, the path "/" of a URL with none, would take every such byte
		// out of the body, and a byte alone names nothing.
		if len(s) > 1 {
			r = append(r, s)
		}
	}
	return r
}

// appendStrings appends every string in value, a decoded JSON value, to found: a string, or
// the strings of an object, which is what every scheme's secret is (bearer's {"token"},
// oauth2cc's {"client_secret", "access_token", ...}).
func appendStrings(found []string, value any) []string {
	switch value := value.(type) {
	case string:
		return append(found, value)
	case map[string]any:
		for _, item := range value {
			found = appendStrings(found, item)
		}
	}
	return found
}

// apply is text with each of r replaced by redacted. Text that is not the whole body may end
// inside one of r, which is cut off: of a body cut at 1 KiB, the start of the token it echoes.
func (r redaction) apply(text string, whole bool) string {
	for _, s := range r {
		text = strings.ReplaceAll(text, s, redacted)
	}
	if whole {
		return text
	}
	cut := 0
	for _, s := range r {
		for k := min(len(s)-1, len(text)); k > cut; k-- {
			if strings.HasSuffix(text, s[:k]) {
				cut = k
				break
			}
		}
	}
	return text[:len(text)-cut]
}
