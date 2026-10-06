package api

import (
	"bytes"
	"compress/gzip"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
)

// What a Stream delivery may be on the wire, and what it may inflate to. Both are checked
// before the signature: verifying means reading, and reading without a bound is the
// cheapest way there is to take this process down. A real event is a few kilobytes.
const (
	hookBodyLimit    = 1 << 20
	hookPayloadLimit = 4 << 20
)

// gzipMagic opens every gzip stream, which is how a compressed delivery is told apart.
var gzipMagic = []byte{0x1f, 0x8b}

// errTooLarge is a delivery past one of the limits.
var errTooLarge = errors.New("api: that delivery is larger than any event Stream sends")

// isHook reports whether a path is one Stream delivers events to. Stream is not a
// customer, so nothing a request to one of them says about who is calling is read.
func isHook(path string) bool {
	for _, hook := range []string{phone.CallHookPath, chat.MessageHookPath} {
		if path == hook || strings.HasPrefix(path, hook+"/") {
			return true
		}
	}
	return false
}

// readHook reads a Stream delivery, inflated when it was compressed, within the limits.
// It answers the request itself when it cannot, and says whether there is a payload to go
// on with. what names the event in the answer.
func readHook(w http.ResponseWriter, r *http.Request, what string) ([]byte, bool) {
	switch strings.ToLower(strings.TrimSpace(r.Header.Get("Content-Encoding"))) {
	case "", "identity", "gzip":
	default:
		http.Error(w, "that "+what+" is in an encoding Stream does not send", http.StatusUnsupportedMediaType)
		return nil, false
	}

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, hookBodyLimit))
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		http.Error(w, "that "+what+" is too large", http.StatusRequestEntityTooLarge)
		return nil, false
	}
	if err != nil {
		http.Error(w, "could not read that "+what, http.StatusBadRequest)
		return nil, false
	}
	// Deliveries may be compressed, and the signature is over what is inside.
	payload, err := inflate(body)
	if errors.Is(err, errTooLarge) {
		http.Error(w, "that "+what+" is too large", http.StatusRequestEntityTooLarge)
		return nil, false
	}
	if err != nil {
		http.Error(w, "that is not a "+what+" from Stream", http.StatusUnauthorized)
		return nil, false
	}
	return payload, true
}

// inflate returns a compressed delivery's contents, refusing to read past the limit.
func inflate(body []byte) ([]byte, error) {
	if !bytes.HasPrefix(body, gzipMagic) {
		return body, nil
	}
	reader, err := gzip.NewReader(bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer reader.Close() //nolint:errcheck // nothing was written
	payload, err := io.ReadAll(io.LimitReader(reader, hookPayloadLimit+1))
	if err != nil {
		return nil, err
	}
	if len(payload) > hookPayloadLimit {
		return nil, errTooLarge
	}
	return payload, nil
}
