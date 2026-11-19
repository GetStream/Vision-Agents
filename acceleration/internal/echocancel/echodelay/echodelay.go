// Package echodelay bounds how late an echo of the agent's voice is looked for, apart from the
// canceller itself, so the router's settings check it without building one.
package echodelay

import (
	"fmt"
	"time"
)

const (
	// Min is one of AEC3's matched filters, the shortest delay it can look across: 32 blocks
	// of 4 ms (kMatchedFilterWindowSizeSubBlocks in WebRTC's aec3_common.h).
	Min = 128 * time.Millisecond
	// Max is half again the 800 ms a voice takes to come back as echo once it has gone out
	// (the agent's echoDelay, a round trip of the 400 ms one-way delay ITU-T G.114 sets as the
	// upper limit for network planning). Each 96 ms more is another matched filter to run.
	Max = 1200 * time.Millisecond
	// Default covers that 800 ms, and the agent's own jitter buffer and decoding on top.
	Default = time.Second
)

// ErrOutOfBounds is a delay outside Min to Max.
var ErrOutOfBounds = fmt.Errorf("echodelay: the longest echo delay must be from %s to %s", Min, Max)

// Check reports whether maxDelay is one an echo can be looked for across.
func Check(maxDelay time.Duration) error {
	if maxDelay < Min || maxDelay > Max {
		return ErrOutOfBounds
	}
	return nil
}
