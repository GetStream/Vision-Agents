package echocancel

import (
	"fmt"
	"testing"
	"time"
)

// BenchmarkLine is the cost of one 10 ms frame of a line: the voice played and the line
// cleaned, with the echo looked for up to each longest delay.
func BenchmarkLine(b *testing.B) {
	for _, delay := range []time.Duration{512 * time.Millisecond, DefaultMaxDelay, MaxMaxDelay} {
		b.Run(fmt.Sprint(delay), func(b *testing.B) {
			call, err := NewCall(Config{MaxDelay: delay})
			if err != nil {
				b.Fatal(err)
			}
			line, err := call.Listen()
			if err != nil {
				b.Fatal(err)
			}
			defer line.Close()
			voice, heard := speech(100*FrameSamples, 1), speech(100*FrameSamples, 2)
			in := make([]int16, FrameSamples)
			b.ReportAllocs()
			for i := 0; b.Loop(); i++ {
				at := i % 100 * FrameSamples
				call.Played(voice[at : at+FrameSamples])
				copy(in, heard[at:at+FrameSamples])
				line.Clean(in)
			}
		})
	}
}
