package main

import (
	"math"
	"time"
)

// sampleRate is the narrowband rate every carrier in the US carries a call at.
const sampleRate = 8000

// symbolSamples is how long one tone lasts: 30 ms, long enough for a tone to be told from
// its neighbours 160 Hz away, short enough that a stamp is over in a third of a second.
const symbolSamples = 240

// toneLevel is the amplitude of a stamp out of 32767: loud enough to clear a carrier's
// comfort noise after the attenuation of two networks, quiet enough not to clip or trip
// an automatic gain control.
const toneLevel = 9000

// rampSamples fade each tone in and out over 2 ms, so a symbol boundary is not a click
// that smears energy into every other tone's bin.
const rampSamples = 16

// onsetLevel is the sample amplitude that counts as a stamp starting. It sits well above
// the noise floor of a quiet call and well below a stamp that has lost 12 dB on the way.
const onsetLevel = 1200

// quietBefore is how much silence has to come before an onset, so the tail of one stamp
// or a burst of line noise is not taken for the start of another.
const quietBefore = 400

// syncSymbols open every stamp with the highest tone and then the lowest, a pattern a
// misread onset is unlikely to produce by chance.
var syncSymbols = [2]int{15, 0}

// dataSymbols carry the 32-bit millisecond, four bits per tone.
const dataSymbols = 8

// crcSymbols carry an 8-bit checksum over the millisecond.
const crcSymbols = 2

// stampSymbols is the length of a whole stamp.
const stampSymbols = len(syncSymbols) + dataSymbols + crcSymbols

// StampSamples is how many samples a stamp takes: 360 ms.
const StampSamples = stampSymbols * symbolSamples

// toneHz is the frequency of each of the sixteen symbols, 600 to 3000 Hz, inside the
// 300 to 3400 Hz every telephone network passes.
func toneHz(symbol int) float64 { return 600 + 160*float64(symbol) }

// Millis is a moment as the low 32 bits of its Unix milliseconds, which is what a stamp
// carries. It wraps every 49 days, which Resolve undoes.
func Millis(t time.Time) uint32 { return uint32(t.UnixMilli()) }

// Resolve turns the millisecond a stamp carried back into a time, using the moment it was
// heard to say which 49-day window it came from. A stamp is never older than that.
func Resolve(sent uint32, heard time.Time) time.Time {
	elapsed := uint32(heard.UnixMilli()) - sent
	return time.UnixMilli(heard.UnixMilli() - int64(elapsed))
}

// Stamp is the audio that carries a millisecond: a sync pattern, the millisecond four bits
// to a tone, most significant first, and a checksum.
func Stamp(millis uint32) []int16 {
	symbols := make([]int, 0, stampSymbols)
	symbols = append(symbols, syncSymbols[:]...)
	for shift := 28; shift >= 0; shift -= 4 {
		symbols = append(symbols, int(millis>>shift)&0xF)
	}
	sum := crc8(millis)
	symbols = append(symbols, int(sum>>4), int(sum&0xF))

	samples := make([]int16, 0, StampSamples)
	for _, symbol := range symbols {
		step := 2 * math.Pi * toneHz(symbol) / sampleRate
		for i := range symbolSamples {
			gain := 1.0
			if i < rampSamples {
				gain = 0.5 - 0.5*math.Cos(math.Pi*float64(i)/rampSamples)
			} else if i >= symbolSamples-rampSamples {
				gain = 0.5 - 0.5*math.Cos(math.Pi*float64(symbolSamples-1-i)/rampSamples)
			}
			samples = append(samples, int16(toneLevel*gain*math.Sin(step*float64(i))))
		}
	}
	return samples
}

// Heard is a stamp that came off a call.
type Heard struct {
	// Sent is the millisecond the stamp carried.
	Sent uint32
	// At is when the packet holding the stamp's first sample arrived.
	At time.Time
}

// Latency is how long the stamp took to arrive.
func (h Heard) Latency() time.Duration { return h.At.Sub(Resolve(h.Sent, h.At)) }

// Detector listens to one direction of a call and reads the stamps in it.
//
// It finds a stamp by its onset, the first loud sample after a stretch of quiet, rather
// than by correlating against what was sent: the time of that sample's arrival is the
// measurement, and the tones after it only have to say which stamp it was.
type Detector struct {
	// buffered holds the samples of a stamp being read, from its onset.
	buffered []int16
	// onsetAt is when the onset arrived, zero while waiting for one.
	onsetAt time.Time
	// quiet counts the samples since the last loud one.
	quiet int
	// garbled counts stamps whose tones did not decode.
	garbled int
}

// Push feeds the detector the samples of one packet and the time its first sample arrived,
// and returns any stamps the packet completed. A sample further into the packet is due
// later by its place in it, so a carrier that repacketises off the sender's boundaries does
// not make a stamp look early by up to a packet.
func (d *Detector) Push(samples []int16, at time.Time) []Heard {
	var heard []Heard
	for i, sample := range samples {
		if !d.onsetAt.IsZero() {
			d.buffered = append(d.buffered, sample)
			if len(d.buffered) == StampSamples {
				if sent, ok := decode(d.buffered); ok {
					heard = append(heard, Heard{Sent: sent, At: d.onsetAt})
				} else {
					d.garbled++
				}
				d.buffered = d.buffered[:0]
				d.onsetAt = time.Time{}
				d.quiet = 0
			}
			continue
		}

		if sample > onsetLevel || sample < -onsetLevel {
			if d.quiet >= quietBefore {
				d.onsetAt = at.Add(time.Duration(i) * time.Second / sampleRate)
				d.buffered = append(d.buffered[:0], sample)
			}
			d.quiet = 0
			continue
		}
		d.quiet++
	}
	return heard
}

// Garbled is how many stamps started but did not decode.
func (d *Detector) Garbled() int { return d.garbled }

// decode reads a stamp's tones, taking each symbol's middle so an onset found a few
// milliseconds late or early still lands inside the right tone.
func decode(samples []int16) (uint32, bool) {
	const margin = 40
	symbols := make([]int, stampSymbols)
	for s := range stampSymbols {
		window := samples[s*symbolSamples+margin : (s+1)*symbolSamples-margin]
		best, bestPower := 0, -1.0
		for symbol := range 16 {
			if power := goertzel(window, toneHz(symbol)); power > bestPower {
				best, bestPower = symbol, power
			}
		}
		symbols[s] = best
	}

	if symbols[0] != syncSymbols[0] || symbols[1] != syncSymbols[1] {
		return 0, false
	}
	var millis uint32
	for _, symbol := range symbols[2 : 2+dataSymbols] {
		millis = millis<<4 | uint32(symbol)
	}
	sum := uint8(symbols[2+dataSymbols]<<4 | symbols[3+dataSymbols])
	if sum != crc8(millis) {
		return 0, false
	}
	return millis, true
}

// goertzel is the power of one frequency in a window, which is all a tone detector needs
// of a Fourier transform.
func goertzel(samples []int16, hz float64) float64 {
	coefficient := 2 * math.Cos(2*math.Pi*hz/sampleRate)
	var previous, before float64
	for _, sample := range samples {
		current := float64(sample) + coefficient*previous - before
		before, previous = previous, current
	}
	return previous*previous + before*before - coefficient*previous*before
}

// crc8 is CRC-8/SMBUS over the millisecond's four bytes. Without it a single misread tone
// would be a latency off by up to four billion milliseconds rather than a garbled stamp.
func crc8(millis uint32) uint8 {
	var sum uint8
	for shift := 24; shift >= 0; shift -= 8 {
		sum ^= uint8(millis >> shift)
		for range 8 {
			if sum&0x80 != 0 {
				sum = sum<<1 ^ 0x07
			} else {
				sum <<= 1
			}
		}
	}
	return sum
}
