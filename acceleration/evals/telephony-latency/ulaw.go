package main

// ulawBias is added before encoding so the segment search can start at the bottom.
const ulawBias = 0x84

// ulawClip is the largest magnitude μ-law can carry once biased.
const ulawClip = 32635

// EncodeULaw is G.711 μ-law, the codec every US carrier answers a call with.
func EncodeULaw(samples []int16) []byte {
	encoded := make([]byte, len(samples))
	for i, sample := range samples {
		value := int(sample)
		sign := 0
		if value < 0 {
			value = -value
			sign = 0x80
		}
		value = min(value, ulawClip) + ulawBias

		exponent := 7
		for mask := 0x4000; value&mask == 0 && exponent > 0; mask >>= 1 {
			exponent--
		}
		mantissa := (value >> (exponent + 3)) & 0x0F
		encoded[i] = ^byte(sign | exponent<<4 | mantissa)
	}
	return encoded
}

// DecodeULaw turns G.711 μ-law back into PCM.
func DecodeULaw(encoded []byte) []int16 {
	samples := make([]int16, len(encoded))
	for i, b := range encoded {
		b = ^b
		exponent := int(b>>4) & 0x07
		mantissa := int(b & 0x0F)
		value := ((mantissa << 3) + ulawBias) << exponent
		value -= ulawBias
		if b&0x80 != 0 {
			value = -value
		}
		samples[i] = int16(value)
	}
	return samples
}
