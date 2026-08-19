package audio

const (
	ulawBias = 0x84
	ulawClip = 32635
)

var segEnd = [...]int{0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF}

// DecodeUlaw converts mu-law bytes to 16-bit PCM.
func DecodeUlaw(src []byte) []int16 {
	out := make([]int16, len(src))
	for i, value := range src {
		out[i] = decodeUlawByte(value)
	}
	return out
}

// EncodeUlaw converts 16-bit PCM to mu-law bytes.
func EncodeUlaw(src []int16) []byte {
	out := make([]byte, len(src))
	for i, sample := range src {
		out[i] = encodeUlawSample(int(sample))
	}
	return out
}

func decodeUlawByte(value byte) int16 {
	v := int(^value)
	sample := ((v & 0x0F) << 3) + ulawBias
	sample <<= (v & 0x70) >> 4
	sample -= ulawBias
	if v&0x80 != 0 {
		return int16(-sample)
	}
	return int16(sample)
}

func encodeUlawSample(sample int) byte {
	var mask int
	if sample < 0 {
		sample = ulawBias - sample
		mask = 0x7F
	} else {
		sample += ulawBias
		mask = 0xFF
	}
	if sample > ulawClip {
		sample = ulawClip
	}
	segment := searchSegment(sample)
	if segment >= 8 {
		return byte(0x7F ^ mask)
	}
	encoded := (segment << 4) | ((sample >> (segment + 3)) & 0x0F)
	return byte(encoded ^ mask)
}

func searchSegment(sample int) int {
	for i, end := range segEnd {
		if sample <= end {
			return i
		}
	}
	return 8
}
