package audio

import "math"

// Levelling raises a phrase a voice rendered far quieter than the rest of its speech. An
// expressive model can drop a phrase 20 to 30 dB below the one before it. A caller on a phone
// line hears that as the agent trailing off, and a caller's own voice detection hears it as
// the agent having finished, and talks over the rest.
//
// It only ever raises: speech at or above the target is left as the voice made it, so one
// provider does not come out louder than another, and nothing is taken away to make room.
const (
	// levelTargetDB is the speech level a quiet phrase is raised towards, as the RMS of a
	// 10 ms block. Ordinary speech from the voices here sits around -26 dBFS.
	levelTargetDB = -24.0
	// levelMaxGainDB caps the boost, so a breath or a whisper the voice meant is lifted to
	// a quiet phrase rather than to a shout.
	levelMaxGainDB = 18.0
	// levelFloorDB is the block level below which audio is a pause, not speech. A pause is
	// passed through as it is, since raising it would bring up its noise, and the speech
	// level estimate is held across it, so the phrase it interrupts is raised again as soon
	// as it resumes.
	levelFloorDB = -55.0
	// levelAttackMs and levelReleaseMs are how fast the speech level estimate follows the
	// voice up and down. It follows a louder phrase at once, so the gain drops before the
	// phrase can clip, and a quieter one slowly, so the dip of a consonant inside a word
	// does not pump the gain.
	levelAttackMs  = 20.0
	levelReleaseMs = 400.0
	// levelPeakLimit is the highest a sample may be raised to, as a fraction of full scale.
	levelPeakLimit = 0.89
	// levelBlockMs is how much audio is measured at a time.
	levelBlockMs = 10
)

// Leveller raises quiet phrases in a stream of speech, a chunk at a time, without holding
// any of it back. It is not safe for concurrent use; one stream has one leveller.
type Leveller struct {
	// level is the running estimate of the speech level, in dBFS. Zero means none yet.
	level float64
	// gain is the linear gain the last block ended on, which the next one starts from.
	gain float64
}

// NewLeveller returns a leveller for one stream of speech.
func NewLeveller() *Leveller {
	return &Leveller{gain: 1}
}

// Level returns pcm with its quiet phrases raised. The chunk passed in is left untouched.
func (l *Leveller) Level(pcm PcmData) PcmData {
	if len(pcm.Samples) == 0 || pcm.SampleRate <= 0 {
		return pcm
	}
	channels := max(pcm.Channels, 1)
	block := max(pcm.SampleRate*levelBlockMs/1000, 1) * channels
	out := PcmData{
		Samples:    make([]int16, len(pcm.Samples)),
		SampleRate: pcm.SampleRate,
		Channels:   pcm.Channels,
	}
	for start := 0; start < len(pcm.Samples); start += block {
		end := min(start+block, len(pcm.Samples))
		l.levelBlock(pcm.Samples[start:end], out.Samples[start:end], float64(end-start)/float64(block))
	}
	return out
}

// levelBlock raises one block into out, ramping from the gain the last block ended on to
// this one's, so the gain never steps within the audio. share is how much of a full block
// it is, for the time constants.
func (l *Leveller) levelBlock(in, out []int16, share float64) {
	var sum float64
	var peak float64
	for _, sample := range in {
		value := float64(sample) / 32768
		sum += value * value
		peak = max(peak, math.Abs(value))
	}
	blockDB := 10 * math.Log10(sum/float64(len(in))+1e-12)

	target := 1.0
	if blockDB > levelFloorDB {
		if l.level == 0 {
			l.level = blockDB
		} else {
			timeMs := levelReleaseMs
			if blockDB > l.level {
				timeMs = levelAttackMs
			}
			coefficient := 1 - math.Exp(-levelBlockMs*share/timeMs)
			l.level += (blockDB - l.level) * coefficient
		}
		gainDB := min(max(levelTargetDB-l.level, 0), levelMaxGainDB)
		target = math.Pow(10, gainDB/20)
	}
	// A block whose peak would clip is raised no further than its peak allows, from its
	// first sample: the ramp down from a quieter block's gain must not clip on the way.
	ceiling := math.Inf(1)
	if peak > 0 {
		ceiling = max(levelPeakLimit/peak, 1)
	}
	target = min(target, ceiling)

	from := l.gain
	for i, sample := range in {
		gain := min(from+(target-from)*float64(i+1)/float64(len(in)), ceiling)
		out[i] = clamp(float64(sample) * gain)
	}
	l.gain = target
}
