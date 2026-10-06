package agent

import (
	"encoding/binary"
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestEOTScoringSnapshotCapturesTimingAndTailStatistics(t *testing.T) {
	samples := make([]int16, eotSampleRate)
	for i := 0; i < eotSampleRate/2; i++ {
		samples[i] = 100
	}
	for i := eotSampleRate / 2; i < eotSampleRate-1600; i++ {
		samples[i] = 200
	}
	for i := eotSampleRate - 1600; i < eotSampleRate; i++ {
		if i%2 != 0 {
			samples[i] = -300
		}
	}

	receivedAt := time.Now().Add(-1250 * time.Millisecond)
	r := newPCM16LERing()
	r.appendTimed(samples, AudioTiming{
		Valid:            true,
		PTS:              0,
		ReceivedAt:       receivedAt,
		Epoch:            11,
		TimestampGap:     31 * time.Millisecond,
		TimestampOnlyGap: 17 * time.Millisecond,
		SequenceLoss:     4,
		ClockResets:      2,
		AmbiguousGaps:    3,
		Overlap:          9 * time.Millisecond,
	})

	snapshot, ok := r.scoringSnapshot()
	require.True(t, ok)
	require.Len(t, snapshot.pcm, len(samples)*2)
	diagnostics := snapshot.claim()
	require.True(t, diagnostics.timingValid)
	require.Zero(t, diagnostics.timing.PTS, "zero is a valid source PTS")
	require.Equal(t, uint64(11), diagnostics.timing.Epoch)
	require.Equal(t, eotSampleRate, diagnostics.samples)
	require.Equal(t, uint64(1), diagnostics.generation)
	require.Equal(t, uint64(1), diagnostics.generationAdvance)
	require.False(t, diagnostics.snapshotGenerationUnchanged)
	require.True(t, diagnostics.sourceAgeValid)
	require.GreaterOrEqual(t, diagnostics.sourceAge, 1200*time.Millisecond)
	require.True(t, diagnostics.appendAgeValid)
	require.Less(t, diagnostics.appendAge, time.Second)
	require.Equal(t, 31*time.Millisecond, diagnostics.timestampGapDelta)
	require.Equal(t, 17*time.Millisecond, diagnostics.timestampOnlyGapDelta)
	require.Equal(t, uint64(4), diagnostics.sequenceLossDelta)
	require.Equal(t, uint64(2), diagnostics.clockResetsDelta)
	require.Equal(t, uint64(3), diagnostics.ambiguousGapsDelta)
	require.Equal(t, 9*time.Millisecond, diagnostics.overlapDelta)

	require.Equal(t, 1600, diagnostics.tail100ms.samples)
	require.InDelta(t, math.Sqrt(45000), diagnostics.tail100ms.rms, 0.001)
	require.Equal(t, 300, diagnostics.tail100ms.peak)
	require.InDelta(t, 0.5, diagnostics.tail100ms.zeroFraction, 0.0001)
	require.Equal(t, 8000, diagnostics.tail500ms.samples)
	require.InDelta(t, math.Sqrt(41000), diagnostics.tail500ms.rms, 0.001)
	require.Equal(t, 300, diagnostics.tail500ms.peak)
	require.InDelta(t, 0.1, diagnostics.tail500ms.zeroFraction, 0.0001)
	require.Equal(t, 16000, diagnostics.tail1000ms.samples)
	require.InDelta(t, math.Sqrt(25500), diagnostics.tail1000ms.rms, 0.001)
	require.Equal(t, 300, diagnostics.tail1000ms.peak)
	require.InDelta(t, 0.05, diagnostics.tail1000ms.zeroFraction, 0.0001)

	decoded := decodeEOTPCM(snapshot.pcm)
	require.Equal(t, samples, decoded, "the diagnostic snapshot must retain every sample, including zeros")
	snapshot.pcm[0] ^= 0xff
	require.Equal(t, samples, decodeEOTPCM(r.snapshot()), "the owned scoring copy must not alias ring storage")
}

func TestEOTScoringSnapshotTracksGenerationAndEpochDeltas(t *testing.T) {
	r := newPCM16LERing()
	base := AudioTiming{
		Valid:            true,
		ReceivedAt:       time.Now(),
		Epoch:            100,
		TimestampGap:     100 * time.Millisecond,
		TimestampOnlyGap: 40 * time.Millisecond,
		SequenceLoss:     12,
		ClockResets:      1,
		AmbiguousGaps:    5,
		Overlap:          50 * time.Millisecond,
	}
	r.appendTimed(make([]int16, eotMinSamples), base)
	first, ok := r.scoringSnapshot()
	require.True(t, ok)
	firstDiagnostics := first.claim()
	require.Equal(t, base.TimestampGap, firstDiagnostics.timestampGapDelta)
	require.Equal(t, base.TimestampOnlyGap, firstDiagnostics.timestampOnlyGapDelta)
	require.Equal(t, base.SequenceLoss, firstDiagnostics.sequenceLossDelta)

	continued := base
	continued.ReceivedAt = time.Now()
	continued.TimestampGap += 7 * time.Millisecond
	continued.TimestampOnlyGap += 3 * time.Millisecond
	continued.SequenceLoss++
	continued.ClockResets++
	continued.AmbiguousGaps += 2
	continued.Overlap += 4 * time.Millisecond
	r.appendTimed([]int16{0}, continued)
	second, ok := r.scoringSnapshot()
	require.True(t, ok)
	secondDiagnostics := second.claim()
	require.Equal(t, uint64(1), secondDiagnostics.generationAdvance)
	require.False(t, secondDiagnostics.snapshotGenerationUnchanged)
	require.Equal(t, 7*time.Millisecond, secondDiagnostics.timestampGapDelta)
	require.Equal(t, 3*time.Millisecond, secondDiagnostics.timestampOnlyGapDelta)
	require.Equal(t, uint64(1), secondDiagnostics.sequenceLossDelta)
	require.Equal(t, uint64(1), secondDiagnostics.clockResetsDelta)
	require.Equal(t, uint64(2), secondDiagnostics.ambiguousGapsDelta)
	require.Equal(t, 4*time.Millisecond, secondDiagnostics.overlapDelta)

	unchanged, ok := r.scoringSnapshot()
	require.True(t, ok)
	unchangedDiagnostics := unchanged.claim()
	require.Zero(t, unchangedDiagnostics.generationAdvance)
	require.True(t, unchangedDiagnostics.snapshotGenerationUnchanged)
	require.Zero(t, unchangedDiagnostics.timestampGapDelta)
	require.Zero(t, unchangedDiagnostics.sequenceLossDelta)

	// The counters on a replacement track start at zero. A new track identity rebases
	// deltas so its first measurements are not hidden by the previous track's totals.
	r.clear()
	newTrack := AudioTiming{
		Valid:         true,
		ReceivedAt:    time.Now(),
		Epoch:         101,
		TimestampGap:  2 * time.Millisecond,
		SequenceLoss:  2,
		ClockResets:   1,
		AmbiguousGaps: 1,
		Overlap:       time.Millisecond,
	}
	r.appendTimed(make([]int16, eotMinSamples), newTrack)
	newSnapshot, ok := r.scoringSnapshot()
	require.True(t, ok)
	newDiagnostics := newSnapshot.claim()
	require.Equal(t, uint64(2), newDiagnostics.generationAdvance, "clear and append each mutate the ring")
	require.False(t, newDiagnostics.snapshotGenerationUnchanged)
	require.Equal(t, newTrack.TimestampGap, newDiagnostics.timestampGapDelta)
	require.Equal(t, newTrack.SequenceLoss, newDiagnostics.sequenceLossDelta)
	require.Equal(t, newTrack.ClockResets, newDiagnostics.clockResetsDelta)
	require.Equal(t, newTrack.AmbiguousGaps, newDiagnostics.ambiguousGapsDelta)
	require.Equal(t, newTrack.Overlap, newDiagnostics.overlapDelta)
}

func TestEOTScoringSnapshotOwnsItsPCMAndMetadataAtCopyTime(t *testing.T) {
	r := newPCM16LERing()
	initial := make([]int16, eotMinSamples)
	for i := range initial {
		initial[i] = int16(i - 160)
	}
	initialTiming := AudioTiming{
		Valid:        true,
		PTS:          20 * time.Millisecond,
		ReceivedAt:   time.Now().Add(-time.Second),
		Epoch:        44,
		SequenceLoss: 2,
	}
	r.appendTimed(initial, initialTiming)
	snapshot, ok := r.scoringSnapshot()
	require.True(t, ok)

	// A later edge chunk mutates the ring, but the model's owned PCM and its
	// diagnostics continue to describe the exact earlier scoring snapshot.
	r.appendTimed([]int16{3000}, AudioTiming{
		Valid:        true,
		PTS:          40 * time.Millisecond,
		ReceivedAt:   time.Now(),
		Epoch:        44,
		SequenceLoss: 3,
	})
	diagnostics := snapshot.claim()
	require.Equal(t, uint64(1), diagnostics.generation)
	require.Equal(t, eotMinSamples, diagnostics.samples)
	require.Equal(t, initialTiming, diagnostics.timing)
	require.Equal(t, initial, decodeEOTPCM(snapshot.pcm))
	require.Len(t, r.snapshot(), (eotMinSamples+1)*2)
}

func TestEOTUntimedAppendDoesNotReusePreviousSourceAge(t *testing.T) {
	r := newPCM16LERing()
	r.appendTimed(make([]int16, eotMinSamples), AudioTiming{
		Valid:      true,
		ReceivedAt: time.Now().Add(-2 * time.Second),
		Epoch:      7,
	})
	r.append(make([]int16, 1))

	snapshot, ok := r.scoringSnapshot()
	require.True(t, ok)
	diagnostics := snapshot.claim()
	require.False(t, diagnostics.timingValid)
	require.False(t, diagnostics.sourceAgeValid)
	require.True(t, diagnostics.appendAgeValid)
	require.Zero(t, diagnostics.timing)
}

func TestRetainEOTAudioKeepsZeroSamplesAndRejectsWrongFormat(t *testing.T) {
	a := &Agent{
		options:      Options{EOT: &EOTClient{}},
		audioHistory: make(map[string]*pcm16leRing),
	}
	samples := make([]int16, eotMinSamples)
	for i := range samples {
		if i%4 == 0 {
			samples[i] = 1234
		}
	}
	a.retainEOTAudioTimed("participant", eotSampleRate, 1, samples, AudioTiming{})
	a.retainEOTAudioTimed("participant", eotSampleRate/2, 1, samples, AudioTiming{})
	a.retainEOTAudioTimed("participant", eotSampleRate, 2, samples, AudioTiming{})

	got := a.eotAudioSnapshot("participant")
	require.Len(t, got, len(samples)*2)
	require.Equal(t, samples, decodeEOTPCM(got), "zero-valued frames remain in the scoring window; no VAD filtering is applied")
}

func TestPCM16LERingAppendTimedDoesNotAllocatePerChunk(t *testing.T) {
	r := newPCM16LERing()
	chunk := make([]int16, 320)
	timing := AudioTiming{Valid: true, Epoch: 1, ReceivedAt: time.Now()}
	allocs := testing.AllocsPerRun(100, func() {
		r.appendTimed(chunk, timing)
	})
	require.Zero(t, allocs)
}

func decodeEOTPCM(pcm []byte) []int16 {
	if len(pcm)%2 != 0 {
		panic("odd PCM byte length")
	}
	samples := make([]int16, len(pcm)/2)
	for i := range samples {
		samples[i] = int16(binary.LittleEndian.Uint16(pcm[i*2:]))
	}
	return samples
}
