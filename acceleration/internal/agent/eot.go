package agent

import (
	"encoding/binary"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt/audioturn"
)

const (
	eotSampleRate = audioturn.SampleRate
	eotMaxSamples = audioturn.MaxSamples
	eotMinSamples = audioturn.MinSamples
)

// EOTMode determines whether an acoustic score gates the semantic flow controller or
// answers eligible quiet-floor completion candidates directly.
type EOTMode string

const (
	EOTModeGate    EOTMode = "gate"
	EOTModePrimary EOTMode = "primary"
)

func (mode EOTMode) valid() bool { return mode == EOTModeGate || mode == EOTModePrimary }

// pcm16leRing retains the last 16 seconds of audio. Scoring takes an owned copy;
// generation prevents retries from scoring the same audio again.
type pcm16leRing struct {
	mu                   sync.Mutex
	sample               []int16
	next                 int
	full                 bool
	generation           uint64
	lastScoredGeneration uint64
}

type eotScoringSnapshot struct {
	pcm        []byte
	ring       *pcm16leRing
	generation uint64
}

func newPCM16LERing() *pcm16leRing {
	return &pcm16leRing{sample: make([]int16, eotMaxSamples)}
}

func (r *pcm16leRing) append(samples []int16) {
	if len(samples) == 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, sample := range samples {
		r.sample[r.next] = sample
		r.next++
		if r.next == len(r.sample) {
			r.next = 0
			r.full = true
		}
	}
	r.generation++
}

func (r *pcm16leRing) snapshot() []byte {
	snapshot, _ := r.scoringSnapshot()
	return snapshot.pcm
}

func (r *pcm16leRing) scoringSnapshot() (eotScoringSnapshot, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	count, start := r.next, 0
	if r.full {
		count, start = len(r.sample), r.next
	}
	if count < eotMinSamples {
		return eotScoringSnapshot{}, false
	}
	pcm := make([]byte, count*2)
	for i := range count {
		binary.LittleEndian.PutUint16(pcm[i*2:], uint16(r.sample[(start+i)%len(r.sample)]))
	}
	return eotScoringSnapshot{pcm: pcm, ring: r, generation: r.generation}, true
}

func (r *pcm16leRing) clear() {
	r.mu.Lock()
	defer r.mu.Unlock()
	clear(r.sample)
	r.next, r.full = 0, false
	r.generation++
}
