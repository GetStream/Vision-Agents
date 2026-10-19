package openaicompat

import (
	"encoding/binary"
	"math"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const (
	realtimeFrameSamples  = stt.SampleRate / 50 // 20 ms keeps endpointing independent of caller chunk sizes.
	realtimeSilenceFrames = 30
	realtimeSpeechRMS     = 500
	realtimePreRollFrames = 10
)

type endpointFrame struct {
	samples     []int16
	participant stt.Participant
	utterance   int64
	include     bool
	commit      bool
	turnSamples int
}

type endpointer struct {
	pending            [realtimeFrameSamples]int16
	pendingSamples     int
	pendingParticipant stt.Participant
	preRoll            [realtimePreRollFrames][realtimeFrameSamples]int16
	preRollLengths     [realtimePreRollFrames]int
	preRollNext        int
	preRollCount       int
	preRollParticipant stt.Participant
	inTurn             bool
	silenceFrames      int
	turnSamples        int
	participant        stt.Participant
	utterance          int64
}

func (e *endpointer) Push(samples []int16, participant stt.Participant, emit func(endpointFrame)) {
	for len(samples) > 0 {
		if e.pendingSamples == 0 {
			e.pendingParticipant = participant
		}
		count := min(len(samples), realtimeFrameSamples-e.pendingSamples)
		copy(e.pending[e.pendingSamples:], samples[:count])
		e.pendingSamples += count
		samples = samples[count:]
		if e.pendingSamples == realtimeFrameSamples {
			e.process(e.pending[:], e.pendingParticipant, emit)
			e.pendingSamples = 0
		}
	}
}

func (e *endpointer) Close(emit func(endpointFrame)) {
	if e.pendingSamples > 0 {
		e.process(e.pending[:e.pendingSamples], e.pendingParticipant, emit)
		e.pendingSamples = 0
	}
	if e.inTurn {
		emit(endpointFrame{
			participant: e.participant,
			utterance:   e.utterance,
			commit:      true,
			turnSamples: e.turnSamples,
		})
		e.inTurn = false
		e.turnSamples = 0
		e.silenceFrames = 0
	}
}

func (e *endpointer) process(samples []int16, participant stt.Participant, emit func(endpointFrame)) {
	if !e.inTurn {
		if e.preRollCount > 0 && e.preRollParticipant != participant {
			e.preRollCount = 0
		}
		e.preRollParticipant = participant
	}
	active := speech(samples)
	if !e.inTurn && active {
		e.inTurn = true
		e.participant = participant
		e.utterance++
		e.turnSamples = 0
		e.silenceFrames = 0
		first := (e.preRollNext - e.preRollCount + realtimePreRollFrames) % realtimePreRollFrames
		for i := 0; i < e.preRollCount; i++ {
			index := (first + i) % realtimePreRollFrames
			e.turnSamples += e.preRollLengths[index]
			emit(endpointFrame{
				samples:     e.preRoll[index][:e.preRollLengths[index]],
				participant: e.participant,
				utterance:   e.utterance,
				include:     true,
				turnSamples: e.turnSamples,
			})
		}
		e.preRollCount = 0
	}

	if !e.inTurn {
		index := e.preRollNext
		copy(e.preRoll[index][:], samples)
		e.preRollLengths[index] = len(samples)
		e.preRollNext = (index + 1) % realtimePreRollFrames
		if e.preRollCount < realtimePreRollFrames {
			e.preRollCount++
		}
		return
	}
	e.turnSamples += len(samples)
	if active {
		e.silenceFrames = 0
	} else {
		e.silenceFrames++
	}

	frame := endpointFrame{
		samples:     samples,
		participant: e.participant,
		utterance:   e.utterance,
		include:     true,
		turnSamples: e.turnSamples,
	}
	if e.silenceFrames >= realtimeSilenceFrames {
		frame.commit = true
		e.inTurn = false
		e.turnSamples = 0
		e.silenceFrames = 0
	}
	emit(frame)
}

func speech(samples []int16) bool {
	if len(samples) == 0 {
		return false
	}
	var sumSquares int64
	for _, sample := range samples {
		value := int64(sample)
		sumSquares += value * value
	}
	return math.Sqrt(float64(sumSquares)/float64(len(samples))) >= realtimeSpeechRMS
}

// resampler preserves its phase and previous sample across calls. The only output it
// defers is a fractional sample that needs the next input sample for interpolation.
type resampler struct {
	inputSamples  int64
	outputSamples int64
	previous      int16
	started       bool
}

func (r *resampler) Append(output []byte, input []int16) []byte {
	for _, current := range input {
		if !r.started {
			r.started = true
			r.previous = current
			r.inputSamples = 1
			output = appendPCM16(output, current)
			r.outputSamples++
			continue
		}
		r.inputSamples++
		index := r.inputSamples - 1
		for 2*r.outputSamples <= 3*index {
			remainder := 2*r.outputSamples - 3*(index-1)
			value := int64(r.previous)*(3-remainder) + int64(current)*remainder
			output = appendPCM16(output, int16(value/3))
			r.outputSamples++
		}
		r.previous = current
	}
	return output
}

func (r *resampler) Flush(output []byte) []byte {
	if !r.started {
		return output
	}
	target := (3*r.inputSamples + 1) / 2
	for r.outputSamples < target {
		output = appendPCM16(output, r.previous)
		r.outputSamples++
	}
	return output
}

func appendPCM16(output []byte, sample int16) []byte {
	return binary.LittleEndian.AppendUint16(output, uint16(sample))
}
