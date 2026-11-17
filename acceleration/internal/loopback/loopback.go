// Package loopback is a call with no network in it.
//
// The agent's edge is deliberately four methods wide so that its flow can be exercised
// in-process rather than only against a real transport. This is that in-process edge:
// microphones write 16 kHz PCM in, and whatever the agent says back is paced out at the
// rate it would have been heard rather than the rate it was synthesised.
//
// The pacing is the part that matters. A voice provider streams an utterance far faster
// than it is spoken, so an edge that swallowed a reply whole would leave the agent
// believing it had finished talking seconds before anybody had heard it, and nothing about
// taking turns would mean anything.
package loopback

import (
	"context"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// chunkDuration is how much audio a call carries per packet.
const chunkDuration = 20 * time.Millisecond

// defaultPlayoutAhead is how much of the agent's speech may be waiting to be heard before
// publishing has to wait, which is what a real edge allows.
const defaultPlayoutAhead = 400 * time.Millisecond

// Options configures an Edge.
type Options struct {
	// PlayoutAhead is how much speech may be queued ahead of the playhead. Zero takes the
	// default.
	PlayoutAhead time.Duration
	// Published, when set, is handed every chunk the agent speaks. It is how somebody on
	// the other end of the loopback hears it.
	Published func(audio.PcmData)
	// Inbound is how many packets may be waiting to be transcribed.
	Inbound int
}

// Edge is the call.
type Edge struct {
	inbound   chan agent.InboundAudio
	ahead     time.Duration
	published func(audio.PcmData)

	mu sync.Mutex
	// playhead is when the speech published so far will have been heard.
	playhead time.Time
	spokenAt []time.Time
	// mics are the ends of the call this edge handed out, so leaving can stop them. The
	// agent hangs up on its own when the conversation is over and knows nothing about who
	// was talking to it, so the edge is the only thing that can get this order right.
	mics     []*Microphone
	leftOnce sync.Once
}

// New returns an edge that has not been joined.
func New(options Options) *Edge {
	if options.PlayoutAhead <= 0 {
		options.PlayoutAhead = defaultPlayoutAhead
	}
	if options.Inbound <= 0 {
		options.Inbound = 64
	}
	return &Edge{
		inbound:   make(chan agent.InboundAudio, options.Inbound),
		ahead:     options.PlayoutAhead,
		published: options.Published,
	}
}

func (e *Edge) Join(context.Context) error { return nil }

func (e *Edge) Audio() <-chan agent.InboundAudio { return e.inbound }

// PublishAudio takes the agent's speech at the rate it is heard rather than the rate it is
// synthesised.
func (e *Edge) PublishAudio(pcm audio.PcmData) error {
	e.mu.Lock()
	if e.playhead.Before(time.Now()) {
		e.playhead = time.Now()
	}
	heardAt := e.playhead
	e.playhead = e.playhead.Add(time.Duration(pcm.DurationMs()) * time.Millisecond)
	e.spokenAt = append(e.spokenAt, heardAt)
	queued := time.Until(e.playhead) - e.ahead
	published := e.published
	e.mu.Unlock()

	if published != nil {
		published(pcm)
	}
	if queued > 0 {
		time.Sleep(queued)
	}
	return nil
}

// Leave hangs up. Closing the channel is what tells the agent the call is over, so every
// microphone is stopped first: one still writing would be writing into a closed channel.
func (e *Edge) Leave() error {
	e.leftOnce.Do(func() {
		e.mu.Lock()
		mics := e.mics
		e.mics = nil
		e.mu.Unlock()

		for _, mic := range mics {
			mic.Stop()
		}
		close(e.inbound)
	})
	return nil
}

// Microphone opens one participant's end of the call.
func (e *Edge) Microphone(participant stt.Participant, room audio.PcmData) *Microphone {
	mic := newMicrophone(e.inbound, participant, room)

	e.mu.Lock()
	e.mics = append(e.mics, mic)
	e.mu.Unlock()
	return mic
}

// Talking reports whether the agent still has speech left to be heard, which is what
// holding the floor means.
//
// Chunk timestamps cannot answer this: a provider hands over a whole sentence at a time, so
// an agent that is still talking can be several seconds from its next chunk.
func (e *Edge) Talking() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return time.Now().Before(e.playhead)
}

// SpokeAfter is when the agent first made a sound after the given moment.
func (e *Edge) SpokeAfter(after time.Time) (time.Time, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()

	for _, at := range e.spokenAt {
		if at.After(after) {
			return at, true
		}
	}
	return time.Time{}, false
}

// Microphone is one participant's end of the call. It always sends something, because a
// real call carries the room even when nobody is talking, and utterances are spliced into
// that stream.
type Microphone struct {
	inbound     chan<- agent.InboundAudio
	participant stt.Participant
	// room is looped between utterances: silence in a quiet call, noise in a busy one.
	room audio.PcmData

	mu      sync.Mutex
	playing *playback
	offset  int

	halt     chan struct{}
	haltOnce sync.Once
	stopped  chan struct{}
}

type playback struct {
	samples  []int16
	at       int
	finished chan time.Time
}

func newMicrophone(
	inbound chan<- agent.InboundAudio,
	participant stt.Participant,
	room audio.PcmData,
) *Microphone {
	sending := &Microphone{
		inbound:     inbound,
		participant: participant,
		room:        room,
		halt:        make(chan struct{}),
		stopped:     make(chan struct{}),
	}
	go sending.run()
	return sending
}

// Play splices an utterance into the stream and returns once all of it has been sent, which
// is the moment the speaker stopped talking.
func (m *Microphone) Play(pcm audio.PcmData) time.Time {
	finished := make(chan time.Time, 1)
	m.mu.Lock()
	m.playing = &playback{samples: pcm.Samples, finished: finished}
	m.mu.Unlock()

	select {
	case at := <-finished:
		return at
	case <-m.halt:
		return time.Now()
	}
}

// Stop closes this end of the call. It returns once the track is really finished, so the
// call can be torn down without a packet arriving on a channel that has been closed.
func (m *Microphone) Stop() {
	m.haltOnce.Do(func() { close(m.halt) })
	<-m.stopped
}

func (m *Microphone) run() {
	ticker := time.NewTicker(chunkDuration)
	defer ticker.Stop()
	defer close(m.stopped)

	for {
		select {
		case <-m.halt:
			return
		case <-ticker.C:
			select {
			case m.inbound <- agent.InboundAudio{Participant: m.participant, Audio: m.next()}:
			case <-m.halt:
				return
			}
		}
	}
}

// next is the next packet of this participant's track.
func (m *Microphone) next() audio.PcmData {
	size := stt.SampleRate * int(chunkDuration/time.Millisecond) / 1000
	samples := make([]int16, size)

	m.mu.Lock()
	defer m.mu.Unlock()

	if playing := m.playing; playing != nil {
		copied := copy(samples, playing.samples[playing.at:])
		playing.at += copied
		if playing.at >= len(playing.samples) {
			playing.finished <- time.Now()
			m.playing = nil
		}
	} else if len(m.room.Samples) > 0 {
		for i := range samples {
			samples[i] = m.room.Samples[(m.offset+i)%len(m.room.Samples)]
		}
		m.offset += size
	}
	return audio.PcmData{Samples: samples, SampleRate: stt.SampleRate, Channels: 1}
}
