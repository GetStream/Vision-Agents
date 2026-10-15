package bench

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/rtc/audio"
	"github.com/GetStream/Vision-Agents/benchmark/rtc/streamrtc"
	"github.com/GetStream/Vision-Agents/benchmark/rtc/transport"
	"github.com/GetStream/Vision-Agents/sdks/go/agents"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

const (
	// frame is how often audio moves in either direction, which is one Opus frame.
	frame = 20 * time.Millisecond
	// sayFor is how long a caller talks before stopping, and pauseFor how long they
	// leave between the end of one answer and their next sentence.
	sayFor   = 1500 * time.Millisecond
	pauseFor = 500 * time.Millisecond
	// replyWithin is how long a caller waits to hear anything before giving up on the
	// turn. It is well past the point a conversation has already failed.
	replyWithin = 6 * time.Second
	// quietFor is how much silence from the agent ends its answer.
	quietFor = 700 * time.Millisecond
	// toneHz and toneAmplitude are the caller's voice: a tone loud enough to be speech
	// to anything measuring energy, which is what the mock transcriber does.
	toneHz        = 180
	toneAmplitude = 8000
	// joinWithin bounds the wait for the agent to be on the call and publishing.
	joinWithin = 45 * time.Second
)

// conversation is what one caller did, and where it stopped if it did.
type conversation struct {
	sessionMs float64
	trackMs   float64
	turns     []turn
	dropped   int
	// failedAt names the step of a conversation that failed, for the error counts, and
	// err is what it said.
	failedAt string
	err      error
}

// turn is one thing the caller said and what came back.
type turn struct {
	// replyMs is the wait between the caller finishing and the first audio of the answer.
	// It is zero for a turn that was never answered.
	replyMs float64
	// gapMs is the longest silence inside the answer, which is what a speaker running
	// out of audio sounds like.
	gapMs float64
	// quality is how faithfully the answer's audio arrived, when scored says a probe
	// caller recorded it.
	quality toneQuality
	scored  bool
}

// answered reports whether the agent said anything back.
func (t turn) answered() bool { return t.replyMs > 0 }

// Caller opens one conversation and talks to it for as long as it is told to.
type Caller struct {
	// Agent is the stored config every conversation runs, and Models what it is
	// overridden with.
	Agent  string
	Models Models
	// Index distinguishes this caller from the others in the same step.
	Index int
	// Probe records every answer and scores its audio against the mock TTS's tone.
	Probe bool
	// Logger is where a caller reports what it could not do.
	Logger *slog.Logger
}

// Converse holds one conversation until the given moment and reports how it went.
func (c Caller) Converse(ctx context.Context, until time.Time) conversation {
	held := conversation{}

	// A user of its own per conversation: Stream holds one user to 50 connections, so a
	// step of more under one agent user measures that limit rather than the router.
	agent, err := agents.New(agents.Options{
		Name:   c.Agent,
		UserID: c.agentID(),
		LLM:    c.pipeline(),
		Logger: c.Logger,
	})
	if err != nil {
		return c.failed(held, "agent.build", err)
	}

	askedAt := time.Now()
	session, err := agent.Join(ctx, agents.SessionOptions{AgentID: c.agentID()})
	if err != nil {
		return c.failed(held, "session.create", err)
	}
	held.sessionMs = msSince(askedAt)
	defer session.Close(context.WithoutCancel(ctx))
	// The SDK stops reading the socket once its buffer of unread events is full, and a
	// socket that stops answering pings is cut after 90s, taking the closing frame with it.
	go func() {
		for range session.Events() {
		}
	}()

	call := session.Call()
	media, err := streamrtc.Join(ctx, streamrtc.Options{
		CallID:   call.ID,
		CallType: call.Type,
		UserID:   fmt.Sprintf("perfbench-caller-%d", c.Index),
		Logger:   c.Logger,
	})
	if err != nil {
		return c.failed(held, "call.join", err)
	}
	defer media.Close()

	joinedAt := time.Now()
	waiting, cancel := context.WithTimeout(ctx, joinWithin)
	defer cancel()
	if err := media.WaitForAgent(waiting); err != nil {
		return c.failed(held, "agent.track", err)
	}
	held.trackMs = msSince(joinedAt)

	held.turns, err = c.talk(ctx, media, until)
	held.dropped = media.Dropped()
	if err != nil {
		return c.failed(held, "call.audio", err)
	}
	return held
}

// talk says a sentence, waits for the answer, and goes again until the time is up.
func (c Caller) talk(ctx context.Context, media transport.Media, until time.Time) ([]turn, error) {
	ctx, cancel := context.WithDeadline(ctx, until)
	defer cancel()

	speech := audio.Tone(audio.FrameSamples, toneHz, toneAmplitude)
	silence := make([]int16, audio.FrameSamples)
	ticker := time.NewTicker(frame)
	defer ticker.Stop()

	var turns []turn
	for ctx.Err() == nil {
		if err := c.send(ctx, media, ticker, speech, sayFor); err != nil {
			return turns, err
		}
		heard, err := c.listen(ctx, media, ticker, silence)
		if err != nil {
			return turns, err
		}
		// A turn cut short by the end of the step was never given its chance to be
		// answered, so it is not counted against the agent.
		if ctx.Err() != nil && !heard.answered() {
			break
		}
		turns = append(turns, heard)
		if err := c.send(ctx, media, ticker, silence, pauseFor); err != nil {
			return turns, err
		}
	}
	return turns, nil
}

// send plays one frame over and over for the given stretch, which is how a caller both
// talks and keeps the leg alive while quiet.
func (c Caller) send(ctx context.Context, media transport.Media, ticker *time.Ticker, pcm []int16, how time.Duration) error {
	deadline := time.Now().Add(how)
	for time.Now().Before(deadline) {
		select {
		case <-ticker.C:
		case <-ctx.Done():
			return nil
		}
		if err := media.Send(pcm); err != nil {
			return err
		}
		drain(media)
	}
	return nil
}

// listen waits for the answer and measures it: how long the caller waited to hear
// anything, and the longest silence once the agent had started.
//
// The caller keeps sending silence throughout, because a leg that stops sending is a leg
// the far end hears as gone.
func (c Caller) listen(ctx context.Context, media transport.Media, ticker *time.Ticker, silence []int16) (heard turn, err error) {
	askedAt := time.Now()
	var startedAt, lastAudibleAt time.Time
	var recorded *recording
	if c.Probe {
		recorded = &recording{}
		defer func() { heard.quality, heard.scored = measureTone(recorded.pcm) }()
	}

	for {
		select {
		case <-ticker.C:
		case <-ctx.Done():
			return heard, nil
		}
		if err := media.Send(silence); err != nil {
			return heard, err
		}

		if audible(media, recorded) {
			now := time.Now()
			if startedAt.IsZero() {
				startedAt = now
				heard.replyMs = msSince(askedAt)
			} else if gap := msBetween(lastAudibleAt, now); gap > heard.gapMs {
				heard.gapMs = gap
			}
			lastAudibleAt = now
			continue
		}

		switch {
		case startedAt.IsZero() && time.Since(askedAt) > replyWithin:
			return heard, nil
		case !startedAt.IsZero() && time.Since(lastAudibleAt) > quietFor:
			// The silence that ended the answer is not a gap inside it.
			return heard, nil
		}
	}
}

// audible drains whatever arrived since the last tick, onto recorded when it is not nil,
// and reports whether any of it was the agent speaking.
func audible(media transport.Media, recorded *recording) bool {
	speaking, arrived := false, false
	for {
		select {
		case frame := <-media.Recv():
			arrived = true
			loud := audio.FrameEnergy(frame.PCM) > audio.DefaultSpeechThreshold
			if recorded != nil {
				recorded.add(frame.PCM, loud)
			}
			speaking = speaking || loud
		default:
			if recorded != nil && !arrived {
				recorded.missed++
			}
			return speaking
		}
	}
}

// recording is what a probe caller heard, played out as a browser would: the agent sends
// nothing through most of a silence (Opus DTX), and the decoder plays the time it skipped as
// silence. The decoder here only hands over what arrived, so that time is put back.
type recording struct {
	pcm []int16
	// missed counts the ticks nothing arrived on since the last frame.
	missed int
	quiet  bool
}

func (r *recording) add(frame []int16, loud bool) {
	// Only next to a quiet frame: a frame missing from speech is loss, which the score has to
	// see, rather than a silence the agent did not send. Either side will do, because the
	// jitter buffer holds the last frame before a gap until the one after it arrives.
	if r.quiet || !loud {
		r.pcm = append(r.pcm, make([]int16, r.missed*len(frame))...)
	}
	r.pcm = append(r.pcm, frame...)
	r.missed, r.quiet = 0, !loud
}

// drain throws away inbound audio nobody is measuring, so the leg's queue stays empty and
// its dropped count stays an honest signal.
func drain(media transport.Media) {
	for {
		select {
		case <-media.Recv():
		default:
			return
		}
	}
}

// pipeline is the conversation this caller holds: the stored agent config, with the three
// models overridden so a run says outright which it measured.
func (c Caller) pipeline() *stream.Pipeline {
	return stream.Accelerated(stream.Config{
		Agent: c.Agent,
		STT:   c.Models.STT,
		TTS:   c.Models.TTS,
		LLM:   c.Models.LLM,
		// No greeting: the agent waits to be spoken to, so every measured reply is an
		// answer to something rather than the line it opens with.
		Greeting: "",
		Logger:   c.Logger,
	})
}

// agentID keys this conversation. Every caller runs the same stored config, so without
// one of these a message for one conversation would reach whichever started last.
func (c Caller) agentID() string { return fmt.Sprintf("%s-%d", c.Agent, c.Index) }

func (c Caller) failed(held conversation, at string, err error) conversation {
	held.failedAt = at
	held.err = err
	c.Logger.Error("a conversation stopped", "caller", c.Index, "at", at, "error", err)
	return held
}

func msSince(from time.Time) float64 { return msBetween(from, time.Now()) }

func msBetween(from, to time.Time) float64 {
	return float64(to.Sub(from).Microseconds()) / 1000
}
