package streamedge

import (
	"io"
	"log/slog"
	"math"
	"slices"
	"strings"
	"time"

	webrtcaudio "github.com/GetStream/getstream-go-webrtc/audio"
	"github.com/GetStream/getstream-go-webrtc/audio/opus"
	audiortc "github.com/GetStream/getstream-go-webrtc/audio/rtc"
	sfu_models "github.com/GetStream/protocol/protobuf/video/sfu/models"
	"github.com/pion/interceptor"
	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4"
	"github.com/thesyncim/gopus"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/echocancel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// packets is a participant's track as the SFU forwards it, one RTP packet at a time.
type packets chan *rtp.Packet

func (p packets) ReadRTP() (*rtp.Packet, interceptor.Attributes, error) {
	packet, ok := <-p
	if !ok {
		return nil, nil, io.EOF
	}
	return packet, nil, nil
}

// sender is a participant's phone or browser sending what its microphone hears as 20 ms Opus
// packets. A sender with dtx sends nothing once it has heard nothing for 200 ms, but for one
// packet every 400 ms, as Opus DTX does; its timestamps move on regardless.
type sender struct {
	track   packets
	encoder *opus.Encoder
	dtx     bool
	seq     uint16
	ts      uint32
	quiet   int
}

// send codes 20 ms the microphone heard and sends it, and returns what it sent, nil for
// nothing.
func (s *EchoSuite) send(to *sender, heard []int16) []byte {
	defer func() { to.ts += 960 }()
	if to.dtx {
		if audio.Audible(heard) {
			to.quiet = 0
		} else if to.quiet++; to.quiet > 10 && to.quiet%20 != 0 {
			return nil
		}
	}
	coded, err := to.encoder.Encode(webrtcaudio.FromInt16(heard, echocancel.SampleRate, 1))
	s.Require().NoError(err)
	s.Require().Len(coded, 1)
	to.track <- &rtp.Packet{Header: rtp.Header{Version: 2, PayloadType: 111, SequenceNumber: to.seq,
		Timestamp: to.ts}, Payload: coded[0]}
	to.seq++
	return coded[0]
}

// edgeCall is what an edge was sent and handed the agent, by user id: heard is what each
// microphone heard, handed what the edge handed the agent of it, and plain the browser's
// packets decoded as the edge decodes them, with nothing cancelled.
type edgeCall struct {
	heard, handed map[string][]int16
	plain         []int16
}

const (
	// edgeCallLength is how long the call on the edge lasts, edgeAgentUntil when the agent
	// says its last reply, and edgeCallerAt when the caller speaks, once its echo has gone.
	edgeCallLength = 20
	edgeAgentUntil = 12
	edgeCallerAt   = 14
)

// heardOnTheEdge plays the reply over and over until edgeAgentUntil on an edge that cancels
// the echo of sip- participants, with a sip- caller whose phone sends nothing while it is
// quiet and a browser user next to them, each hearing the agent's voice back 300 ms later. The
// caller speaks from edgeCallerAt on. It returns what each was sent and handed over the 20 ms
// both were sent in.
func (s *EchoSuite) heardOnTheEdge() edgeCall {
	edge := s.sipEdge()
	s.T().Cleanup(func() { _ = edge.speaker.Close() })

	codec := webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: opusSampleRate, Channels: 2}
	senders := map[string]*sender{}
	for _, user := range []string{"sip-15550100", "browser"} {
		// Both are in the call before their audio arrives, as the SFU reports them.
		edge.report(&sfu_models.Participant{UserId: user, SessionId: user + "-session"}, true)
		encoder, err := opus.NewEncoder(opus.Config{SampleRate: echocancel.SampleRate, FrameDuration: opusFrameDuration,
			Application: opus.ApplicationVoIP})
		s.Require().NoError(err)
		senders[user] = &sender{track: make(packets, 4), encoder: encoder, dtx: user != "browser"}
		reader, err := audiortc.NewRTPReader(senders[user].track, codec,
			audiortc.ReaderConfig{Opus: opus.Config{SampleRate: stt.SampleRate}})
		s.Require().NoError(err)
		go edge.hear(reader, edge.quality.inbound(senders[user].track, opusSampleRate),
			stt.Participant{ID: user + "-session", UserID: user}, make(chan struct{}))
	}
	defer func() {
		for _, to := range senders {
			close(to.track)
		}
	}()

	decoder, err := gopus.NewDecoder(gopus.DefaultDecoderConfig(echocancel.SampleRate, 1))
	s.Require().NoError(err)
	plain, err := opus.NewDecoder(opus.Config{SampleRate: stt.SampleRate})
	s.Require().NoError(err)
	phone := newPhone()
	call := edgeCall{heard: map[string][]int16{}, handed: map[string][]int16{}}
	at, pause := 0, 0
	frame := make([]int16, 5760)
	for i := range edgeCallLength * 50 {
		if pause > 0 || i >= edgeAgentUntil*50 {
			pause--
		} else if len(edge.speaker.frames) < 2 {
			end := min(at+s.reply.SampleRate/50, len(s.reply.Samples))
			s.Require().NoError(edge.PublishAudio(audio.PcmData{Samples: s.reply.Samples[at:end],
				SampleRate: s.reply.SampleRate, Channels: 1}))
			if at = end; at == len(s.reply.Samples) {
				at, pause = 0, 50
			}
		}
		sample, err := edge.speaker.NextSample(s.ctx)
		s.Require().NoError(err)
		played := make([]int16, echoFrameSamples)
		if sample.Data != nil {
			n, err := decoder.DecodeInt16(sample.Data, frame)
			s.Require().NoError(err)
			copy(played, frame[:n])
		}
		echo := phone.back(played)
		for user, to := range senders {
			mic := echo
			if user != "browser" {
				mic = slices.Clone(echo)
				if from := i*echoFrameSamples - edgeCallerAt*echocancel.SampleRate; from >= 0 && from < len(s.caller) {
					said := s.caller[from:min(from+echoFrameSamples, len(s.caller))]
					for j, v := range said {
						mic[j] = int16(max(math.MinInt16, min(math.MaxInt16, int(mic[j])+int(v))))
					}
				}
			}
			payload := s.send(to, mic)
			if payload == nil {
				continue
			}
			if user == "browser" {
				decoded, err := plain.Decode(payload)
				s.Require().NoError(err)
				call.plain = append(call.plain, decoded.ToInt16().Int16()...)
			}
			select {
			case got := <-edge.Audio():
				s.Require().Equal(user, got.Participant.UserID)
				s.Require().Len(got.Audio.Samples, len(mic), "the edge handed the agent part of a packet")
				call.heard[user] = append(call.heard[user], mic...)
				call.handed[user] = append(call.handed[user], got.Audio.Samples...)
			case <-time.After(5 * time.Second):
				s.FailNow("the edge handed the agent nothing of what was sent", user)
			}
		}
	}
	return call
}

func (s *EchoSuite) TestOnlyTheAudioOfTheParticipantsNamedIsCleanedBeforeTheAgentHearsIt() {
	call := s.heardOnTheEdge()
	heard, handed := call.heard["sip-15550100"], call.handed["sip-15550100"]

	// The caller's phone sends nothing while quiet, so its audio is shorter than the call:
	// the caller's own voice is measured in its last 5 s, which hold it alone.
	end := float64(len(heard)) / echocancel.SampleRate
	echoGone := gone(heard, handed, 5, edgeAgentUntil-3)
	callerKept := -gone(heard, handed, end-5, end)
	s.T().Logf("from a sip- caller whose phone stops sending while quiet, the agent's voice %.1f dB quieter "+
		"and their own %.1f dB", echoGone, callerKept)
	s.Greater(echoGone, 20.0, "the caller's line was not cleaned")
	s.InDelta(0, callerKept, 2, "the caller's own voice was not kept")
	s.Equal(call.plain, call.handed["browser"], "the browser's audio was not handed over as it was decoded")
}

func (s *EchoSuite) TestEchoCancellationThatNamesNobodyIsRefused() {
	_, err := New(Options{CallID: "call", User: User{ID: "agent"}, APIKey: "key", UserToken: "token",
		EchoCancellation: &EchoCancellation{}})

	s.Error(err)
}

// sipEdge is an edge that cancels the echo of sip- participants.
func (s *EchoSuite) sipEdge() *Edge {
	edge, err := New(Options{CallID: "call", User: User{ID: "agent"}, APIKey: "key", UserToken: "token",
		Logger: slog.New(slog.DiscardHandler),
		EchoCancellation: &EchoCancellation{From: func(userID string) bool {
			return strings.HasPrefix(userID, "sip-")
		}}})
	s.Require().NoError(err)
	return edge
}

func (s *EchoSuite) TestAReplacedTrackStopsTakingTheAgentsVoiceAtOnce() {
	edge := s.sipEdge()
	track := make(packets)
	s.T().Cleanup(func() { close(track) })
	reader, err := audiortc.NewRTPReader(track, webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus,
		ClockRate: opusSampleRate, Channels: 2}, audiortc.ReaderConfig{Opus: opus.Config{SampleRate: stt.SampleRate}})
	s.Require().NoError(err)
	stop := make(chan struct{})
	go edge.hear(reader, edge.quality.inbound(track, opusSampleRate), stt.Participant{UserID: "sip-1"}, stop)
	s.Eventually(edge.echo.Wanted, time.Second, time.Millisecond, "the caller's line never opened")

	// The track sends nothing more, as one the SFU has replaced does.
	close(stop)

	s.Eventually(func() bool { return !edge.echo.Wanted() }, time.Second, time.Millisecond,
		"the replaced track's line still counts as listening")
	// Two seconds of voice would overflow the second a line holds, had the line still taken it.
	dropped := s.counted("router_agent_echo_voice_dropped_frames_total")
	for range 100 {
		edge.echo.Played(make([]int16, echoFrameSamples))
	}
	s.Equal(dropped, s.counted("router_agent_echo_voice_dropped_frames_total"),
		"the replaced track's line still takes the agent's voice")
}

// A track that ends closes its line, which then no longer takes the agent's voice.
func (s *EchoSuite) TestAPhoneCallersLineClosesWhenItsTrackEnds() {
	edge := s.sipEdge()
	track := make(packets, 1)
	reader, err := audiortc.NewRTPReader(track, webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus,
		ClockRate: opusSampleRate, Channels: 2}, audiortc.ReaderConfig{Opus: opus.Config{SampleRate: stt.SampleRate}})
	s.Require().NoError(err)
	sending := &sender{track: track}
	sending.encoder, err = opus.NewEncoder(opus.Config{SampleRate: echocancel.SampleRate,
		FrameDuration: opusFrameDuration, Application: opus.ApplicationVoIP})
	s.Require().NoError(err)
	heard := make(chan struct{})
	go func() {
		edge.hear(reader, edge.quality.inbound(track, opusSampleRate), stt.Participant{UserID: "sip-1"},
			make(chan struct{}))
		close(heard)
	}()
	s.send(sending, make([]int16, echoFrameSamples))
	<-edge.Audio()
	s.True(edge.echo.Wanted(), "the caller's line never opened")

	close(track)
	<-heard

	s.False(edge.echo.Wanted())
}

func (s *EchoSuite) TestTheAgentsVoiceIsKeptForAPhoneCallerFromWhenTheyArrive() {
	edge := s.sipEdge()

	edge.report(&sfu_models.Participant{UserId: "browser", SessionId: "b"}, true)
	s.False(edge.echo.Wanted(), "kept for a browser")
	edge.report(&sfu_models.Participant{UserId: "sip-1", SessionId: "c"}, true)
	s.True(edge.echo.Wanted(), "not kept for a caller who arrived without a name")
	edge.report(&sfu_models.Participant{UserId: "sip-1", SessionId: "c", Name: "Caller"}, true)
	s.True(edge.echo.Wanted())
	edge.report(&sfu_models.Participant{UserId: "sip-1", SessionId: "c"}, false)
	s.False(edge.echo.Wanted(), "still kept once the caller left, or their name counted them twice")
}
