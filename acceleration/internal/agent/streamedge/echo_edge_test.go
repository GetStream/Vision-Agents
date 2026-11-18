package streamedge

import (
	"io"
	"log/slog"
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

// send codes 20 ms the microphone heard and sends it, and says whether it did.
func (s *EchoSuite) send(to *sender, heard []int16) bool {
	defer func() { to.ts += 960 }()
	if to.dtx {
		if audio.Audible(heard) {
			to.quiet = 0
		} else if to.quiet++; to.quiet > 10 && to.quiet%20 != 0 {
			return false
		}
	}
	coded, err := to.encoder.Encode(webrtcaudio.FromInt16(heard, echocancel.SampleRate, 1))
	s.Require().NoError(err)
	s.Require().Len(coded, 1)
	to.track <- &rtp.Packet{Header: rtp.Header{Version: 2, PayloadType: 111, SequenceNumber: to.seq,
		Timestamp: to.ts}, Payload: coded[0]}
	to.seq++
	return true
}

// heardOnTheEdge plays the reply over and over for seconds on an edge that cancels the echo of
// sip- participants, with a sip- caller whose phone sends nothing while it is quiet and a
// browser user next to them, each hearing the agent's voice back 300 ms later. It returns what
// each microphone heard and what the edge handed the agent of it, by user id, over the 20 ms
// both were sent in.
func (s *EchoSuite) heardOnTheEdge(seconds int) (heard, handed map[string][]int16) {
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
	phone := newPhone()
	heard, handed = map[string][]int16{}, map[string][]int16{}
	at, pause := 0, 0
	frame := make([]int16, 5760)
	for range seconds * 50 {
		if pause > 0 {
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
		back := phone.back(played)
		for user, to := range senders {
			if !s.send(to, back) {
				continue
			}
			select {
			case got := <-edge.Audio():
				s.Require().Equal(user, got.Participant.UserID)
				heard[user] = append(heard[user], back...)
				handed[user] = append(handed[user], got.Audio.Samples...)
			case <-time.After(5 * time.Second):
				s.FailNow("the edge handed the agent nothing of what was sent", user)
			}
		}
	}
	return heard, handed
}

func (s *EchoSuite) TestOnlyTheAudioOfTheParticipantsNamedIsCleanedBeforeTheAgentHearsIt() {
	heard, handed := s.heardOnTheEdge(30)

	caller := gone(heard["sip-15550100"], handed["sip-15550100"], 5, 25)
	browser := gone(heard["browser"], handed["browser"], 5, 25)
	s.T().Logf("the agent's voice coming back from a sip- caller whose phone stops sending while quiet: "+
		"%.1f dB quieter; from a browser: %.1f dB", caller, browser)
	s.Greater(caller, 20.0, "the caller's line was not cleaned")
	s.InDelta(0, browser, 3, "the browser's audio was cleaned")
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
		"the replaced track's line still takes the agent's voice")
}

func (s *EchoSuite) TestTheAgentsVoiceIsKeptForAPhoneCallerFromWhenTheyArrive() {
	edge := s.sipEdge()

	edge.report(&sfu_models.Participant{UserId: "browser", SessionId: "b"}, true)
	s.False(edge.echo.Wanted(), "kept for a browser")
	edge.report(&sfu_models.Participant{UserId: "sip-1", SessionId: "c"}, true)
	edge.report(&sfu_models.Participant{UserId: "sip-1", SessionId: "c", Name: "Caller"}, true)
	s.True(edge.echo.Wanted())
	edge.report(&sfu_models.Participant{UserId: "sip-1", SessionId: "c"}, false)
	s.False(edge.echo.Wanted(), "still kept once the caller left, or their name counted them twice")
}
