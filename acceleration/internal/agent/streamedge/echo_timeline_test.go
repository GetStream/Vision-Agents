package streamedge

import (
	"time"

	webrtcaudio "github.com/GetStream/getstream-go-webrtc/audio"
	"github.com/GetStream/getstream-go-webrtc/audio/opus"
	audiortc "github.com/GetStream/getstream-go-webrtc/audio/rtc"
	sfu_models "github.com/GetStream/protocol/protobuf/video/sfu/models"
	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4"
	"github.com/thesyncim/gopus"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/echocancel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// timelineCall is 20 s of the agent talking to a sip- caller whose phone sends every packet,
// each 20 ms of what it heard of the agent, 300 ms late, as stamp says: stamp is given each
// packet's number and the packet, with the timestamp and payload a phone would send, to change.
// It returns what the phone heard and what the edge handed the agent of it, 20 ms for each
// packet.
func (s *EchoSuite) timelineCall(stamp func(i int, packet *rtp.Packet)) (heard, handed []int16) {
	edge := s.sipEdge()
	s.T().Cleanup(func() { _ = edge.speaker.Close() })
	edge.report(&sfu_models.Participant{UserId: "sip-1", SessionId: "sip-1-session"}, true)
	encoder, err := opus.NewEncoder(opus.Config{SampleRate: echocancel.SampleRate, FrameDuration: opusFrameDuration,
		Application: opus.ApplicationVoIP})
	s.Require().NoError(err)
	track := make(packets, 4)
	s.T().Cleanup(func() { close(track) })
	reader, err := audiortc.NewRTPReader(track, webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus,
		ClockRate: opusSampleRate, Channels: 2}, audiortc.ReaderConfig{Opus: opus.Config{SampleRate: stt.SampleRate}})
	s.Require().NoError(err)
	go edge.hear(reader, edge.quality.inbound(track, opusSampleRate),
		stt.Participant{ID: "sip-1-session", UserID: "sip-1"}, make(chan struct{}))

	decoder, err := gopus.NewDecoder(gopus.DefaultDecoderConfig(echocancel.SampleRate, 1))
	s.Require().NoError(err)
	phone := newPhone()
	at, pause := 0, 0
	frame := make([]int16, 5760)
	for i := range 20 * 50 {
		if pause > 0 {
			pause--
		} else if len(edge.speaker.frames) < 2 {
			end := min(at+s.reply.SampleRate/50, len(s.reply.Samples))
			s.Require().NoError(edge.PublishAudio(audio.PcmData{Samples: s.reply.Samples[at:end],
				SampleRate: s.reply.SampleRate, Channels: 1}))
			if at = end; at == len(s.reply.Samples) {
				at, pause = 0, 10
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
		coded, err := encoder.Encode(webrtcaudio.FromInt16(back, echocancel.SampleRate, 1))
		s.Require().NoError(err)
		packet := &rtp.Packet{Header: rtp.Header{Version: 2, PayloadType: 111, SequenceNumber: uint16(i),
			Timestamp: uint32(i) * 960}, Payload: coded[0]}
		stamp(i, packet)
		track <- packet
		select {
		case got := <-edge.Audio():
			heard = append(heard, back...)
			handed = append(handed, got.Audio.Samples...)
		case <-time.After(5 * time.Second):
			s.FailNow("the edge handed the agent nothing of a packet", "packet %d", i)
		}
	}
	return heard, handed
}

// A phone's packets are stamped from its own clock and leave it a little unevenly; the reader
// times each frame by its packet's stamp.
func (s *EchoSuite) TestPacketsStampedLateNowAndThenLeaveTheEchoCancelled() {
	heard, handed := s.timelineCall(func(i int, packet *rtp.Packet) {
		if i%10 == 5 {
			packet.Timestamp += 720 // 15 ms late
		}
	})

	quieter := gone(heard, handed, 5, 20)
	s.T().Logf("every 10th packet stamped 15 ms late: %.1f dB quieter from 5 s on", quieter)
	s.Greater(quieter, 60.0)
}

// The reader conceals a packet it cannot decode, and stamps the frame it makes up a frame
// before the packet's own time.
func (s *EchoSuite) TestAPacketThatCannotBeDecodedLeavesTheEchoCancelled() {
	const broken = 600
	heard, handed := s.timelineCall(func(i int, packet *rtp.Packet) {
		if i == broken {
			packet.Payload = []byte{0x03, 0xff}
		}
	})

	at := float64(broken) / 50
	quieter := gone(heard, handed, at, at+2)
	s.T().Logf("a packet that cannot be decoded at %.0f s: %.1f dB quieter in the 2 s after it", at, quieter)
	s.Greater(quieter, 60.0)
}

// A phone whose clock steps back, as one restarting its stream does, sends earlier stamps
// than the ones before.
func (s *EchoSuite) TestStampsThatStepBackLeaveTheEchoCancelled() {
	const back = 600
	heard, handed := s.timelineCall(func(i int, packet *rtp.Packet) {
		if i >= back {
			packet.Timestamp -= 2 * opusSampleRate
		}
	})

	at := float64(back) / 50
	quieter := gone(heard, handed, at, at+3)
	s.T().Logf("stamps 2 s back from %.0f s on: %.1f dB quieter in the 3 s after", at, quieter)
	s.Greater(quieter, 60.0)
}
