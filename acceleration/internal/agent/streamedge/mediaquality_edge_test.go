package streamedge

import (
	"net"
	"sync/atomic"
	"time"

	rtc "github.com/GetStream/getstream-go-webrtc"
	sfu_models "github.com/GetStream/protocol/protobuf/video/sfu/models"
	"github.com/pion/interceptor"
	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4"
)

// connectionUp bounds how long two connections on the loopback interface take to connect.
const connectionUp = 10 * time.Second

// loopbackPeer is a pion connection that gathers only loopback candidates, with pion's default
// interceptors, so it sends receiver reports on what it reads as the SFU does.
func (s *StreamEdgeSuite) loopbackPeer() *webrtc.PeerConnection {
	settings := webrtc.SettingEngine{}
	settings.SetIncludeLoopbackCandidate(true)
	settings.SetNetworkTypes([]webrtc.NetworkType{webrtc.NetworkTypeUDP4})
	settings.SetInterfaceFilter(func(name string) bool {
		found, err := net.InterfaceByName(name)
		return err == nil && found.Flags&net.FlagLoopback != 0
	})
	media := &webrtc.MediaEngine{}
	s.Require().NoError(media.RegisterDefaultCodecs())
	registry := &interceptor.Registry{}
	s.Require().NoError(webrtc.RegisterDefaultInterceptors(media, registry))
	api := webrtc.NewAPI(webrtc.WithSettingEngine(settings), webrtc.WithMediaEngine(media),
		webrtc.WithInterceptorRegistry(registry))
	pc, err := api.NewPeerConnection(webrtc.Configuration{})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = pc.Close() })
	return pc
}

// connect negotiates offerer with answerer and waits for both to be connected. sending is the
// track the answerer sends back, added once it has seen the offer so it rides the same
// transceiver.
func (s *StreamEdgeSuite) connect(offerer, answerer *webrtc.PeerConnection, sending webrtc.TrackLocal) {
	offer, err := offerer.CreateOffer(nil)
	s.Require().NoError(err)
	gathered := webrtc.GatheringCompletePromise(offerer)
	s.Require().NoError(offerer.SetLocalDescription(offer))
	<-gathered
	s.Require().NoError(answerer.SetRemoteDescription(*offerer.LocalDescription()))
	_, err = answerer.AddTrack(sending)
	s.Require().NoError(err)
	answer, err := answerer.CreateAnswer(nil)
	s.Require().NoError(err)
	gathered = webrtc.GatheringCompletePromise(answerer)
	s.Require().NoError(answerer.SetLocalDescription(answer))
	<-gathered
	s.Require().NoError(offerer.SetRemoteDescription(*answerer.LocalDescription()))
	s.Eventually(func() bool {
		return offerer.ConnectionState() == webrtc.PeerConnectionStateConnected &&
			answerer.ConnectionState() == webrtc.PeerConnectionStateConnected
	}, connectionUp, 10*time.Millisecond, "the two connections came up")
}

func (s *StreamEdgeSuite) TestACallsMediaQualityIsObservedOnceWhenTheAgentLeaves() {
	// Catches the edge's own wiring: the participant's packets not counted on the way to the
	// decoder, the SFU's receiver reports on the agent's track not read, or the session not
	// observed when the agent leaves. The connections stand in for the agent's and the SFU's.
	edge, err := New(Options{CallID: "call-1", User: User{ID: "agent"}, APIKey: "key", UserToken: "token"})
	s.Require().NoError(err)
	agentPC, sfuPC := s.loopbackPeer(), s.loopbackPeer()

	voice, err := edge.voiceTrack(&sfu_models.TrackInfo{TrackId: "agent-voice",
		TrackType: sfu_models.TrackType_TRACK_TYPE_AUDIO})
	s.Require().NoError(err)
	_, err = agentPC.AddTrack(voice)
	s.Require().NoError(err)
	agentPC.OnTrack(func(remote *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		edge.listen(rtc.OnTrackReceived{Track: remote, TrackType: sfu_models.TrackType_TRACK_TYPE_AUDIO,
			ParticipantID: rtc.ParticipantID{UserID: "caller", SessionID: "caller-session"}})
	})
	// Read, as the SFU reads, since pion reports only on the packets read.
	sfuPC.OnTrack(func(remote *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		for {
			if _, _, err := remote.ReadRTP(); err != nil {
				return
			}
		}
	})
	caller, err := webrtc.NewTrackLocalStaticRTP(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: opusSampleRate, Channels: 2},
		"caller-voice", "caller")
	s.Require().NoError(err)
	s.connect(agentPC, sfuPC, caller)

	// Heard as the agent hears, while the packets come: the decoder reads a packet only once
	// the buffer before it is taken.
	var heard atomic.Int64
	go func() {
		for range edge.Audio() {
			heard.Add(1)
		}
	}()
	s.Require().NoError(edge.PublishAudio(speech(16_000, 4_000)))
	frame, err := silentOpusFrame()
	s.Require().NoError(err)
	// A hundred packets 20 ms apart, the SFU never forwarding 40 to 49.
	ticker := time.NewTicker(opusFrameDuration)
	defer ticker.Stop()
	for seq := range uint16(100) {
		<-ticker.C
		if seq >= 40 && seq < 50 {
			continue
		}
		s.Require().NoError(caller.WriteRTP(&rtp.Packet{Header: rtp.Header{Version: 2,
			SequenceNumber: seq, Timestamp: uint32(seq) * 960}, Payload: frame}))
	}
	s.Eventually(func() bool { return heard.Load() == 100 }, connectionUp, 10*time.Millisecond,
		"the agent heard every buffer the decoder made")
	s.Eventually(func() bool { return edge.quality.summary(nil).outboundLoss != nil }, connectionUp,
		50*time.Millisecond, "the SFU side reported twice on the agent's audio")

	before := mediaSeries()
	s.Require().NoError(edge.Leave())
	s.Require().NoError(edge.Leave())
	after := mediaSeries()

	inboundLoss := "router_agent_media_packet_loss_ratio/inbound"
	s.Equal(before[inboundLoss].count+1, after[inboundLoss].count, "one session, observed once")
	s.InDelta(0.1, after[inboundLoss].sum-before[inboundLoss].sum, 1e-9, "ten of a hundred never came")
	concealed := "router_agent_media_concealed_ratio/"
	s.InDelta(0.1, after[concealed].sum-before[concealed].sum, 1e-9, "ten of a hundred buffers were made up")
	for _, series := range []string{"router_agent_media_jitter_seconds/inbound",
		"router_agent_media_packet_loss_ratio/outbound", "router_agent_media_jitter_seconds/outbound"} {
		s.Equal(before[series].count+1, after[series].count, series)
	}
	outboundLoss := "router_agent_media_packet_loss_ratio/outbound"
	s.InDelta(0, after[outboundLoss].sum-before[outboundLoss].sum, 1e-9, "the loopback loses nothing")

	rtt := iceRoundTrip(agentPC)
	s.Require().NotNil(rtt, "the connection's checks came back")
	s.Greater(*rtt, 0.0)
	s.Less(*rtt, 0.1, "a loopback round trip is well under 100 ms")
}
