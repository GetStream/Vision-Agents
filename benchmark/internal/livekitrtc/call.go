//go:build cgo && webrtc

package livekitrtc

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"

	"github.com/livekit/protocol/livekit"
	lksdk "github.com/livekit/server-sdk-go/v2"
	"github.com/pion/webrtc/v4"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/rtcaudio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/transport"
)

type rtcRoom struct {
	options Options
	logger  *slog.Logger
	inbound *rtcaudio.Inbound
	speaker *rtcaudio.Speaker

	room *lksdk.Room

	agentJoined     chan struct{}
	agentTrack      chan struct{}
	agentJoinedOnce sync.Once
	agentTrackOnce  sync.Once
}

func join(ctx context.Context, options Options) (transport.Media, error) {
	if options.Room == "" {
		return nil, errors.New("livekitrtc: a room is required")
	}
	if options.Identity == "" {
		options.Identity = defaultIdentity
	}
	if err := options.Resolve(); err != nil {
		return nil, err
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	c := &rtcRoom{
		options:     options,
		logger:      options.Logger.With("room", options.Room),
		inbound:     rtcaudio.NewInbound(),
		speaker:     rtcaudio.NewSpeaker(),
		agentJoined: make(chan struct{}),
		agentTrack:  make(chan struct{}),
	}
	if err := c.connect(ctx); err != nil {
		return nil, err
	}
	return c, nil
}

func (c *rtcRoom) connect(ctx context.Context) error {
	callback := &lksdk.RoomCallback{
		OnParticipantConnected: func(rp *lksdk.RemoteParticipant) {
			if rp == nil {
				return
			}
			c.logger.Info("livekit participant joined", "identity", rp.Identity())
			c.markAgentJoined()
		},
		OnParticipantDisconnected: func(rp *lksdk.RemoteParticipant) {
			if rp == nil {
				return
			}
			c.logger.Info("livekit participant left", "identity", rp.Identity())
		},
		ParticipantCallback: lksdk.ParticipantCallback{
			OnTrackSubscribed: func(track *webrtc.TrackRemote, _ *lksdk.RemoteTrackPublication, rp *lksdk.RemoteParticipant) {
				c.listen(track, rp)
			},
		},
	}
	room, err := lksdk.ConnectToRoom(c.options.URL, lksdk.ConnectInfo{
		APIKey:              c.options.APIKey,
		APISecret:           c.options.APISecret,
		RoomName:            c.options.Room,
		ParticipantName:     c.options.Identity,
		ParticipantIdentity: c.options.Identity,
	}, callback, lksdk.WithAutoSubscribe(true))
	if err != nil {
		return fmt.Errorf("livekitrtc: join %s: %w", c.options.Room, err)
	}
	c.room = room
	for _, rp := range room.GetRemoteParticipants() {
		c.logger.Info("livekit participant present", "identity", rp.Identity())
		c.markAgentJoined()
	}
	if err := c.publish(); err != nil {
		room.Disconnect()
		return err
	}
	c.logger.Info("joined the room", "identity", c.options.Identity)
	return nil
}

func (c *rtcRoom) publish() error {
	track, err := lksdk.NewLocalSampleTrack(webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus})
	if err != nil {
		return fmt.Errorf("livekitrtc: build audio track: %w", err)
	}
	// The SDK paces reads from the speaker itself, with clock-drift compensation.
	if err := track.StartWrite(c.speaker, nil); err != nil {
		return fmt.Errorf("livekitrtc: write audio track: %w", err)
	}
	if _, err := c.room.LocalParticipant.PublishTrack(track, &lksdk.TrackPublicationOptions{
		Name:   "voicebench-caller",
		Source: livekit.TrackSource_MICROPHONE,
	}); err != nil {
		return fmt.Errorf("livekitrtc: publish audio track: %w", err)
	}
	return nil
}

func (c *rtcRoom) listen(track *webrtc.TrackRemote, rp *lksdk.RemoteParticipant) {
	if track.Kind() != webrtc.RTPCodecTypeAudio {
		return
	}
	if rp != nil && rp.Identity() == c.options.Identity {
		return
	}
	if err := c.inbound.Track(track); err != nil {
		c.logger.Error("could not decode agent audio", "error", err)
		return
	}
	c.markAgentTrack()
	c.logger.Info("listening to agent audio", "track", track.ID())
}

func (c *rtcRoom) Send(pcm []int16) error {
	return c.speaker.Write(audio.PCM{Rate: audio.Rate, Samples: pcm})
}

func (c *rtcRoom) Recv() <-chan transport.Frame { return c.inbound.Recv() }

func (c *rtcRoom) Dropped() int { return c.inbound.Dropped() }

// WaitForAgent blocks until the dispatched agent joins the room and publishes audio.
func (c *rtcRoom) WaitForAgent(ctx context.Context) error {
	if err := waitFor(ctx, c.agentJoined, "livekitrtc: no agent participant joined"); err != nil {
		return err
	}
	return waitFor(ctx, c.agentTrack, "livekitrtc: agent joined but published no audio track")
}

func (c *rtcRoom) markAgentJoined() {
	c.agentJoinedOnce.Do(func() { close(c.agentJoined) })
}

func (c *rtcRoom) markAgentTrack() {
	c.agentTrackOnce.Do(func() { close(c.agentTrack) })
}

func waitFor(ctx context.Context, ch <-chan struct{}, message string) error {
	select {
	case <-ch:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("%s: %w", message, ctx.Err())
	}
}

func (c *rtcRoom) Close() error {
	if c.inbound.Closed() {
		return nil
	}
	var failures []error
	if err := c.inbound.Close(); err != nil {
		failures = append(failures, err)
	}
	if err := c.speaker.Close(); err != nil {
		failures = append(failures, err)
	}
	if c.room != nil {
		c.room.Disconnect()
	}
	return errors.Join(failures...)
}
