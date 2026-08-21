//go:build cgo && webrtc

package streamrtc

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"sync"

	videosdk "github.com/GetStream/getstream-go-webrtc"
	sdktrack "github.com/GetStream/getstream-go-webrtc/track"
	sfu_events "github.com/GetStream/protocol/protobuf/video/sfu/event"
	sfu_models "github.com/GetStream/protocol/protobuf/video/sfu/models"
	"github.com/GetStream/protocol/protobuf/video/sfu/signal_rpc"
	"github.com/google/uuid"
	"github.com/pion/webrtc/v4"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/rtcaudio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/transport"
)

type rtcCall struct {
	options Options
	logger  *slog.Logger
	inbound *rtcaudio.Inbound
	speaker *rtcaudio.Speaker

	client *videosdk.Client
	call   *videosdk.Call

	mu             sync.Mutex
	subscribed     []*signal_rpc.TrackSubscriptionDetails
	unregister     func()
	closed         bool
	agentTrack     chan struct{}
	agentTrackOnce sync.Once
}

func join(ctx context.Context, options Options) (transport.Media, error) {
	if options.CallID == "" {
		return nil, errors.New("streamrtc: a call id is required")
	}
	if options.CallType == "" {
		options.CallType = defaultCallType
	}
	if options.UserID == "" {
		options.UserID = defaultUserID
	}
	if options.UserName == "" {
		options.UserName = options.UserID
	}
	if err := options.Resolve(); err != nil {
		return nil, err
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}

	c := &rtcCall{
		options:    options,
		logger:     options.Logger.With("call", options.CallType+":"+options.CallID),
		inbound:    rtcaudio.NewInbound(),
		speaker:    rtcaudio.NewSpeaker(),
		agentTrack: make(chan struct{}),
	}
	if err := c.connect(ctx); err != nil {
		return nil, err
	}
	return c, nil
}

func (c *rtcCall) connect(ctx context.Context) error {
	user := videosdk.User{ID: c.options.UserID, Name: c.options.UserName}
	var client *videosdk.Client
	var err error
	if c.options.UserToken != "" {
		client, err = videosdk.NewClient(c.options.APIKey, user, videosdk.StaticToken(c.options.UserToken))
	} else {
		client, err = videosdk.NewClientWithSecret(c.options.APIKey, c.options.APISecret, user)
	}
	if err != nil {
		return fmt.Errorf("streamrtc: connect: %w", err)
	}
	c.client = client
	c.call = client.Call(c.options.CallType, c.options.CallID)

	joined, err := c.call.Join(ctx, videosdk.WithOnTrack(videosdk.SubscriberFunc(func(remote videosdk.OnTrackReceived) {
		c.listen(remote)
	})))
	if err != nil {
		return fmt.Errorf("streamrtc: join %s:%s: %w", c.options.CallType, c.options.CallID, err)
	}

	c.mu.Lock()
	c.subscribed = audioSubscriptions(joined.GetCallState(), c.options.UserID)
	subscriptions := slices.Clone(c.subscribed)
	c.mu.Unlock()
	if err := c.call.SubscribeToTracks(ctx, subscriptions...); err != nil {
		return fmt.Errorf("streamrtc: subscribe: %w", err)
	}
	c.watchForNewTracks(ctx)
	if err := c.publish(); err != nil {
		return err
	}
	c.logger.Info("joined the call", "user", c.options.UserID, "tracks", len(subscriptions))
	return nil
}

func (c *rtcCall) publish() error {
	info := &sfu_models.TrackInfo{
		TrackId:   uuid.NewString(),
		TrackType: sfu_models.TrackType_TRACK_TYPE_AUDIO,
	}
	voice, err := sdktrack.NewAudioTrack(info, c.speaker, webrtc.RTPCodecCapability{
		MimeType:  webrtc.MimeTypeOpus,
		ClockRate: rtcaudio.OpusSampleRate,
		Channels:  opusNegotiatedChannels,
	})
	if err != nil {
		return fmt.Errorf("streamrtc: build audio track: %w", err)
	}
	if _, err := c.call.AddTrack(info, voice); err != nil {
		return fmt.Errorf("streamrtc: publish audio track: %w", err)
	}
	return nil
}

func (c *rtcCall) listen(remote videosdk.OnTrackReceived) {
	if remote.TrackType != sfu_models.TrackType_TRACK_TYPE_AUDIO {
		return
	}
	if string(remote.ParticipantID.UserID) == c.options.UserID {
		return
	}
	if err := c.inbound.Track(remote.Track); err != nil {
		c.logger.Error("could not decode agent audio", "error", err)
		return
	}
	c.agentTrackOnce.Do(func() { close(c.agentTrack) })
	c.logger.Info("listening to agent audio")
}

func (c *rtcCall) watchForNewTracks(ctx context.Context) {
	unregister := videosdk.HandleCallEvent(c.call, func(event *sfu_events.SfuEvent_TrackPublished) {
		published := event.TrackPublished
		if published.GetUserId() == c.options.UserID {
			return
		}
		if published.GetType() != sfu_models.TrackType_TRACK_TYPE_AUDIO {
			return
		}
		c.mu.Lock()
		if c.closed {
			c.mu.Unlock()
			return
		}
		c.subscribed = append(c.subscribed, &signal_rpc.TrackSubscriptionDetails{
			UserId:    published.GetUserId(),
			SessionId: published.GetSessionId(),
			TrackType: published.GetType(),
		})
		subscriptions := slices.Clone(c.subscribed)
		c.mu.Unlock()
		if err := c.call.SubscribeToTracks(ctx, subscriptions...); err != nil {
			c.logger.Error("could not subscribe to a new track", "error", err)
		}
	})
	c.mu.Lock()
	c.unregister = unregister
	c.mu.Unlock()
}

func audioSubscriptions(state *sfu_models.CallState, selfUserID string) []*signal_rpc.TrackSubscriptionDetails {
	var subscriptions []*signal_rpc.TrackSubscriptionDetails
	for _, participant := range state.GetParticipants() {
		if participant.GetUserId() == selfUserID {
			continue
		}
		for _, trackType := range participant.GetPublishedTracks() {
			if trackType != sfu_models.TrackType_TRACK_TYPE_AUDIO {
				continue
			}
			subscriptions = append(subscriptions, &signal_rpc.TrackSubscriptionDetails{
				UserId:    participant.GetUserId(),
				SessionId: participant.GetSessionId(),
				TrackType: trackType,
			})
		}
	}
	return subscriptions
}

func (c *rtcCall) Send(pcm []int16) error {
	return c.speaker.Write(audio.PCM{Rate: audio.Rate, Samples: pcm})
}

func (c *rtcCall) Recv() <-chan transport.Frame { return c.inbound.Recv() }

func (c *rtcCall) Dropped() int { return c.inbound.Dropped() }

// WaitForAgent blocks until the target publishes an audio track.
func (c *rtcCall) WaitForAgent(ctx context.Context) error {
	select {
	case <-c.agentTrack:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("streamrtc: agent published no audio track: %w", ctx.Err())
	}
}

func (c *rtcCall) Close() error {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil
	}
	c.closed = true
	unregister := c.unregister
	c.mu.Unlock()

	if unregister != nil {
		unregister()
	}
	var failures []error
	if err := c.inbound.Close(); err != nil {
		failures = append(failures, err)
	}
	if err := c.speaker.Close(); err != nil {
		failures = append(failures, err)
	}
	if c.call != nil {
		if err := c.call.Leave("voicebench finished"); err != nil {
			failures = append(failures, err)
		}
	}
	if c.client != nil {
		c.client.Close()
	}
	return errors.Join(failures...)
}
