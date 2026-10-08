import 'package:flutter/material.dart';
import 'package:stream_video_flutter/stream_video_flutter.dart' as video;

import 'voice_session.dart';

/// The device camera, with the agent's annotated track inset beside it.
///
/// There is no overlay drawing here: bounding boxes are already burned into the agent's
/// published track. That track has crossed the network to the processor and back, so it lags
/// the camera by a round trip and is inset rather than filling the frame. What fills the frame
/// is this device's own camera, which is immediate. A call where the device publishes nothing
/// shows the agent's track full-frame instead.
class AgentVideoView extends StatelessWidget {
  const AgentVideoView({super.key, required this.voice});

  final VoiceSession voice;

  @override
  Widget build(BuildContext context) => StreamBuilder<VoiceState>(
    stream: voice.state.stream,
    initialData: voice.state.value,
    builder: (context, snapshot) {
      final state = snapshot.data!;
      final call = state.call;
      if (call == null) {
        return const _Waiting();
      }
      return StreamBuilder<video.CallState>(
        stream: call.state.valueStream,
        initialData: call.state.value,
        builder: (context, snapshot) {
          final participants = snapshot.data!;
          // The camera being on is what this device decided, true from the moment it was
          // turned on rather than once a track has been negotiated and reported back.
          final local = state.isCameraEnabled ? participants.localParticipant : null;
          final remote = _remoteVideo(participants.otherParticipants);
          if (local != null) {
            return Stack(
              fit: StackFit.expand,
              children: [
                _render(call, local),
                if (remote != null)
                  Align(
                    alignment: Alignment.bottomRight,
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(12),
                        child: SizedBox(width: 180, height: 240, child: _render(call, remote)),
                      ),
                    ),
                  ),
              ],
            );
          }
          if (remote != null) {
            return _render(call, remote);
          }
          return const _Waiting();
        },
      );
    },
  );

  Widget _render(video.Call call, video.CallParticipantState participant) =>
      video.StreamVideoRenderer(
        call: call,
        participant: participant,
        videoTrackType: video.SfuTrackType.video,
        videoFit: video.VideoFit.cover,
        placeholderBuilder: (_) => const _Waiting(),
      );

  /// The video worker joins as `{agent_user_id}-video`; anyone else with video is a fallback
  /// for a call where the annotated track has not arrived yet.
  static video.CallParticipantState? _remoteVideo(List<video.CallParticipantState> others) {
    for (final participant in others) {
      if (participant.userId.endsWith('-video') && participant.isVideoEnabled) {
        return participant;
      }
    }
    for (final participant in others) {
      if (participant.isVideoEnabled) {
        return participant;
      }
    }
    return null;
  }
}

class _Waiting extends StatelessWidget {
  const _Waiting();

  @override
  Widget build(BuildContext context) => const ColoredBox(
    color: Colors.black,
    child: Center(
      child: Text('waiting for video', style: TextStyle(color: Colors.white, fontSize: 12)),
    ),
  );
}
