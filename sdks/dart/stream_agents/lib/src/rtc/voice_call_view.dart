// ignore: deprecated_member_use
import 'package:flutter/material.dart';

import 'voice_session.dart';

/// The controls for a spoken conversation: mute, camera, and hang up. Joins the call when it
/// is first built, and joins again when handed another voice session.
///
/// It draws no transcript. Showing what was said is the UI package's job, over the same
/// `session` the voice session holds, so a host can put the two together however it likes.
class VoiceCallView extends StatefulWidget {
  const VoiceCallView({super.key, required this.voice, this.camera = false});

  final VoiceSession voice;

  /// Whether the camera is joined with and offered as a control. Turned false, the camera is
  /// turned off.
  final bool camera;

  @override
  State<VoiceCallView> createState() => _VoiceCallViewState();
}

class _VoiceCallViewState extends State<VoiceCallView> {
  @override
  void initState() {
    super.initState();
    widget.voice.join(camera: widget.camera);
  }

  @override
  void didUpdateWidget(VoiceCallView old) {
    super.didUpdateWidget(old);
    if (!identical(widget.voice, old.voice)) {
      widget.voice.join(camera: widget.camera);
    } else if (old.camera && !widget.camera && widget.voice.state.value.isCameraEnabled) {
      widget.voice.setCameraEnabled(false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final voice = widget.voice;
    return StreamBuilder<VoiceState>(
      stream: voice.state.stream,
      initialData: voice.state.value,
      builder: (context, snapshot) {
        final state = snapshot.data!;
        final joined = state.isJoined;
        final status = state.isJoining
            ? 'joining'
            : state.isEnded
            ? 'call ended'
            : null;
        return Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (state.failure case final failure?)
              Text(
                '$failure',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.error),
              )
            else if (status != null)
              Text(
                status,
                style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.outline),
              ),
            const SizedBox(height: 16),
            Row(
              mainAxisSize: MainAxisSize.min,
              spacing: 24,
              children: [
                IconButton.filledTonal(
                  tooltip: state.isMuted ? 'Unmute' : 'Mute',
                  iconSize: 28,
                  onPressed: joined ? () => voice.setMuted(!state.isMuted) : null,
                  icon: Icon(state.isMuted ? Icons.mic_off : Icons.mic),
                ),
                if (widget.camera)
                  IconButton.filledTonal(
                    tooltip: state.isCameraEnabled ? 'Turn camera off' : 'Turn camera on',
                    iconSize: 28,
                    onPressed: joined ? () => voice.setCameraEnabled(!state.isCameraEnabled) : null,
                    icon: Icon(state.isCameraEnabled ? Icons.videocam : Icons.videocam_off),
                  ),
                IconButton.filled(
                  tooltip: 'Hang up',
                  iconSize: 28,
                  style: IconButton.styleFrom(backgroundColor: Colors.red),
                  onPressed: state.isEnded ? null : voice.end,
                  icon: const Icon(Icons.call_end),
                ),
              ],
            ),
          ],
        );
      },
    );
  }
}
