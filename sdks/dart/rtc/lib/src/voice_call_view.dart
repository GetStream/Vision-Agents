import 'package:flutter/material.dart';

import 'voice_session.dart';

/// The controls for a spoken conversation: mute, camera, and hang up. Joins the call when it
/// is first built.
///
/// It draws no transcript. Showing what was said is the UI package's job, over the same
/// `session` the voice session holds, so a host can put the two together however it likes.
class VoiceCallView extends StatefulWidget {
  /// [credentials] is asked for the token this device joins the call with. Minting one is
  /// server-side only, so it comes from the app's own backend.
  const VoiceCallView({
    super.key,
    required this.voice,
    required this.credentials,
    this.camera = false,
  });

  final VoiceSession voice;
  final CallCredentialsProvider credentials;
  final bool camera;

  @override
  State<VoiceCallView> createState() => _VoiceCallViewState();
}

class _VoiceCallViewState extends State<VoiceCallView> {
  @override
  void initState() {
    super.initState();
    widget.voice.join(camera: widget.camera, credentials: widget.credentials);
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
        return Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (state.failure case final failure?)
              Text(
                '$failure',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.error),
              )
            else if (!joined)
              Text(
                'joining',
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
                  onPressed: voice.end,
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
