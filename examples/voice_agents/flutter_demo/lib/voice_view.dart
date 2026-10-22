import 'dart:async';

import 'package:flutter/material.dart';
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';
import 'package:stream_video_flutter/stream_video_flutter.dart' as video;
import 'package:vision_agents_core/vision_agents_core.dart';
import 'package:vision_agents_rtc/vision_agents_rtc.dart';

import 'demo.dart';

/// How the agent is shown on a call.
enum VoiceStyle {
  orb('Orb'),
  wave('Wave');

  const VoiceStyle(this.label);

  final String label;
}

/// A spoken conversation, with the agent shown as Stream's voice glow and what is said
/// captioned underneath.
///
/// Tapping talk does two things: the router starts a session, which puts the agent on a
/// call, and this device joins it as the user the agents were set up with, with the same
/// token. The glow shows what the agent is doing: connecting while the call is joined,
/// thinking while it answers, and moving with whoever is talking. A tool that asks first
/// asks here, over the call, and the agent waits for the answer.
class VoiceView extends StatefulWidget {
  const VoiceView({super.key, required this.style});

  final VoiceStyle style;

  @override
  State<VoiceView> createState() => _VoiceViewState();
}

class _VoiceViewState extends State<VoiceView> {
  VoiceSession? _voice;
  String? _failure;
  bool _starting = false;
  final _subscriptions = <StreamSubscription<Object?>>[];

  // Who is talking, from the call's audio levels rather than the session: a reply is written
  // before it is spoken and done before the agent stops talking.
  video.CallParticipantState? _agent;
  video.CallParticipantState? _person;

  @override
  void dispose() {
    unawaited(_leave());
    super.dispose();
  }

  Future<void> _start() async {
    setState(() {
      _starting = true;
      _failure = null;
    });
    try {
      final voice = await VoiceSession.start(
        Demo.agents,
        agent: Demo.agentName,
        options: SessionOptions(
          tools: Demo.tools,
          tts: 'inworld/inworld-tts-2-flash',
        ),
      );
      if (!mounted) {
        await voice.end();
        return;
      }
      _voice = voice;
      _subscriptions
        ..add(voice.session.conversation.stream.listen(_conversationChanged))
        ..add(voice.session.approvals.stream.listen((_) => setState(() {})))
        ..add(voice.state.stream.listen(_voiceChanged));
      await voice.join();
    } on Object catch (error) {
      if (mounted) setState(() => _failure = '$error');
    } finally {
      if (mounted) setState(() => _starting = false);
    }
  }

  StreamSubscription<video.CallState>? _participants;

  void _voiceChanged(VoiceState state) {
    final call = state.call;
    if (call != null && _participants == null) {
      _participants = call.state.valueStream.listen((callState) {
        setState(() {
          _person = callState.callParticipants
              .where((p) => p.isLocal)
              .firstOrNull;
          _agent = callState.callParticipants
              .where((p) => !p.isLocal)
              .firstOrNull;
        });
      });
    }
    setState(() {});
  }

  void _conversationChanged(Conversation conversation) {
    // The agent leaving is the call being over, so let go of it and offer to start another
    // rather than leaving dead controls on screen.
    if (conversation.state is Ended) {
      unawaited(_leave());
      return;
    }
    setState(() {});
  }

  Future<void> _leave() async {
    final voice = _voice;
    _voice = null;
    for (final subscription in _subscriptions) {
      unawaited(subscription.cancel());
    }
    _subscriptions.clear();
    unawaited(_participants?.cancel());
    _participants = null;
    _agent = null;
    _person = null;
    if (mounted) setState(() {});
    await voice?.leave();
  }

  bool get _inCall => _voice?.state.value.call != null;

  ({AIVoiceAgentState state, String status}) get _presence {
    final voice = _voice;
    if (voice == null) {
      return (
        state: _starting
            ? AIVoiceAgentState.connecting
            : AIVoiceAgentState.idle,
        status: _starting ? 'Connecting' : ' ',
      );
    }
    if (!_inCall) {
      return (state: AIVoiceAgentState.connecting, status: 'Connecting');
    }
    if (_agent?.isSpeaking ?? false) {
      return (state: AIVoiceAgentState.speaking, status: 'Speaking');
    }
    if (_person?.isSpeaking ?? false) {
      return (state: AIVoiceAgentState.listening, status: 'Listening');
    }
    return switch (voice.session.state) {
      Listening() => (state: AIVoiceAgentState.listening, status: 'Listening'),
      Responding() => (state: AIVoiceAgentState.thinking, status: 'Thinking'),
      Working(:final skills) => (
        state: AIVoiceAgentState.thinking,
        status:
            'Working on ${skills.map((s) => s.replaceAll('_', ' ')).join(' and ')}',
      ),
      _ => (state: AIVoiceAgentState.idle, status: 'Ready'),
    };
  }

  /// Stream's blue, which the glow takes while the agent is talking or nobody is.
  static const _agentColors = [Color(0xFF005FFF)];

  /// The colors the glow takes while the person is talking.
  static const _personColors = [Color(0xFFFF2626)];

  @override
  Widget build(BuildContext context) {
    final voice = _voice;
    final presence = _presence;
    final personTalking =
        !(_agent?.isSpeaking ?? false) && (_person?.isSpeaking ?? false);
    final level = _agent?.isSpeaking ?? false
        ? _agent!.audioLevel
        : personTalking
        ? _person!.audioLevel
        : null;
    final colors = personTalking ? _personColors : _agentColors;
    final theme = Theme.of(context);
    final request = voice?.session.approvals.value.firstOrNull;
    final caption = voice?.session.turns.lastOrNull?.text;

    return SafeArea(
      child: Column(
        children: [
          const Spacer(),
          ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 320, maxHeight: 320),
            child: AspectRatio(
              aspectRatio: 1,
              child: switch (widget.style) {
                VoiceStyle.orb => AIVoiceGlowView(
                  state: presence.state,
                  colors: colors,
                  audioLevel: level,
                ),
                VoiceStyle.wave => AIVoiceWaveView(
                  state: presence.state,
                  colors: colors,
                  audioLevel: level,
                ),
              },
            ),
          ),
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 200),
            child: Text(
              presence.status,
              key: ValueKey(presence.status),
              style: theme.textTheme.titleSmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ),
          const Spacer(),
          if (caption != null && caption.isNotEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
              child: Text(
                caption,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                textAlign: TextAlign.center,
              ),
            ),
          if (voice != null && request != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: AIToolApprovalCard(
                key: ValueKey(request.id),
                approval: AIToolApproval(
                  title: request.approval.title,
                  message: request.approval.message,
                  reason: request.reason.isEmpty ? null : request.reason,
                  allowTitle: request.approval.allowTitle,
                  declineTitle: request.approval.declineTitle,
                ),
                onAnswer: (decision) => voice.session.decide(
                  request.id,
                  allowed: decision == AIToolApprovalDecision.allowed,
                ),
              ),
            ),
          if (_failure case final failure?)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Text(
                failure,
                textAlign: TextAlign.center,
                style: TextStyle(color: theme.colorScheme.error, fontSize: 12),
              ),
            ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: voice == null
                ? FilledButton.icon(
                    onPressed: _starting ? null : _start,
                    icon: const Icon(Icons.call),
                    label: const Text('Talk to the agent'),
                  )
                : _CallControls(
                    voice: voice,
                    onHangUp: () => unawaited(voice.end().whenComplete(_leave)),
                  ),
          ),
        ],
      ),
    );
  }
}

class _CallControls extends StatelessWidget {
  const _CallControls({required this.voice, required this.onHangUp});

  final VoiceSession voice;
  final VoidCallback onHangUp;

  @override
  Widget build(BuildContext context) {
    final muted = voice.state.value.isMuted;
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      spacing: 24,
      children: [
        IconButton.filledTonal(
          iconSize: 28,
          tooltip: muted ? 'Unmute' : 'Mute',
          onPressed: () => voice.setMuted(!muted),
          icon: Icon(muted ? Icons.mic_off : Icons.mic),
        ),
        IconButton.filled(
          iconSize: 28,
          tooltip: 'Hang up',
          style: IconButton.styleFrom(
            backgroundColor: Colors.red,
            foregroundColor: Colors.white,
          ),
          onPressed: onHangUp,
          icon: const Icon(Icons.call_end),
        ),
      ],
    );
  }
}

/// The orb and the wave, each moving as it would on a call, to pick how the agent is shown.
class VoiceStyleSheet extends StatelessWidget {
  const VoiceStyleSheet({
    super.key,
    required this.style,
    required this.onChanged,
  });

  final VoiceStyle style;
  final ValueChanged<VoiceStyle> onChanged;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        spacing: 16,
        children: [
          Text('Voice style', style: theme.textTheme.titleLarge),
          Row(
            spacing: 16,
            children: [
              for (final option in VoiceStyle.values)
                Expanded(
                  child: Semantics(
                    selected: option == style,
                    button: true,
                    child: InkWell(
                      borderRadius: BorderRadius.circular(12),
                      onTap: () => onChanged(option),
                      child: Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: option == style
                                ? theme.colorScheme.primary
                                : theme.colorScheme.outlineVariant,
                            width: option == style ? 2 : 1,
                          ),
                        ),
                        child: Column(
                          spacing: 8,
                          children: [
                            AspectRatio(
                              aspectRatio: 1,
                              child: switch (option) {
                                VoiceStyle.orb => const AIVoiceGlowView(
                                  state: AIVoiceAgentState.speaking,
                                ),
                                VoiceStyle.wave => const AIVoiceWaveView(
                                  state: AIVoiceAgentState.speaking,
                                ),
                              },
                            ),
                            Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              spacing: 6,
                              children: [
                                if (option == style)
                                  Icon(
                                    Icons.check_circle,
                                    size: 18,
                                    color: theme.colorScheme.primary,
                                  ),
                                Text(
                                  option.label,
                                  style: theme.textTheme.titleSmall,
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }
}
