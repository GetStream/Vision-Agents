import 'dart:async';
import 'dart:io' show Platform;

import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';
import 'package:stream_core_flutter/core.dart';
import 'package:stream_agents_core/stream_agents_core.dart';
import 'package:stream_agents_rtc/stream_agents_rtc.dart';

import 'agent_chat.dart';
import 'captions.dart';
import 'demo.dart';
import 'talking.dart';

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

/// What the glow shows and the word under it.
///
/// [session] is what the agent is doing, or null before there is a call; [isStarting] says
/// one is on its way. Joined, whoever is [heard] comes first, then what the agent is doing.
({AIVoiceAgentState state, String status}) voicePresence({
  required bool isStarting,
  required ConversationState? session,
  required bool isJoined,
  required Heard? heard,
}) {
  if (session == null) {
    return (
      state: isStarting ? AIVoiceAgentState.connecting : AIVoiceAgentState.idle,
      status: isStarting ? 'Connecting' : ' ',
    );
  }
  if (!isJoined) {
    return (state: AIVoiceAgentState.connecting, status: 'Connecting');
  }
  if (heard case (:final isPerson, level: _)) {
    return isPerson
        ? (state: AIVoiceAgentState.listening, status: 'Listening')
        : (state: AIVoiceAgentState.speaking, status: 'Speaking');
  }
  return switch (session) {
    Listening() => (state: AIVoiceAgentState.listening, status: 'Listening'),
    Responding() => (state: AIVoiceAgentState.thinking, status: 'Thinking'),
    Working(:final skills) => (
      state: AIVoiceAgentState.thinking,
      status: describeWork(skills) ?? 'Thinking',
    ),
    _ => (state: AIVoiceAgentState.idle, status: 'Ready'),
  };
}

class _VoiceViewState extends State<VoiceView> {
  VoiceSession? _voice;
  String? _failure;
  bool _starting = false;
  final _subscriptions = <StreamSubscription<Object?>>[];

  // Who is talking and how loud, or null while nobody is. This comes from the audio rather
  // than the session: a reply is written before it is spoken and is done before the agent
  // stops talking, and the person is transcribed only after they have started.
  Heard? _heard;
  var _whoIsTalking = WhoIsTalking();
  var _captions = Captions();
  Timer? _meter;
  final _clock = Stopwatch()..start();
  var _lastTick = Duration.zero;

  // Stream's blue, which the glow takes while the agent is talking or nobody is.
  static const _agentColors = [Color(0xFF005FFF)];

  // The colors the glow takes while the person is talking.
  static const _personColors = [Color(0xFFFF2626)];

  // How often who is talking is read and the captions move on: thirty times a second.
  static const _meterInterval = Duration(milliseconds: 33);

  @override
  void dispose() {
    final voice = _voice;
    _reset();
    if (voice != null) unawaited(_leaveQuietly(voice));
    super.dispose();
  }

  Future<void> _start() async {
    setState(() {
      _starting = true;
      _failure = null;
    });
    if (Platform.isAndroid &&
        !await Permission.microphone.request().isGranted) {
      if (mounted) {
        setState(() {
          _starting = false;
          _failure = 'Talking to the agent needs the microphone.';
        });
      }
      return;
    }
    try {
      final voice = await VoiceSession.start(
        Demo.agents,
        agent: Demo.agentName,
        options: SessionOptions(tools: Demo.tools),
      );
      if (!mounted) {
        await voice.end();
        return;
      }
      _voice = voice;
      _subscriptions
        ..add(voice.session.conversation.stream.listen(_handleConversation))
        // A tool asked first, or its question was answered: the approval card comes or goes.
        ..add(voice.session.approvals.stream.listen((_) => setState(() {})))
        ..add(voice.state.stream.listen(_handleVoiceState));
      await voice.join();
      // Let go of while it joined, because the conversation ended or the screen went away:
      // leaving then found no call yet, so the call it just joined is left here.
      if (!identical(_voice, voice)) await _leaveQuietly(voice);
    } on Object catch (error) {
      // A join that throws rather than failing leaves no call to hang up: let go of the
      // session, so the failure shows and the person can try again. Set first, so a failure
      // to let go cannot hide it.
      if (mounted) setState(() => _failure = '$error');
      await _leave();
    } finally {
      if (mounted) setState(() => _starting = false);
    }
  }

  void _handleVoiceState(VoiceState state) {
    if (state.call != null && _meter == null) {
      _lastTick = _clock.elapsed;
      _meter = Timer.periodic(_meterInterval, (_) => _tick());
    }
    // Joined, muted or failed: the controls follow.
    setState(() {});
  }

  // Reads who is talking, and moves the captions on.
  void _tick() {
    final voice = _voice;
    if (voice == null) return;
    final now = _clock.elapsed;
    final elapsed = now - _lastTick;
    _lastTick = now;

    final participants = voice.call?.state.value.callParticipants ?? const [];
    final person = participants.where((p) => p.isLocal).firstOrNull;
    final agent = participants
        .where((p) => !p.isLocal && p.isAudioEnabled)
        .firstOrNull;
    final agentIsAudible = agent?.isSpeaking ?? false;
    final heard = _whoIsTalking.hear(
      agentIsSpeaking: agentIsAudible,
      agentLevel: agent?.audioLevel ?? 0.0,
      personIsSpeaking: person?.isSpeaking ?? false,
      personLevel: person?.audioLevel ?? 0.0,
      now: now,
    );

    final caption = _captions.caption;
    final captionIsVisible = _captions.isVisible;
    _captions.update(
      turn: voice.session.turns.lastOrNull,
      agentIsAudible: agentIsAudible,
      agentIsAnswering: switch (voice.session.state) {
        Responding() || Working() => true,
        _ => false,
      },
      now: now,
      elapsed: elapsed,
    );
    if (heard != _heard ||
        caption != _captions.caption ||
        captionIsVisible != _captions.isVisible) {
      setState(() => _heard = heard);
    }
  }

  void _handleConversation(Conversation conversation) {
    // The agent leaving is the call being over, so let go of it and offer to start another
    // rather than leaving dead controls on screen.
    if (conversation.state is Ended) {
      // A dropped socket ends the conversation too, and is said rather than looking like the
      // agent hanging up.
      if (_voice?.session.failure case final failure?) {
        setState(() => _failure = '$failure');
      }
      unawaited(_leave());
      return;
    }
    // What the agent is doing changed, which the status under the glow says.
    setState(() {});
  }

  Future<void> _leave() async {
    final voice = _voice;
    _reset();
    if (mounted) setState(() {});
    if (voice != null) await _leaveQuietly(voice);
  }

  // Hangs up [voice], and lets go of it if it is still the call on screen: by the time the
  // hang-up finishes, the person may have started another.
  Future<void> _hangUp(VoiceSession voice) async {
    try {
      await voice.end();
    } on Object catch (error) {
      debugPrint('Hanging up failed: $error');
    }
    if (identical(_voice, voice)) await _leave();
  }

  // Leaves [voice]'s call. Nothing waits on it, so a failure is logged rather than thrown.
  static Future<void> _leaveQuietly(VoiceSession voice) async {
    try {
      await voice.leave();
    } on Object catch (error) {
      debugPrint('Leaving the call failed: $error');
    }
  }

  // Forgets the call and everything read from it.
  void _reset() {
    _voice = null;
    for (final subscription in _subscriptions) {
      unawaited(subscription.cancel());
    }
    _subscriptions.clear();
    _meter?.cancel();
    _meter = null;
    _heard = null;
    _whoIsTalking = WhoIsTalking();
    _captions = Captions();
  }

  bool get _inCall => _voice?.state.value.call != null;

  ({AIVoiceAgentState state, String status}) get _presence => voicePresence(
    isStarting: _starting,
    session: _voice?.session.state,
    isJoined: _inCall,
    heard: _heard,
  );

  @override
  Widget build(BuildContext context) {
    final voice = _voice;
    final presence = _presence;
    final level = _heard?.level;
    final colors = _heard?.isPerson ?? false ? _personColors : _agentColors;
    final scheme = context.streamColorScheme;
    final text = context.streamTextTheme;
    final request = voice?.session.approvals.value.firstOrNull;

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
              style: text.bodyEmphasis.copyWith(color: scheme.textSecondary),
            ),
          ),
          const Spacer(),
          CaptionView(
            caption: _captions.caption,
            isVisible: _captions.isVisible,
          ),
          if (voice != null && request != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              // Keyed by the tool call, so a second question starts with no answer sending.
              child: _ApprovalCard(
                key: ValueKey(request.id),
                request: request,
                session: voice.session,
              ),
            ),
          if (voice == null && _failure != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Text(
                _failure!,
                textAlign: TextAlign.center,
                style: text.captionDefault.copyWith(color: scheme.accentError),
              ),
            ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: voice == null
                ? StreamButton(
                    size: .large,
                    onPressed: _starting ? null : _start,
                    iconLeft: Icon(context.streamIcons.phoneFill),
                    child: const Text('Talk to the agent'),
                  )
                : _CallControls(
                    voice: voice,
                    onHangUp: () => unawaited(_hangUp(voice)),
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
    final state = voice.state.value;
    final muted = state.isMuted;
    final icons = context.streamIcons;
    return Column(
      mainAxisSize: MainAxisSize.min,
      spacing: 16,
      children: [
        if (state.failure case final failure?)
          Text(
            '$failure',
            textAlign: TextAlign.center,
            style: context.streamTextTheme.captionDefault.copyWith(
              color: context.streamColorScheme.accentError,
            ),
          ),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          spacing: 24,
          children: [
            StreamButton.icon(
              style: .secondary,
              size: .large,
              isSelected: muted,
              tooltip: muted ? 'Unmute' : 'Mute',
              onPressed: state.isJoined ? () => voice.setMuted(!muted) : null,
              icon: Icon(muted ? icons.voiceOffFill : icons.voiceFill),
            ),
            StreamButton.icon(
              style: .destructive,
              size: .large,
              tooltip: 'Hang up',
              // Not until the call is joined, or failed to: leaving while it is still being
              // joined would let the join land after it, in a call nobody can leave.
              onPressed: state.isJoined || state.failure != null
                  ? onHangUp
                  : null,
              icon: Icon(icons.phoneDownFill),
            ),
          ],
        ),
      ],
    );
  }
}

/// A tool's question, asked over the call with Stream's approval card. The tool runs only
/// once it is allowed, and the agent is told when it is not.
class _ApprovalCard extends StatefulWidget {
  const _ApprovalCard({
    super.key,
    required this.request,
    required this.session,
  });

  final ToolApprovalRequest request;
  final AgentSession session;

  @override
  State<_ApprovalCard> createState() => _ApprovalCardState();
}

class _ApprovalCardState extends State<_ApprovalCard> {
  AIToolApprovalState _state = const AIToolApprovalIdle();

  // Sending lasts until the tool call leaves the session's approvals, which takes the card
  // away.
  void _handleAnswer(AIToolApprovalDecision decision) {
    setState(() => _state = AIToolApprovalSending(decision));
    try {
      widget.session.decide(
        widget.request.id,
        allowed: decision == AIToolApprovalDecision.allowed,
      );
    } on StateError catch (error) {
      setState(() => _state = AIToolApprovalFailed(decision, error));
    } on AgentsException catch (error) {
      // The socket closed under the card, so the answer cannot go; the person may try again.
      setState(() => _state = AIToolApprovalFailed(decision, error));
    }
  }

  @override
  Widget build(BuildContext context) {
    final request = widget.request;
    return AIToolApprovalCard(
      approval: AIToolApproval(
        title: request.approval.title,
        message: request.approval.message,
        reason: request.reason.isEmpty ? null : request.reason,
        allowTitle: request.approval.allowTitle,
        declineTitle: request.approval.declineTitle,
      ),
      state: _state,
      onAnswer: _handleAnswer,
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
    final colors = context.streamColorScheme;
    final text = context.streamTextTheme;
    final spacing = context.streamSpacing;
    final radius = BorderRadius.all(context.streamRadius.lg);
    return SafeArea(
      top: false,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          StreamSheetHeader(title: const Text('Voice style')),
          Padding(
            padding: EdgeInsets.fromLTRB(spacing.md, 0, spacing.md, spacing.md),
            child: Row(
              spacing: spacing.md,
              children: [
                for (final option in VoiceStyle.values)
                  Expanded(
                    child: Semantics(
                      selected: option == style,
                      button: true,
                      child: InkWell(
                        borderRadius: radius,
                        onTap: () => onChanged(option),
                        child: Container(
                          padding: EdgeInsets.all(spacing.sm),
                          decoration: BoxDecoration(
                            borderRadius: radius,
                            border: Border.all(
                              color: option == style
                                  ? colors.accentPrimary
                                  : colors.borderDefault,
                              width: option == style ? 2 : 1,
                            ),
                          ),
                          child: Column(
                            spacing: spacing.xs,
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
                                      context.streamIcons.checkmark,
                                      size: 18,
                                      color: colors.accentPrimary,
                                    ),
                                  Text(
                                    option.label,
                                    style: text.bodyEmphasis.copyWith(
                                      color: colors.textPrimary,
                                    ),
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
          ),
        ],
      ),
    );
  }
}
