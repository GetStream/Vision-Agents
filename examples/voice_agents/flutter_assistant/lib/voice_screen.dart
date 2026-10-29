import 'dart:async';

import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';
import 'package:stream_video_flutter/stream_video_flutter.dart' as video;
import 'package:vision_agents_core/vision_agents_core.dart';
import 'package:vision_agents_rtc/vision_agents_rtc.dart';

import 'agent_chat.dart';
import 'assistant.dart';
import 'captions.dart';
import 'permissions.dart';

/// Talking to Juno, full screen, in the conversation that was open.
///
/// Voice is started on the chat's own session, so the agent joins its call with everything
/// said so far, and what is said on the call is written into the same channel: the chat shows
/// it once voice mode is closed. The chat's socket stays open the whole time and is what runs
/// this phone's tools, so the call is followed with a second, tool-less session of its own.
///
/// The camera button shows Juno what the phone is pointing at, through the back lens, with a
/// preview of it.
class VoiceScreen extends StatefulWidget {
  const VoiceScreen({super.key, required this.chat});

  final AgentChat chat;

  @override
  State<VoiceScreen> createState() => _VoiceScreenState();
}

class _VoiceScreenState extends State<VoiceScreen> {
  VoiceSession? _voice;
  String? _failure;
  final _subscriptions = <StreamSubscription<Object?>>[];
  StreamSubscription<video.CallState>? _participants;

  // The call's participants, whose audio levels say who is talking.
  video.CallParticipantState? _agent;
  video.CallParticipantState? _person;

  /// The turns the conversation held when the call started.
  ///
  /// The call carries on the chat's conversation, so what was written before it is already there; only what is said on the
  /// call moves the glow and the captions.
  Set<String> _before = const {};

  /// How long after the agent was last heard the microphone is still ignored, so what is left
  /// of its voice after echo cancellation never reads as the person.
  static const _echoTail = Duration(milliseconds: 300);

  /// How long the person has to keep talking to count, so room noise does not.
  static const _onset = Duration(milliseconds: 120);

  /// Who is talking and how loud, or null while nobody is.
  ///
  /// This comes from the audio rather than the session: a reply is written before it is spoken and is done before the agent
  /// stops talking, and the person is transcribed only after they have started.
  ({bool isPerson, double level})? _heard;
  final _captions = Captions();
  Timer? _meter;
  final _clock = Stopwatch()..start();
  var _lastTick = Duration.zero;
  Duration? _agentLastHeard;
  Duration? _personSince;

  /// What went wrong on the agent's side of the call, in words for the person, until it next
  /// hears, answers or speaks.
  String? _trouble;

  /// Starting voice, which leaving waits for: the agent is only taken off the call once it has
  /// been put on it.
  late final Future<void> _starting;

  @override
  void initState() {
    super.initState();
    _starting = _start();
  }

  @override
  void dispose() {
    for (final subscription in _subscriptions) {
      unawaited(subscription.cancel());
    }
    unawaited(_participants?.cancel());
    _meter?.cancel();
    final session = widget.chat.session;
    unawaited(_starting.whenComplete(() => _hangUp(_voice, session)));
    super.dispose();
  }

  Future<void> _start() async {
    if (!await Permissions.allOf([Permission.microphone])) {
      if (mounted) {
        setState(() => _failure = 'Talking to Juno needs the microphone.');
      }
      return;
    }
    final chat = widget.chat;
    try {
      await chat.session.startVoice();
      if (!mounted) return;
      final voice = _voice = await VoiceSession.attach(Assistant.agents, chat.session.id);
      if (!mounted) return;
      _subscriptions
        ..add(voice.session.conversation.stream.listen(_conversationChanged))
        ..add(voice.session.events().listen(_eventReceived))
        // The card over the call shows the chat session's first question.
        ..add(chat.session.approvals.stream.listen((_) => setState(() {})))
        ..add(voice.state.stream.listen(_voiceChanged));
      // The call's session is attached: the controls and the status read it.
      setState(() {});
      await voice.join();
    } on Object catch (error) {
      if (mounted) setState(() => _failure = '$error');
    }
  }

  /// Leaves [voice]'s call and takes the agent off [chat]'s session, which carries the
  /// conversation on in writing.
  ///
  /// Leaving an attached call keeps the session, and so does not closing the call's own
  /// session: closing one ends the session, the chat's included.
  static Future<void> _hangUp(VoiceSession? voice, AgentSession chat) async {
    await voice?.leave();
    try {
      await chat.stopVoice();
    } on Object {
      // The agent has already left.
    }
  }

  void _voiceChanged(VoiceState state) {
    final call = state.call;
    if (call != null && _participants == null) {
      _before = {for (final turn in _voice!.session.turns) turn.id};
      _participants = call.state.valueStream.listen((callState) {
        _person = callState.callParticipants.where((p) => p.isLocal).firstOrNull;
        _agent = callState.callParticipants.where((p) => !p.isLocal).firstOrNull;
      });
      _lastTick = _clock.elapsed;
      _meter = Timer.periodic(const Duration(milliseconds: 33), (_) => _tick());
    }
    // The call, its camera and its failure are read from the voice state in build.
    setState(() {});
  }

  /// Reads who is talking, and moves the captions on, thirty times a second.
  void _tick() {
    final voice = _voice;
    if (voice == null) return;
    final now = _clock.elapsed;
    final elapsed = now - _lastTick;
    _lastTick = now;

    final agent = _agent;
    final person = _person;
    final agentIsAudible = agent?.isSpeaking ?? false;
    _personSince = person?.isSpeaking ?? false ? _personSince ?? now : null;
    final personSince = _personSince;
    final agentLastHeard = _agentLastHeard;
    // The agent first: while its voice plays, the microphone hears it too.
    final ({bool isPerson, double level})? heard;
    if (agentIsAudible) {
      _agentLastHeard = now;
      heard = (isPerson: false, level: agent!.audioLevel);
    } else if (personSince != null &&
        now - personSince >= _onset &&
        (agentLastHeard == null || now - agentLastHeard > _echoTail)) {
      heard = (isPerson: true, level: person!.audioLevel);
    } else {
      heard = null;
    }

    final turn = voice.session.turns.lastOrNull;
    final caption = _captions.caption;
    final captionIsVisible = _captions.isVisible;
    _captions.update(
      turn: turn == null || _before.contains(turn.id) ? null : turn,
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

  void _conversationChanged(Conversation conversation) {
    // The agent leaving is the call being over, which goes back to the chat.
    if (conversation.state is Ended) {
      _close();
      return;
    }
    // The status says what the agent is doing, from the conversation's state.
    setState(() {});
  }

  /// Shows what failed on the agent's side by the part of it that failed, so the person knows
  /// whether they were not heard or not answered, and clears it once the agent hears, answers
  /// or speaks again.
  void _eventReceived(AgentEvent event) {
    switch (event.kind) {
      case AgentEventKind.error:
        setState(() => _trouble = _troubleText('${event['context'] ?? ''}'));
      case AgentEventKind.heard || AgentEventKind.responding || AgentEventKind.spoke:
        if (_trouble != null) setState(() => _trouble = null);
      default:
        break;
    }
  }

  static String _troubleText(String context) => switch (context) {
    'stt' => "Juno can't hear you right now: speech recognition isn't working.",
    'llm' || 'sts' => "Juno can't answer right now: its model isn't working.",
    'tts' => "Juno can't speak right now: its voice isn't working.",
    _ => "Something went wrong on Juno's side.",
  };

  /// Goes back to the chat, once: the agent ending the call can be heard more than once, and
  /// the person may already be on their way back.
  void _close() {
    if (mounted && (ModalRoute.of(context)?.isCurrent ?? false)) {
      Navigator.of(context).pop();
    }
  }

  Future<void> _toggleCamera(VoiceSession voice) async {
    final enabled = voice.state.value.isCameraEnabled;
    if (!enabled && !await Permissions.allOf([Permission.camera])) {
      if (mounted) {
        setState(() => _failure = 'Showing Juno the camera needs the camera.');
      }
      return;
    }
    if (_failure != null) setState(() => _failure = null);
    await voice.setCameraEnabled(!enabled);
  }

  bool get _inCall => _voice?.state.value.call != null;

  ({AIVoiceAgentState state, String status}) get _presence {
    final voice = _voice;
    if (voice == null) {
      return _failure == null
          ? (state: AIVoiceAgentState.connecting, status: 'Connecting')
          : (state: AIVoiceAgentState.idle, status: ' ');
    }
    if (!_inCall) {
      return (state: AIVoiceAgentState.connecting, status: 'Connecting');
    }
    switch (_heard?.isPerson) {
      case true:
        return (state: AIVoiceAgentState.listening, status: 'Listening');
      case false:
        return (state: AIVoiceAgentState.speaking, status: 'Speaking');
      case null:
        break;
    }
    return switch (voice.session.state) {
      Listening() => (state: AIVoiceAgentState.listening, status: 'Listening'),
      Responding() => (state: AIVoiceAgentState.thinking, status: 'Thinking'),
      Working(:final skills) => (
        state: AIVoiceAgentState.thinking,
        status: AgentChat.workingOnSkills(skills),
      ),
      _ => (state: AIVoiceAgentState.idle, status: 'Ready'),
    };
  }

  /// Stream's blue, which the glow takes while the agent is talking or nobody is.
  static const _agentColors = [Color(0xFF005FFF)];

  /// The colors the glow takes while the person holding the phone is talking.
  static const _personColors = [Color(0xFFFF2626)];

  @override
  Widget build(BuildContext context) {
    final voice = _voice;
    final voiceState = voice?.state.value;
    final call = voiceState?.call;
    final presence = _presence;
    final theme = Theme.of(context);
    final request = widget.chat.session.approvals.value.firstOrNull;
    final failure = _failure ?? voiceState?.failure?.toString() ?? _trouble;

    return Scaffold(
      body: SafeArea(
        child: Column(
          spacing: 16,
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Align(
                alignment: Alignment.centerRight,
                child: IconButton(
                  tooltip: 'Back to chat',
                  onPressed: _close,
                  style: IconButton.styleFrom(
                    backgroundColor: theme.colorScheme.onSurfaceVariant.withValues(alpha: 0.15),
                    padding: const EdgeInsets.all(12),
                  ),
                  icon: const Icon(Icons.close),
                ),
              ),
            ),
            if (call != null && voiceState!.isCameraEnabled) _CameraPreview(call: call),
            Expanded(
              child: Column(
                spacing: 16,
                children: [
                  // The glow shrinks to what is left, as the camera's preview takes room.
                  Expanded(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      spacing: 16,
                      children: [
                        Flexible(
                          child: ConstrainedBox(
                            constraints: const BoxConstraints(maxWidth: 320, maxHeight: 320),
                            child: AspectRatio(
                              aspectRatio: 1,
                              child: AIVoiceGlowView(
                                state: presence.state,
                                colors: _heard?.isPerson ?? false ? _personColors : _agentColors,
                                audioLevel: _heard?.level,
                              ),
                            ),
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
                      ],
                    ),
                  ),
                  CaptionView(caption: _captions.caption, isVisible: _captions.isVisible),
                ],
              ),
            ),
            if (request != null)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                // Keyed by the call, so a second question starts with no answer sending.
                child: _ApprovalCard(
                  key: ValueKey(request.id),
                  request: request,
                  session: widget.chat.session,
                ),
              ),
            if (failure != null)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Text(
                  failure,
                  textAlign: TextAlign.center,
                  style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.error),
                ),
              ),
            if (voice != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 16),
                child: _CallControls(
                  voice: voice,
                  onCamera: () => _toggleCamera(voice),
                  onHangUp: _close,
                ),
              ),
          ],
        ),
      ),
    );
  }
}

/// What the camera shows, which is what Juno sees when it looks.
class _CameraPreview extends StatelessWidget {
  const _CameraPreview({required this.call});

  final video.Call call;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      label: 'What Juno sees',
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        child: ClipRRect(
          borderRadius: BorderRadius.circular(20),
          child: ColoredBox(
            color: Theme.of(context).colorScheme.onSurfaceVariant.withValues(alpha: 0.15),
            child: SizedBox(
              height: 260,
              width: double.infinity,
              child: StreamBuilder(
                stream: call.state.valueStream,
                builder: (context, _) {
                  final me = call.state.value.localParticipant;
                  return me == null
                      ? const SizedBox.shrink()
                      : video.StreamVideoRenderer(
                          call: call,
                          participant: me,
                          videoTrackType: video.SfuTrackType.video,
                        );
                },
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// The camera, mute, and back to the chat.
class _CallControls extends StatelessWidget {
  const _CallControls({required this.voice, required this.onCamera, required this.onHangUp});

  final VoiceSession voice;
  final VoidCallback onCamera;
  final VoidCallback onHangUp;

  static const _size = Size.square(64);

  @override
  Widget build(BuildContext context) {
    final state = voice.state.value;
    final muted = state.isMuted;
    final camera = state.isCameraEnabled;
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      spacing: 24,
      children: [
        IconButton.filledTonal(
          iconSize: 28,
          style: IconButton.styleFrom(fixedSize: _size),
          tooltip: camera ? 'Stop showing Juno' : 'Show Juno the camera',
          onPressed: state.isJoined ? onCamera : null,
          icon: Icon(camera ? Icons.videocam : Icons.videocam_off),
        ),
        IconButton.filledTonal(
          iconSize: 28,
          style: IconButton.styleFrom(fixedSize: _size),
          tooltip: muted ? 'Unmute' : 'Mute',
          onPressed: state.isJoined ? () => voice.setMuted(!muted) : null,
          icon: Icon(muted ? Icons.mic_off : Icons.mic),
        ),
        IconButton.filled(
          iconSize: 28,
          tooltip: 'Hang up',
          style: IconButton.styleFrom(
            fixedSize: _size,
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

/// A tool's question, asked over the call with Stream's approval card. The tool runs only
/// once it is allowed, and the agent is told when it is not.
class _ApprovalCard extends StatefulWidget {
  const _ApprovalCard({super.key, required this.request, required this.session});

  final ToolApprovalRequest request;
  final AgentSession session;

  @override
  State<_ApprovalCard> createState() => _ApprovalCardState();
}

class _ApprovalCardState extends State<_ApprovalCard> {
  AIToolApprovalState _state = const AIToolApprovalIdle();

  /// Sending lasts until the call leaves the session's approvals, which takes the card away.
  void _answer(AIToolApprovalDecision decision) {
    setState(() => _state = AIToolApprovalSending(decision));
    try {
      widget.session.decide(widget.request.id, allowed: decision == AIToolApprovalDecision.allowed);
    } on StateError catch (error) {
      setState(() => _state = AIToolApprovalFailed(decision, error));
    } on SocketClosedException catch (error) {
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
      onAnswer: _answer,
    );
  }
}
