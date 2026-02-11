import 'dart:async';

import 'package:flutter/material.dart';
import 'package:stream_chat/stream_chat.dart' as chat;
import 'package:stream_video_flutter/stream_video_flutter.dart';
import 'package:uuid/uuid.dart';

import 'agent_service.dart';

/// Main overlay UI: Start/Stop button and scrollable coaching suggestions.
class OverlayScreen extends StatefulWidget {
  const OverlayScreen({
    super.key,
    required this.videoClient,
    required this.chatClient,
  });

  final StreamVideo videoClient;
  final chat.StreamChatClient chatClient;

  @override
  State<OverlayScreen> createState() => _OverlayScreenState();
}

class _OverlayScreenState extends State<OverlayScreen> {
  final _agentService = AgentService();
  final _suggestions = <_Suggestion>[];
  final _scrollController = ScrollController();

  Call? _call;
  chat.Channel? _chatChannel;
  StreamSubscription<chat.Event>? _chatSubscription;
  bool _isActive = false;
  bool _isStarting = false;
  String _status = 'Ready';
  String _meetingContext = '';

  /// The user ID the Vision Agents SDK uses for the agent.
  static const _agentUserId = 'sales-assistant-agent';

  @override
  void dispose() {
    _scrollController.dispose();
    _stop();
    super.dispose();
  }

  // ---------------------------------------------------------------------------
  // Session lifecycle
  // ---------------------------------------------------------------------------

  Future<void> _start() async {
    if (_isStarting || _isActive) return;
    setState(() {
      _isStarting = true;
      _status = 'Creating call…';
    });

    try {
      // 1. Create a unique call
      final callId = 'sales-assistant-${const Uuid().v4().substring(0, 8)}';
      final call = widget.videoClient.makeCall(
        callType: StreamCallType.defaultType(),
        id: callId,
      );

      // 2. Join the call.
      //    - Mic ON: the ScreenAudioMixer mixes system audio into the mic track,
      //      so the agent hears the meeting through a single audio stream.
      //    - Camera OFF: not needed for coaching.
      //    - Screen share ON with captureScreenAudio: starts the macOS system
      //      audio capture whose output is mixed into the mic pipeline.
      setState(() => _status = 'Joining call…');
      await call.getOrCreate();
      await call.join(
        connectOptions: CallConnectOptions(
          camera: TrackOption.disabled(),
          microphone: TrackOption.enabled(),
          screenShare: TrackOption.enabled(
            constraints: const ScreenShareConstraints(
              captureScreenAudio: true,
            ),
          ),
        ),
      );

      // 4. Watch the Stream Chat channel the agent will write to.
      //    Vision Agents SDK uses channel type "messaging" with the call ID.
      //    We must be a *member* (not just a watcher) to receive message.new events.
      setState(() => _status = 'Connecting to agent chat…');
      final currentUserId = widget.chatClient.state.currentUser?.id;
      _chatChannel = widget.chatClient.channel(
        'messaging',
        id: callId,
        extraData: const {'name': 'Sales Assistant Session'},
      );
      await _chatChannel!.watch();
      if (currentUserId != null) {
        try {
          await _chatChannel!.addMembers([currentUserId]);
        } catch (e) {
          debugPrint('[SalesAssistant] addMembers note: $e');
        }
      }
      debugPrint('[SalesAssistant] Chat channel ready: messaging:$callId');
      _listenForAgentMessages();

      // 5. Send meeting context to the agent server.
      await _agentService.setContext(_meetingContext.trim());

      // 6. Tell the agent server to join
      setState(() => _status = 'Starting AI agent…');
      try {
        await _agentService.startSession(callId: callId);
      } catch (e) {
        debugPrint('Agent server not reachable: $e');
        _addSuggestion(
          'Could not reach agent server at ${_agentService.baseUrl}. '
          'Make sure the Python agent is running.',
          isError: true,
        );
      }

      _call = call;
      setState(() {
        _isActive = true;
        _isStarting = false;
        _status = 'Coaching active';
      });

      _addSuggestion(
        'Session started. Listening to your meeting…',
        isSystem: true,
      );
    } catch (e) {
      setState(() {
        _isStarting = false;
        _status = 'Error: $e';
      });
      _addSuggestion('Failed to start: $e', isError: true);
    }
  }

  Future<void> _stop() async {
    final call = _call;
    if (call == null) return;

    setState(() => _status = 'Stopping…');

    _chatSubscription?.cancel();
    _chatSubscription = null;

    try {
      await call.leave();
    } catch (_) {}

    await _agentService.stopSession();

    _call = null;
    _chatChannel = null;
    setState(() {
      _isActive = false;
      _status = 'Ready';
    });

    _addSuggestion('Session ended.', isSystem: true);
  }

  // ---------------------------------------------------------------------------
  // Agent message listener (via Stream Chat)
  // ---------------------------------------------------------------------------

  void _listenForAgentMessages() {
    final channel = _chatChannel;
    if (channel == null) return;

    _chatSubscription = channel.on().listen((chat.Event event) {
      final message = event.message;
      if (message == null) return;

      final senderId = message.user?.id;
      if (senderId != _agentUserId) return;

      final text = message.text;
      if (text == null || text.isEmpty) return;

      if (event.type == 'message.new') {
        _addSuggestion(text, messageId: message.id);
      } else if (event.type == 'message.updated') {
        _updateSuggestion(message.id, text);
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Suggestion management
  // ---------------------------------------------------------------------------

  void _addSuggestion(
    String text, {
    String? messageId,
    bool isSystem = false,
    bool isError = false,
  }) {
    setState(() {
      _suggestions.add(
        _Suggestion(
          text: text,
          time: DateTime.now(),
          messageId: messageId,
          isSystem: isSystem,
          isError: isError,
        ),
      );
    });

    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _updateSuggestion(String messageId, String text) {
    final idx = _suggestions.lastIndexWhere((s) => s.messageId == messageId);
    if (idx == -1) return;
    setState(() {
      _suggestions[idx].text = text;
    });
  }

  // ---------------------------------------------------------------------------
  // Context dialog
  // ---------------------------------------------------------------------------

  void _showContextDialog() {
    final controller = TextEditingController(text: _meetingContext);
    showDialog<void>(
      context: context,
      barrierColor: Colors.black26,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xDD1a1a2e),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        title: Text(
          'Meeting Context',
          style: TextStyle(
            color: Colors.white.withValues(alpha: 0.8),
            fontSize: 15,
            fontWeight: FontWeight.w600,
          ),
        ),
        content: TextField(
          controller: controller,
          minLines: 3,
          maxLines: 5,
          autofocus: true,
          style: TextStyle(
            color: Colors.white.withValues(alpha: 0.85),
            fontSize: 13,
          ),
          cursorColor: Colors.white.withValues(alpha: 0.5),
          decoration: InputDecoration(
            hintText:
                'e.g. "Enterprise SaaS sales call with a CTO"\n'
                'or "Technical interview for a senior engineer"',
            hintStyle: TextStyle(
              color: Colors.white.withValues(alpha: 0.2),
              fontSize: 13,
            ),
            filled: true,
            fillColor: Colors.white.withValues(alpha: 0.05),
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
              borderSide: BorderSide(
                color: Colors.white.withValues(alpha: 0.08),
              ),
            ),
            enabledBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
              borderSide: BorderSide(
                color: Colors.white.withValues(alpha: 0.08),
              ),
            ),
            focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
              borderSide: BorderSide(
                color: Colors.white.withValues(alpha: 0.15),
              ),
            ),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: Text(
              'Cancel',
              style: TextStyle(color: Colors.white.withValues(alpha: 0.35)),
            ),
          ),
          TextButton(
            onPressed: () {
              setState(() => _meetingContext = controller.text);
              Navigator.pop(ctx);
            },
            child: Text(
              'Save',
              style: TextStyle(color: Colors.white.withValues(alpha: 0.7)),
            ),
          ),
        ],
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Build
  // ---------------------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        _buildHeader(),
        Expanded(child: _buildSuggestionList()),
        _buildFooter(),
      ],
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 32, 20, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 10,
                height: 10,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: _isActive
                      ? Colors.greenAccent.withValues(alpha: 0.7)
                      : Colors.white.withValues(alpha: 0.15),
                ),
              ),
              const SizedBox(width: 10),
              Text(
                'Sales Assistant',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  color: Colors.white.withValues(alpha: 0.8),
                  letterSpacing: -0.5,
                ),
              ),
              if (_meetingContext.trim().isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(left: 6),
                  child: Icon(
                    Icons.circle,
                    size: 6,
                    color: Colors.blueAccent.withValues(alpha: 0.6),
                  ),
                ),
              const Spacer(),
              IconButton(
                onPressed: _isActive ? null : _showContextDialog,
                icon: Icon(
                  Icons.tune_rounded,
                  size: 18,
                  color: _isActive
                      ? Colors.white.withValues(alpha: 0.15)
                      : Colors.white.withValues(alpha: 0.4),
                ),
                tooltip: 'Meeting context',
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
              ),
              const SizedBox(width: 4),
              _buildActionButton(),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            _status,
            style: TextStyle(
              fontSize: 11,
              color: Colors.white.withValues(alpha: 0.35),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildActionButton() {
    if (_isStarting) {
      return const SizedBox(
        width: 20,
        height: 20,
        child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white54),
      );
    }

    return TextButton(
      onPressed: _isActive ? _stop : _start,
      style: TextButton.styleFrom(
        backgroundColor: _isActive
            ? Colors.redAccent.withValues(alpha: 0.12)
            : Colors.white.withValues(alpha: 0.08),
        foregroundColor: _isActive
            ? Colors.redAccent.withValues(alpha: 0.8)
            : Colors.white.withValues(alpha: 0.6),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      ),
      child: Text(_isActive ? 'Stop' : 'Start'),
    );
  }

  Widget _buildSuggestionList() {
    if (_suggestions.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.assistant,
                size: 48,
                color: Colors.white.withValues(alpha: 0.15),
              ),
              const SizedBox(height: 16),
              Text(
                'Press Start to begin coaching.\n'
                'Your screen and its audio\n'
                'will be shared with the AI agent.\n\n'
                'Tap the tune icon to set meeting context.',
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.35),
                  fontSize: 13,
                  height: 1.5,
                ),
              ),
            ],
          ),
        ),
      );
    }

    return ListView.builder(
      controller: _scrollController,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      itemCount: _suggestions.length,
      itemBuilder: (context, index) {
        final s = _suggestions[index];
        return _SuggestionCard(suggestion: s);
      },
    );
  }

  Widget _buildFooter() {
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 16),
      child: Text(
        'Powered by Stream Vision Agents',
        style: TextStyle(
          fontSize: 11,
          color: Colors.white.withValues(alpha: 0.25),
        ),
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// Data model
// -----------------------------------------------------------------------------

class _Suggestion {
  _Suggestion({
    required this.text,
    required this.time,
    this.messageId,
    this.isSystem = false,
    this.isError = false,
  });

  String text;
  final DateTime time;
  final String? messageId;
  final bool isSystem;
  final bool isError;
}

// -----------------------------------------------------------------------------
// Suggestion card widget
// -----------------------------------------------------------------------------

class _SuggestionCard extends StatelessWidget {
  const _SuggestionCard({required this.suggestion});

  final _Suggestion suggestion;

  @override
  Widget build(BuildContext context) {
    final Color bgColor;
    final Color textColor;
    final Color timeColor;

    if (suggestion.isError) {
      bgColor = Colors.redAccent.withValues(alpha: 0.12);
      textColor = Colors.redAccent.shade100.withValues(alpha: 0.95);
      timeColor = Colors.redAccent.withValues(alpha: 0.35);
    } else if (suggestion.isSystem) {
      bgColor = Colors.white.withValues(alpha: 0.05);
      textColor = Colors.white.withValues(alpha: 0.45);
      timeColor = Colors.white.withValues(alpha: 0.2);
    } else {
      bgColor = Colors.white.withValues(alpha: 0.1);
      textColor = Colors.white.withValues(alpha: 0.92);
      timeColor = Colors.white.withValues(alpha: 0.3);
    }

    final time = suggestion.time;
    final timeStr =
        '${time.hour.toString().padLeft(2, '0')}:'
        '${time.minute.toString().padLeft(2, '0')}:'
        '${time.second.toString().padLeft(2, '0')}';

    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            suggestion.text,
            style: TextStyle(
              color: textColor,
              fontSize: 13,
              height: 1.45,
            ),
          ),
          const SizedBox(height: 4),
          Align(
            alignment: Alignment.centerRight,
            child: Text(
              timeStr,
              style: TextStyle(color: timeColor, fontSize: 10),
            ),
          ),
        ],
      ),
    );
  }
}
