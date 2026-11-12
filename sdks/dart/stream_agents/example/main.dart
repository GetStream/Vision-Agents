// A conversation with an agent, in writing and then out loud.
//
// It is an app's lib/main.dart. Run that app with the Stream API key, a user and that user's
// token, which your backend signs with the Stream secret:
//
//     flutter run --dart-define=STREAM_API_KEY=... \
//       --dart-define=STREAM_USER_ID=... --dart-define=STREAM_TOKEN=...
//
// The conversation is rendered as plain text here. A real app shows it with Stream Chat's AI
// components, stream_chat_flutter_ai; examples/voice_agents/flutter_demo in the Vision-Agents
// repository is that app.
import 'dart:async';

// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:stream_agents/stream_agents.dart';

void main() => runApp(const MaterialApp(home: Example()));

class Example extends StatefulWidget {
  const Example({super.key});

  @override
  State<Example> createState() => _ExampleState();
}

class _ExampleState extends State<Example> {
  // The screen's own, so leaving it closes everything it opened.
  final _agents = StreamAgents(
    apiKey: const String.fromEnvironment('STREAM_API_KEY'),
    userId: const String.fromEnvironment('STREAM_USER_ID'),
    token: () async => const String.fromEnvironment('STREAM_TOKEN'),
  );
  AgentSession? _chat;
  VoiceSession? _voice;
  StreamSubscription<Connection>? _voiceEnded;
  bool _busy = false;

  Future<void> _ask() => _once(() async {
    final chat = _chat ?? await _agents.agent('support').chat();
    if (!mounted) {
      return chat.close();
    }
    setState(() => _chat = chat);
    await chat.responses.create('What are your opening hours?');
  });

  Future<void> _talk() => _once(() async {
    final voice = await VoiceSession.start(_agents, agent: 'support');
    if (!mounted) {
      return voice.end();
    }
    // Hanging up, or the agent ending the conversation, stops the session: back to text.
    _voiceEnded = voice.session.connection.stream.listen((connection) {
      if (connection is Disconnected) {
        _voiceEnded?.cancel();
        unawaited(voice.leave());
        setState(() => _voice = null);
      }
    });
    setState(() => _voice = voice);
  });

  /// Runs [action] unless one is running, so a second tap opens no second session.
  Future<void> _once(Future<void> Function() action) async {
    if (_busy) {
      return;
    }
    setState(() => _busy = true);
    try {
      await action();
    } finally {
      if (mounted) {
        setState(() => _busy = false);
      }
    }
  }

  @override
  void dispose() {
    _voiceEnded?.cancel();
    final chat = _chat;
    final voice = _voice;
    // The sessions first, then the clients they ran on.
    unawaited(() async {
      await chat?.close();
      await voice?.leave();
      await _agents.disconnect();
      _agents.close();
    }());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final voice = _voice;
    // While talking, what is said; otherwise what was written.
    final session = voice?.session ?? _chat;
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: session == null
                  ? const Center(child: Text('Ask the agent something'))
                  // The transcript as it is now, and again each time it changes.
                  : StreamBuilder<Conversation>(
                      stream: session.conversation.stream,
                      initialData: session.conversation.value,
                      builder: (context, snapshot) => ListView(
                        children: [
                          for (final turn in snapshot.requireData.turns)
                            ListTile(title: Text(turn.text)),
                        ],
                      ),
                    ),
            ),
            // Joins the call when it is first built: mute, camera and hang up.
            if (voice != null) VoiceCallView(voice: voice),
            OverflowBar(
              children: [
                TextButton(onPressed: _busy ? null : _ask, child: const Text('Ask')),
                TextButton(
                  onPressed: _busy || voice != null ? null : _talk,
                  child: const Text('Talk'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
