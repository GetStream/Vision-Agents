// A conversation with an agent, in writing and then out loud.
//
// Drop this into an app's lib/main.dart. By default it reaches a local router on port 8080 and
// swift_demo's token server (`go run ./backend`) on port 8099; pass --dart-define=STREAM_API_KEY
// to enable the voice call.
//
// The conversation is rendered as plain text here. For the full app built with Stream Chat's AI
// components (stream_chat_flutter_ai), see examples/voice_agents/flutter_demo.
import 'dart:async';
import 'dart:convert';

// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:stream_agents/stream_agents.dart';

void main() => runApp(const MaterialApp(home: Example()));

class Example extends StatefulWidget {
  const Example({super.key});

  @override
  State<Example> createState() => _ExampleState();
}

class _ExampleState extends State<Example> {
  static const _userId = 'demo-caller';
  static const _apiKey = String.fromEnvironment('STREAM_API_KEY');
  static final _routerUrl = Uri.parse(
    const String.fromEnvironment('ROUTER_URL', defaultValue: 'http://localhost:8080'),
  );
  static final _backendUrl = Uri.parse(
    const String.fromEnvironment('BACKEND_URL', defaultValue: 'http://localhost:8099'),
  );

  final _agents = StreamAgents(
    url: _routerUrl,
    customerId: 'examples',
    apiKey: _apiKey,
    userId: _userId,
    token: _streamToken,
  );
  AgentSession? _chat;
  VoiceSession? _voice;
  bool _busy = false;

  static Future<String> _streamToken() async {
    final response = await http
        .post(
          _backendUrl.resolve('/stream-token'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'user_id': _userId}),
        )
        .timeout(const Duration(seconds: 15));
    if (response.statusCode != 200) {
      throw StateError('the backend did not mint a Stream token (${response.statusCode})');
    }
    return (jsonDecode(response.body) as Map<String, Object?>)['token']! as String;
  }

  Future<void> _ask() => _once(() async {
    final chat = _chat ?? await _agents.agent('swift_demo').chat();
    if (!mounted) {
      return chat.close();
    }
    setState(() => _chat = chat);
    await chat.responses.create('Where is order A-1042?');
  });

  Future<void> _talk() => _once(() async {
    final voice = await VoiceSession.start(_agents, agent: 'swift_demo');
    if (!mounted) {
      return voice.end();
    }
    setState(() => _voice = voice);
    unawaited(
      voice.session.connection.stream.firstWhere((c) => c is Disconnected).then((_) {
        unawaited(voice.leave());
        if (mounted && identical(_voice, voice)) {
          setState(() => _voice = null);
        }
      }),
    );
  });

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
    final chat = _chat;
    final voice = _voice;
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
    final session = voice?.session ?? _chat;
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: session == null
                  ? const Center(child: Text('Ask the agent something'))
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
            if (voice != null) VoiceCallView(voice: voice),
            OverflowBar(
              children: [
                TextButton(onPressed: _busy ? null : _ask, child: const Text('Ask')),
                TextButton(
                  onPressed: _busy || voice != null || _apiKey.isEmpty ? null : _talk,
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
