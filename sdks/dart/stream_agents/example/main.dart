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
import 'package:flutter/material.dart';
import 'package:stream_agents/stream_agents.dart';

final agents = StreamAgents(
  apiKey: const String.fromEnvironment('STREAM_API_KEY'),
  userId: const String.fromEnvironment('STREAM_USER_ID'),
  token: () async => const String.fromEnvironment('STREAM_TOKEN'),
);

void main() => runApp(const MaterialApp(home: Example()));

class Example extends StatefulWidget {
  const Example({super.key});

  @override
  State<Example> createState() => _ExampleState();
}

class _ExampleState extends State<Example> {
  AgentSession? _chat;
  VoiceSession? _voice;

  Future<void> _ask() async {
    final chat = _chat ?? await agents.agent('support').chat();
    setState(() => _chat = chat);
    await chat.responses.create('What are your opening hours?');
  }

  Future<void> _talk() async {
    final voice = await VoiceSession.start(agents, agent: 'support');
    setState(() => _voice = voice);
  }

  @override
  void dispose() {
    _chat?.close();
    _voice?.leave();
    agents.disconnect();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final chat = _chat;
    final voice = _voice;
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: chat == null
                  ? const Center(child: Text('Ask the agent something'))
                  // The transcript as it is now, and again each time it changes.
                  : StreamBuilder<Conversation>(
                      stream: chat.conversation.stream,
                      initialData: chat.conversation.value,
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
                TextButton(onPressed: _ask, child: const Text('Ask')),
                TextButton(onPressed: voice == null ? _talk : null, child: const Text('Talk')),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
