import 'package:flutter/material.dart';

import 'chat_view.dart';
import 'demo.dart';
import 'voice_view.dart';

void main() => runApp(const FlutterDemoApp());

class FlutterDemoApp extends StatelessWidget {
  const FlutterDemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    // Stream's blue, which the voice glow and the composer take as the accent.
    const seed = Color(0xFF005FFF);
    return MaterialApp(
      title: 'Larkspur support',
      theme: ThemeData(colorSchemeSeed: seed),
      darkTheme: ThemeData(colorSchemeSeed: seed, brightness: Brightness.dark),
      home: const HomePage(),
    );
  }
}

/// Talks to the agent, in writing or out loud.
///
/// There is no agent picker. Reading the configs is server-side only, so a device cannot ask
/// which agents exist; it is told which one it talks to, the way a real app would be, in
/// [Demo.agentName].
class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  var _tab = 0;
  var _voiceStyle = VoiceStyle.orb;

  void _pickStyle() => showModalBottomSheet<void>(
    context: context,
    builder: (context) => VoiceStyleSheet(
      style: _voiceStyle,
      onChanged: (style) {
        setState(() => _voiceStyle = style);
        Navigator.pop(context);
      },
    ),
  );

  @override
  Widget build(BuildContext context) {
    if (Demo.streamApiKey.isEmpty) {
      return const Scaffold(
        body: Center(
          child: Padding(
            padding: EdgeInsets.all(24),
            child: Text(
              'No Stream key yet.\n\nRun with --dart-define=STREAM_API_KEY=<the router\'s STREAM_API_KEY>.',
              textAlign: TextAlign.center,
            ),
          ),
        ),
      );
    }
    return Scaffold(
      appBar: AppBar(
        title: const Text('Larkspur support'),
        actions: [
          if (_tab == 1)
            IconButton(
              tooltip: 'Voice style',
              onPressed: _pickStyle,
              icon: const Icon(Icons.tune),
            ),
        ],
      ),
      // Both tabs stay alive, so a call carries on while the Chat tab is shown.
      body: IndexedStack(
        index: _tab,
        sizing: StackFit.expand,
        children: [
          const ChatView(),
          VoiceView(style: _voiceStyle),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: (tab) => setState(() => _tab = tab),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.chat_bubble_outline),
            label: 'Chat',
          ),
          NavigationDestination(icon: Icon(Icons.graphic_eq), label: 'Voice'),
        ],
      ),
    );
  }
}
