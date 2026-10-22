import 'package:flutter/material.dart';
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';

import 'chat_view.dart';
import 'demo.dart';
import 'voice_view.dart';

void main() => runApp(const FlutterDemoApp());

class FlutterDemoApp extends StatelessWidget {
  const FlutterDemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    // Stream's design system, which Stream's AI components are built on: its colors, text
    // styles and icons, in light and dark.
    final light = StreamTheme.light();
    final dark = StreamTheme.dark();
    return MaterialApp(
      title: 'Larkspur support',
      theme: ThemeData(
        brightness: Brightness.light,
        scaffoldBackgroundColor: light.colorScheme.backgroundApp,
        extensions: [light],
      ),
      darkTheme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: dark.colorScheme.backgroundApp,
        extensions: [dark],
      ),
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

  void _pickStyle() => showStreamSheet<void>(
    context: context,
    isDismissible: true,
    builder: (context, _) => VoiceStyleSheet(
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
      appBar: StreamAppBar(
        title: const Text('Larkspur support'),
        trailing: _tab == 1
            ? StreamButton.icon(
                style: .secondary,
                type: .ghost,
                tooltip: 'Voice style',
                onPressed: _pickStyle,
                icon: Icon(context.streamIcons.slidersFill),
              )
            : null,
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
      bottomNavigationBar: StreamBottomNavBar(
        currentIndex: _tab,
        onTap: (tab) => setState(() => _tab = tab),
        items: [
          StreamBottomNavBarItem(
            icon: Icon(context.streamIcons.messageBubble),
            selectedIcon: Icon(context.streamIcons.messageBubbleFill),
            label: 'Chat',
          ),
          StreamBottomNavBarItem(
            icon: Icon(context.streamIcons.voice),
            selectedIcon: Icon(context.streamIcons.voiceFill),
            label: 'Voice',
          ),
        ],
      ),
    );
  }
}
