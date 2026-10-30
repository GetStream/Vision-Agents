import 'dart:io' show Platform;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'assistant.dart';
import 'chat_screen.dart';
import 'conversations.dart';
import 'notice.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  runApp(const AssistantApp());
}

class AssistantApp extends StatelessWidget {
  const AssistantApp({super.key});

  @override
  Widget build(BuildContext context) {
    // Stream's blue, which the voice glow and the composer take as the accent.
    const seed = Color(0xFF005FFF);
    return MaterialApp(
      title: 'Juno',
      theme: ThemeData(colorSchemeSeed: seed),
      darkTheme: ThemeData(colorSchemeSeed: seed, brightness: Brightness.dark),
      home: Assistant.streamApiKey.isEmpty
          ? const _NoKey()
          : const _LocalNetwork(child: RootScreen()),
    );
  }
}

/// The conversation in front, and the ones there were in a drawer that slides over it.
class RootScreen extends StatefulWidget {
  const RootScreen({super.key});

  @override
  State<RootScreen> createState() => _RootScreenState();
}

class _RootScreenState extends State<RootScreen> {
  final _conversations = Conversations();
  final _scaffold = GlobalKey<ScaffoldState>();

  /// The conversation open, or null for a new one.
  String? _selected;

  /// Which conversation the screen was opened for.
  ///
  /// It changes only when the person picks one, so a new conversation getting its session
  /// keeps the screen it was started on.
  var _screen = 0;

  @override
  void dispose() {
    _conversations.dispose();
    super.dispose();
  }

  void _show(String? sessionId) {
    setState(() {
      _selected = sessionId;
      _screen += 1;
    });
    _scaffold.currentState?.closeDrawer();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      key: _scaffold,
      appBar: AppBar(
        centerTitle: true,
        titleTextStyle: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w600),
        leading: IconButton(
          tooltip: 'Conversations',
          onPressed: () => _scaffold.currentState?.openDrawer(),
          icon: const Icon(Icons.menu),
        ),
        title: ListenableBuilder(
          listenable: _conversations,
          builder: (context, _) {
            final open = _conversations.items
                .where((session) => session.id == _selected)
                .firstOrNull;
            return Text(open?.displayTitle ?? 'Juno');
          },
        ),
      ),
      drawer: Drawer(
        child: ConversationList(
          conversations: _conversations,
          selected: _selected,
          onNew: () => _show(null),
          onOpen: (session) => _show(session.id),
          onDelete: (session) {
            _conversations.delete(session);
            // The one open is gone, so the screen shows a new conversation instead.
            if (session.id == _selected) {
              setState(() {
                _selected = null;
                _screen += 1;
              });
            }
          },
        ),
      ),
      // The router titles a conversation as it goes, so the list is read again each time it
      // is shown. The drawer slides over the composer without ending its editing, so the
      // keyboard is put away here.
      onDrawerChanged: (isOpen) {
        if (!isOpen) return;
        FocusManager.instance.primaryFocus?.unfocus();
        _conversations.reload();
      },
      body: ChatScreen(
        key: ValueKey(_screen),
        sessionId: _selected,
        onStarted: (sessionId) {
          setState(() => _selected = sessionId);
          _conversations.reload();
        },
      ),
    );
  }
}

/// A gate that asks for the local network before anything reaches the Mac at its address on
/// the Wi-Fi, which Android 17 would otherwise let time out rather than fail.
///
/// Through adb reverse the router and the backend are on the phone's own localhost, which
/// needs no asking, and neither does iOS, which asks by itself.
class _LocalNetwork extends StatefulWidget {
  const _LocalNetwork({required this.child});

  final Widget child;

  @override
  State<_LocalNetwork> createState() => _LocalNetworkState();
}

class _LocalNetworkState extends State<_LocalNetwork> {
  static const _channel = MethodChannel('io.getstream.visionagents.flutterassistant/localNetwork');

  static bool get _onLoopback => [
    Assistant.routerUrl,
    Assistant.backendUrl,
  ].every((url) => const {'localhost', '127.0.0.1'}.contains(url.host));

  late final Future<bool> _allowed = !Platform.isAndroid || _onLoopback
      ? Future.value(true)
      : _channel.invokeMethod<bool>('ask').then((allowed) => allowed ?? false);

  @override
  Widget build(BuildContext context) => FutureBuilder(
    future: _allowed,
    builder: (context, answer) => switch (answer.data) {
      true => widget.child,
      false => const Scaffold(
        body: Center(
          child: Notice(
            icon: Icons.wifi_off,
            title: 'No local network access',
            message:
                'The router runs on your Mac, which the app reaches over the local network. '
                "Allow it in the app's settings.",
          ),
        ),
      ),
      null => const Scaffold(),
    },
  );
}

class _NoKey extends StatelessWidget {
  const _NoKey();

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('Juno')),
    body: const Center(
      child: Notice(
        icon: Icons.key_outlined,
        title: 'No Stream key yet',
        message: "Run with --dart-define=STREAM_API_KEY=<the router's STREAM_API_KEY>.",
      ),
    ),
  );
}
