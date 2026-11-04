import 'dart:io' show InternetAddress, InternetAddressType, Platform;

import 'package:flutter/material.dart';
import 'package:stream_core_flutter/core.dart';
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
    // Stream's design system, which Stream's AI components are built on: its colors, text
    // styles and icons, in light and dark.
    final light = StreamTheme.light();
    final dark = StreamTheme.dark();
    return MaterialApp(
      title: 'Juno',
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
    return Scaffold(
      key: _scaffold,
      appBar: StreamAppBar(
        leading: StreamButton.icon(
          style: .secondary,
          type: .ghost,
          tooltip: 'Conversations',
          onPressed: () => _scaffold.currentState?.openDrawer(),
          icon: Icon(context.streamIcons.menu),
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
        backgroundColor: context.streamColorScheme.backgroundApp,
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
/// Through adb reverse the router and the backend are on the phone's own localhost, and a
/// hosted router is on the internet, neither of which needs asking, and neither does iOS,
/// which asks by itself.
class _LocalNetwork extends StatefulWidget {
  const _LocalNetwork({required this.child});

  final Widget child;

  @override
  State<_LocalNetwork> createState() => _LocalNetworkState();
}

class _LocalNetworkState extends State<_LocalNetwork> {
  static const _channel = MethodChannel('io.getstream.visionagents.flutterassistant/localNetwork');

  /// Whether the router or the backend is on the local network: a private address, or a
  /// `.local` name. The phone's own localhost and a hosted router need no asking.
  static bool get _onLocalNetwork =>
      [Assistant.routerUrl, Assistant.backendUrl].any((url) => isLocalNetwork(url.host));

  late final Future<bool> _allowed = !Platform.isAndroid || !_onLocalNetwork
      ? Future.value(true)
      : _channel.invokeMethod<bool>('ask').then((allowed) => allowed ?? false);

  @override
  Widget build(BuildContext context) => FutureBuilder(
    future: _allowed,
    builder: (context, answer) => switch (answer) {
      AsyncSnapshot(hasError: true, :final error) => Scaffold(
        body: Center(
          child: Notice(
            icon: context.streamIcons.lowBandwidthFill,
            title: 'Could not ask for the local network',
            message: '$error',
          ),
        ),
      ),
      AsyncSnapshot(data: true) => widget.child,
      AsyncSnapshot(data: false) => Scaffold(
        body: Center(
          child: Notice(
            icon: context.streamIcons.lowBandwidthFill,
            title: 'No local network access',
            message:
                'The router or the backend runs on your Mac, which the app reaches over the '
                "local network. Allow it in the app's settings.",
          ),
        ),
      ),
      _ => const Scaffold(),
    },
  );
}

class _NoKey extends StatelessWidget {
  const _NoKey();

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: StreamAppBar(title: const Text('Juno')),
    body: Center(
      child: Notice(
        icon: context.streamIcons.lock,
        title: 'No Stream key yet',
        message: "Run with --dart-define=STREAM_API_KEY=<the router's STREAM_API_KEY>.",
      ),
    ),
  );
}

/// Whether [host] is on the local network, which Android 17 asks the person about: a private
/// or link-local address, or a `.local` name. Loopback is the phone itself.
bool isLocalNetwork(String host) {
  if (host.endsWith('.local')) return true;
  final address = InternetAddress.tryParse(host);
  if (address == null || address.isLoopback) return false;
  if (address.isLinkLocal) return true;
  final bytes = address.rawAddress;
  if (address.type == InternetAddressType.IPv6) return bytes[0] & 0xfe == 0xfc;
  return bytes[0] == 10 ||
      (bytes[0] == 172 && bytes[1] & 0xf0 == 16) ||
      (bytes[0] == 192 && bytes[1] == 168);
}
