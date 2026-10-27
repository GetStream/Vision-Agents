import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:vision_agents_core/vision_agents_core.dart';

import 'device_tools.dart';

/// Where the router is, where the token backend is, and who this phone is.
///
/// There is no sign-in. The router runs in the mode where it trusts the customer and user ids
/// it is given, `ROUTER_AUTH_MODE=proxy`, so these constants are the whole configuration. In
/// front of a real deployment the router is reached by the key and the same token, and nothing
/// else here would change.
abstract final class Assistant {
  /// The phone's own localhost, which the iOS simulator shares with the Mac and which
  /// `adb reverse tcp:8080 tcp:8080` hands on to it from an Android phone or emulator. To
  /// reach the Mac over the network instead, pass its address:
  /// `--dart-define=ROUTER_URL=http://192.168.1.20:8080`.
  static final routerUrl = Uri.parse(
    const String.fromEnvironment(
      'ROUTER_URL',
      defaultValue: 'http://localhost:8080',
    ),
  );

  /// swift_assistant's backend, `go run ./backend`. A phone holds no secret to sign a Stream
  /// token with and may not reach its person's Gmail connection on the router, so it asks
  /// this.
  static final backendUrl = Uri.parse(
    const String.fromEnvironment(
      'BACKEND_URL',
      defaultValue: 'http://localhost:8098',
    ),
  );

  /// The customer swift_assistant's `go run ./configure` stored the agent under.
  static const customerId = 'examples';

  /// The `STREAM_API_KEY` the router runs with, passed as
  /// `--dart-define=STREAM_API_KEY=...`: the conversations' channels and calls are in that
  /// Stream app.
  static const streamApiKey = String.fromEnvironment('STREAM_API_KEY');

  /// The agent swift_assistant's `go run ./configure` stored, by the name in its
  /// `agent.yaml`. The iOS and Android apps talk to it too.
  static const agentName = 'swift_assistant';

  /// The model a written conversation runs on. Grok reads photos and streams its reasoning,
  /// which is what fills a reply's thinking panel. A call keeps the agent's faster model.
  static const chatModel = 'xai/grok-4.7';

  /// Who this phone is: the same person as the iOS and Android apps, so all three list the
  /// same conversations.
  static const userId = 'juno-owner';
  static const userName = 'Alex';

  static final agents = VisionAgents(
    url: routerUrl,
    customerId: customerId,
    apiKey: streamApiKey,
    userId: userId,
    token: _streamToken,
  );

  /// The agent's conversations: opening one, and listing and searching the ones there were.
  static Sessions get sessions => agents.agent(agentName).sessions;

  /// The tools the agent may call, which run on this phone.
  static final tools = DeviceTools.all;

  /// Asks the backend for a Stream user token, which opens the chat and joins the call.
  static Future<String> _streamToken() async {
    final response = await http.post(
      backendUrl.resolve('/stream-token'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'user_id': userId}),
    );
    if (response.statusCode != 200) {
      throw StateError(
        'the backend would not mint a Stream token (${response.statusCode})',
      );
    }
    return (jsonDecode(response.body) as Map<String, Object?>)['token']!
        as String;
  }
}
