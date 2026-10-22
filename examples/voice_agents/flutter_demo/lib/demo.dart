import 'dart:convert';
import 'dart:io' show Platform;

import 'package:http/http.dart' as http;
import 'package:vision_agents_core/vision_agents_core.dart';

/// Where the router is, where this app's own backend is, and the tools it answers itself.
///
/// There is no sign-in. The router runs in the mode where it trusts the customer and user ids
/// it is given, `ROUTER_AUTH_MODE=proxy`, so these constants are the whole configuration. In
/// front of a real deployment the router is reached by the key and the same token, and nothing
/// else here would change.
abstract final class Demo {
  /// The iOS simulator reaches the Mac's localhost, and the Android emulator reaches it as
  /// 10.0.2.2. On a device, pass your Mac's address on the network with
  /// `--dart-define=ROUTER_URL=http://192.168.1.20:8080`.
  static final routerUrl = Uri.parse(
    const String.fromEnvironment('ROUTER_URL').isNotEmpty
        ? const String.fromEnvironment('ROUTER_URL')
        : 'http://$_localhost:8080',
  );

  /// swift_demo's backend, `go run ./backend`, which signs the Stream token this device
  /// connects with. A device holds no secret to sign one with.
  static final backendUrl = Uri.parse(
    const String.fromEnvironment('BACKEND_URL').isNotEmpty
        ? const String.fromEnvironment('BACKEND_URL')
        : 'http://$_localhost:8099',
  );

  static String get _localhost => Platform.isAndroid ? '10.0.2.2' : 'localhost';

  /// Whichever customer id you started the router with. `compose.yaml` builds the dashboard
  /// with `examples`, and a config belongs to one customer.
  static const customerId = 'examples';

  /// The `STREAM_API_KEY` the router runs with, passed as
  /// `--dart-define=STREAM_API_KEY=...`: the chat channel and the call are in that Stream app.
  static const streamApiKey = String.fromEnvironment('STREAM_API_KEY');

  /// The agent swift_demo's `go run ./configure` stored, by the name in its `agent.yaml`.
  static const agentName = 'swift_demo';

  /// Who this device is. A call waiting for approval is addressed to them.
  static const userId = 'demo-caller';

  static final agents = VisionAgents(
    url: routerUrl,
    customerId: customerId,
    apiKey: streamApiKey,
    userId: userId,
    token: _streamToken,
  );

  /// Asks the backend for a Stream user token, which the chat and the call connect with.
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

  /// The tools the agent may call, which run on this phone.
  static final tools = [_lookupOrder, _issueRefund];

  /// Orders the agent can look up. Their data never leaves the phone: the agent asks, and
  /// only sees the answer.
  static const _orders = {
    'A-1042':
        '2 Larkspur wool throws, 78.00, paid by card ending 4242, delivered on 14 August, unopened',
    'A-1043':
        '1 linen apron, 24.00, paid by card ending 4242, delivered on 2 September, worn',
  };

  static final _lookupOrder = AgentTool(
    name: 'lookup_order',
    description:
        "Look up one of the caller's orders by its order number, such as A-1042.",
    parameters: AgentTool.strings(
      {'order_id': 'the order number, such as A-1042'},
      required: ['order_id'],
    ),
    displayTitle: 'Looking up your order',
    run: (arguments) async {
      final id = '${arguments['order_id'] ?? ''}'.toUpperCase();
      final order = _orders[id];
      return order == null
          ? 'There is no order $id on this account.'
          : 'Order $id: $order.';
    },
  );

  /// Refunds an order, once the person says it may. The call waits in the session's
  /// `approvals` until they answer: on the reply's step in Chat, or on the card in Voice.
  static final _issueRefund = AgentTool(
    name: 'issue_refund',
    description:
        "Refund one of the caller's orders to the card it was paid with, once the refund skill "
        'has said they are owed it.',
    parameters: AgentTool.strings(
      {
        'order_id': 'the order number, such as A-1042',
        'amount': 'how much to refund, such as 78.00',
        'reason':
            'why the caller is owed it, in a few words they would recognise',
      },
      required: ['order_id', 'amount', 'reason'],
    ),
    displayTitle: 'Refunding your order',
    approval: const ToolApproval(
      title: 'Refund this order?',
      message: 'The money goes back to the card the order was paid with.',
      reasonArgument: 'reason',
      allowTitle: 'Refund',
      declineTitle: 'Not now',
    ),
    run: (arguments) async {
      final id = '${arguments['order_id'] ?? ''}'.toUpperCase();
      if (!_orders.containsKey(id)) {
        return 'There is no order $id on this account.';
      }
      return 'Refunded ${arguments['amount']} for order $id to the card ending 4242. It shows within five '
          'working days.';
    },
  );
}
