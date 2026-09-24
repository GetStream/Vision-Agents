import 'package:http/http.dart' as http;

import 'agent_session.dart';
import 'backend.dart';
import 'convert.dart';
import 'errors.dart';
import 'generated/api.dart' as api;
import 'guests.dart';
import 'models.dart';
import 'router.dart';
import 'sessions.dart';
import 'tools.dart';
import 'wire.dart';

/// The router, as a device sees it.
///
/// Two lines get a conversation going:
///
///     final agents = VisionAgents(url: Uri.parse('https://your-router'), customerId: 'acme');
///     final chat = await agents.agent('docs').chat();
///
/// Opening a conversation, following it, reading back its turns, going back to one of them,
/// branching off and ending it, a guest to do it as, and looking something up are the whole
/// of what is here, because they are the whole of what the router lets a device do. What an
/// agent is configured as and a token to join a call with are server-side only: they belong
/// to the app's backend, which hands down what the app needs.
///
/// The constructor does no I/O.
final class VisionAgents {
  VisionAgents({
    required Uri url,
    String customerId = '',
    String apiKey = '',
    String userId = '',
    TokenProvider? token,
    http.Client? httpClient,
  }) : this.withBackend(
         Backend(url: url, customerId: customerId, apiKey: apiKey, userId: userId, token: token),
         httpClient: httpClient,
       );

  /// An injected [httpClient] is the caller's and is never closed here.
  VisionAgents.withBackend(this.backend, {http.Client? httpClient})
    : _client = httpClient ?? http.Client(),
      _ownsClient = httpClient == null {
    _operations = api.Operations(HttpWire(backend, _client));
    sessions = Sessions(_operations);
  }

  final Backend backend;
  final http.Client _client;
  final bool _ownsClient;
  late final api.Operations _operations;

  /// This caller's conversations, whichever agent held them.
  late final Sessions sessions;

  /// One agent, by the name its config was stored under.
  Agent agent(String name) => Agent._(this, name);

  /// The same router, asked by somebody else. Shares this one's HTTP client.
  VisionAgents withUser(String userId, {TokenProvider? token}) =>
      VisionAgents.withBackend(backend.withUser(userId, token: token), httpClient: _client);

  /// The same router, asked by a guest. On a router verifying tokens, the guest's own token is
  /// the credential.
  VisionAgents withGuest(GuestUser guest) =>
      withUser(guest.id, token: backend.apiKey.isEmpty ? null : () async => guest.token);

  /// Holds a conversation in writing: no call is joined, nothing is transcribed or spoken.
  ///
  /// The replies still come from the model with the same instructions, skills and knowledge a
  /// call would have, and arrive as deltas on the socket, which is already open when this
  /// returns.
  Future<AgentSession> chat([SessionOptions options = const SessionOptions()]) =>
      _follow(sessions, options, null);

  /// Puts an agent on a call and follows it.
  ///
  /// The agent joins as soon as this returns. Joining the same call from this device is what
  /// the rtc package is for; this only starts the agent and gives you the state.
  Future<AgentSession> voice(String callId, [SessionOptions options = const SessionOptions()]) =>
      _follow(sessions, options, callId);

  /// Follows a session this caller already has, without creating one: after a relaunch, on
  /// another screen, or a fork.
  ///
  /// A session opened by somebody else is not found, because reading one is reading a
  /// conversation.
  Future<AgentSession> attach(String sessionId, {List<AgentTool> tools = const []}) async {
    final session = AgentSession(
      await sessions.get(sessionId),
      backend: backend,
      sessions: sessions,
      tools: tools,
    );
    await session.start();
    return session;
  }

  /// Looking something up, under a stored router config or none.
  SearchRouter router({String config = '', Map<String, String> tags = const {}}) =>
      SearchRouter(_operations, config: config, tags: tags);

  /// Gets or creates a guest, so somebody can talk to an agent before they sign up.
  ///
  /// Remembered in [store] so coming back is the same guest, with the same conversations,
  /// rather than a second one with an empty history. A remembered guest whose token has
  /// expired is minted a fresh token under the same id. [fresh] mints a new guest whatever is
  /// remembered, for a "not me" button: reusing the remembered one would hand the person at
  /// the keyboard somebody else's conversations.
  ///
  /// Throws a [RouterException] with status 403 when the app does not admit guests.
  Future<GuestUser> guestUser({
    String? name,
    Map<String, Object?>? custom,
    GuestStore? store,
    bool fresh = false,
  }) async {
    final held = store == null || fresh ? null : await readGuest(store);
    final expires = held?.expiresAt;
    if (held != null && (expires == null || expires.isAfter(DateTime.now().add(_leeway)))) {
      return held;
    }
    final minted = guestOf(
      await _operations.createGuestUser(
        body: api.GuestUserRequest(id: held?.id, name: name, custom: custom),
      ),
    );
    if (store != null) {
      await writeGuest(store, minted);
    }
    return minted;
  }

  /// Forgets the remembered guest without minting another, which is what signing out does.
  Future<void> forgetGuest(GuestStore store) => store.clear();

  /// Closes the HTTP client, if this created it. Sessions close on their own.
  void close() {
    if (_ownsClient) {
      _client.close();
    }
  }

  Future<AgentSession> _follow(Sessions sessions, SessionOptions options, String? callId) async {
    final session = AgentSession(
      await sessions.create(options, callId),
      backend: backend,
      sessions: sessions,
      tools: options.tools,
    );
    try {
      await session.start();
    } on AgentsException {
      // The agent is in the conversation whether or not anybody here can follow it, so it is
      // ended rather than left holding a call nobody is listening to.
      try {
        await sessions.close(session.id);
      } on AgentsException {
        // Reporting why the socket did not open matters more than this.
      }
      rethrow;
    }
    return session;
  }
}

/// A token about to expire is as good as expired by the time it reaches the router.
const _leeway = Duration(minutes: 1);

/// One agent: its conversations, and holding a new one.
///
/// The JavaScript SDK's `client.agent("docs")`: every session opened or listed here is this
/// agent's, unless the options name another config.
final class Agent {
  Agent._(this._agents, this.name) : sessions = Sessions(_agents._operations, agent: name);

  final VisionAgents _agents;

  /// The name its config was stored under.
  final String name;

  /// This agent's conversations.
  final Sessions sessions;

  /// Holds a conversation with this agent in writing.
  Future<AgentSession> chat([SessionOptions options = const SessionOptions()]) =>
      _agents._follow(sessions, options, null);

  /// Puts this agent on a call and follows it.
  Future<AgentSession> voice(String callId, [SessionOptions options = const SessionOptions()]) =>
      _agents._follow(sessions, options, callId);
}
