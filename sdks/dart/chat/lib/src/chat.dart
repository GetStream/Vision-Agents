import 'package:stream_chat/stream_chat.dart' as stream;
import 'package:vision_agents_core/vision_agents_core.dart';

const _kind = 'chat';

/// The app's own Stream Chat client.
extension ChatClients on VisionAgents {
  /// Hands over the app's own Stream Chat client, so every conversation opens on it rather
  /// than on a second one. Connecting it, as the agents' `userId`, stays the app's job, and
  /// [disconnect] leaves it alone.
  void useChat(stream.StreamChatClient client) {
    final connected = client.state.currentUser?.id;
    if (connected != null && connected != backend.userId) {
      throw ArgumentError(
        'the chat client is connected as $connected but these agents are for '
        '${backend.userId}: connect it as the same user, or the conversation is somebody '
        'else\'s',
      );
    }
    backend.give(_kind, client);
  }
}

/// The Stream Chat channel a conversation is kept in.
extension SessionChat on AgentSession {
  /// The channel this conversation is kept in, not yet watched.
  ///
  /// It opens on the client handed over with `useChat`, or else on one built here and
  /// connected with the agents' key, user and token, which every session then shares. Only a
  /// text session that is not incognito keeps one; asking any other is a [StateError].
  Future<stream.Channel> chat() async {
    final cid = session.conversationId;
    final colon = cid.indexOf(':');
    if (colon <= 0 || colon == cid.length - 1) {
      throw StateError(
        'this session keeps no transcript, so there is no channel to read: only a text '
        'session has one, and an incognito session never does',
      );
    }
    final client = await backend.shared<stream.StreamChatClient>(
      _kind,
      open: (credentials) => _connect(backend, credentials),
      disconnect: (client) => client.dispose(),
    );
    return client.channel(cid.substring(0, colon), id: cid.substring(colon + 1));
  }
}

/// A client connected as the agents' user, asking them for a fresh token when Stream says
/// the last one expired.
Future<stream.StreamChatClient> _connect(Backend backend, StreamCredentials credentials) async {
  final client = stream.StreamChatClient(credentials.apiKey);
  String? unused = credentials.token;
  try {
    await client.connectUserWithProvider(stream.User(id: credentials.userId), (_) async {
      final token = unused ?? (await backend.streamCredentials()).token;
      unused = null;
      return token;
    });
  } catch (_) {
    await client.dispose();
    rethrow;
  }
  return client;
}
