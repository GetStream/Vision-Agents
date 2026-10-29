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
  /// connected with the agents' key, user and token, which every session then shares. A text
  /// or voice session keeps one unless it is incognito; asking an incognito one is a
  /// [StateError].
  Future<stream.Channel> chat() async {
    final cid = session.conversationId;
    final colon = cid.indexOf(':');
    if (colon <= 0 || colon == cid.length - 1) {
      throw StateError(
        'this session keeps no transcript, so there is no channel to read: an incognito '
        'session never has one',
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

/// A reply's thinking, put back together from its live updates.
///
/// The router stores only the opening of each round of reasoning, as its step's `preview`.
/// The whole of it reaches people watching the reply live: each update carries the next
/// window of the round being streamed, in the message's `reasoning` field. [read] every update
/// into one of these, from the client's `message.updated` events rather than the channel's
/// message list, which can fold several updates into one, and show a step by its reply and its
/// id, as `reasoning[(message: message.id, step: step.id)]`: every reply numbers its steps
/// from `r1`.
final class LiveReasoning {
  final _steps = <({String message, String step}), ({String text, int end})>{};

  /// The thinking of reasoning step `step` of reply `message` so far, or null before any of it
  /// arrived.
  String? operator [](({String message, String step}) key) => _steps[key]?.text;

  /// Reads the window a live update carries, if it carries one.
  void read(stream.Message message) {
    if (message.extraData['reasoning'] case {
      'id': final String id,
      'offset': final num offset,
      'text': final String text,
    }) {
      _add(text, offset.toInt(), (message: message.id, step: id));
    }
  }

  /// Adds what [text], which starts [offset] code points into the thinking of the step [key]
  /// names, has that is not held yet. A window that repeats recent thinking, for somebody who
  /// joined midway, overlaps what is held; one that starts past it means updates were missed,
  /// and is kept after a paragraph break rather than run on.
  void _add(String text, int offset, ({String message, String step}) key) {
    final held = _steps[key] ?? (text: '', end: 0);
    final runes = text.runes;
    final end = offset + runes.length;
    if (end <= held.end) {
      return;
    }
    final gap = offset > held.end && held.text.isNotEmpty ? '\n\n' : '';
    final fresh = String.fromCharCodes(runes.skip(held.end > offset ? held.end - offset : 0));
    _steps[key] = (text: '${held.text}$gap$fresh', end: end);
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
