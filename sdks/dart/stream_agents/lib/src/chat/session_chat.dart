import 'dart:math';

import 'package:stream_chat/stream_chat.dart' as stream;
import 'package:stream_agents/src/core/core.dart';

const _kind = 'chat';

/// The app's own Stream Chat client.
extension ChatClients on StreamAgents {
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
    final heldRunes = max(0, held.end - offset);
    var runes = 0;
    var fresh = text.length;
    for (final iterator = text.runes.iterator; iterator.moveNext(); runes++) {
      if (runes == heldRunes) {
        fresh = iterator.rawIndex;
      }
    }
    final end = offset + runes;
    if (end <= held.end) {
      return;
    }
    final gap = offset > held.end && held.text.isNotEmpty ? '\n\n' : '';
    _steps[key] = (text: '${held.text}$gap${text.substring(fresh)}', end: end);
  }
}

/// Overlays 0ms WebSocket `ui_delta` and `ui_completed` updates from [Conversation] onto
/// Stream Chat assistant messages while they stream, handing off cleanly once Stream Chat's
/// durable `extraData['ai_ui']` settles (`status == 'complete'`).
final class LiveUI {
  final _byTurn = <String, TurnUI>{};
  String? _latestTurnId;
  final _cache = Expando<({TurnUI ui, stream.Message overlaid})>();

  /// The live UI recorded for [turnId], or null.
  TurnUI? operator [](String turnId) => _byTurn[turnId];

  /// The latest agent turn's live UI, for voice screens or an in-flight reply before its
  /// `turn_id` lands in Stream Chat.
  TurnUI? get latest => _latestTurnId == null ? null : _byTurn[_latestTurnId!];

  /// Folds [conversation]'s turns in and reports whether any turn's UI changed.
  bool observe(Conversation conversation) {
    var changed = false;
    for (final turn in conversation.turns) {
      if (!turn.speaker.isAgent || turn.ui == null) continue;
      final next = turn.ui!;
      if (_byTurn[turn.id] != next) {
        _byTurn[turn.id] = next;
        _latestTurnId = turn.id;
        changed = true;
      }
    }
    return changed;
  }

  /// Returns [messages] with live socket UI overlaid onto any assistant reply whose durable
  /// Stream Chat `ai_ui` has not yet reached `status == 'complete'`.
  List<stream.Message> overlay(List<stream.Message> messages) {
    if (_byTurn.isEmpty || messages.isEmpty) return messages;
    final latestGenerating = _findLatestGenerating(messages);
    List<stream.Message>? mutated;

    for (var i = 0; i < messages.length; i++) {
      final message = messages[i];
      if (!_isAssistant(message)) continue;
      if (_existingStatus(message) == 'complete') continue;
      final turnId = _turnIdOf(message);
      final ui =
          (turnId != null ? _byTurn[turnId] : null) ?? (i == latestGenerating ? latest : null);
      if (ui == null) continue;
      final next = _withUI(message, ui);
      if (!identical(next, message)) {
        mutated ??= [...messages];
        mutated[i] = next;
      }
    }
    return mutated ?? messages;
  }

  stream.Message _withUI(stream.Message message, TurnUI ui) {
    final cached = _cache[message];
    if (cached != null && cached.ui == ui) return cached.overlaid;
    final overlaid = message.copyWith(extraData: {...message.extraData, 'ai_ui': ui.toWireJson()});
    _cache[message] = (ui: ui, overlaid: overlaid);
    return overlaid;
  }

  static int _findLatestGenerating(List<stream.Message> messages) {
    for (var i = messages.length - 1; i >= 0; i--) {
      final m = messages[i];
      if (_isAssistant(m) && m.extraData['generating'] == true) return i;
    }
    return -1;
  }

  static bool _isAssistant(stream.Message message) {
    if (message.extraData['ai_generated'] == true) return true;
    if (message.extraData['support_runtime'] case {'role': 'assistant'}) return true;
    return false;
  }

  static String? _existingStatus(stream.Message message) {
    if (message.extraData['ai_ui'] case final Map<Object?, Object?> ui) {
      if (ui['overflow'] == true) return null;
      if (ui['status'] case final String status) return status;
    }
    return null;
  }

  static String? _turnIdOf(stream.Message message) {
    if (message.extraData['turn_id'] case final String turnId when turnId.isNotEmpty) {
      return turnId;
    }
    if (message.extraData['support_runtime'] case {
      'turn_id': final String turnId,
    } when turnId.isNotEmpty) {
      return turnId;
    }
    return null;
  }
}

/// A client connected as the agents' user, asking them for a fresh token when Stream says
/// the last one expired.
Future<stream.StreamChatClient> _connect(Backend backend, StreamCredentials credentials) async {
  final client = stream.StreamChatClient(credentials.apiKey);
  String? unused = credentials.token;
  try {
    await client.connectUserWithProvider(stream.User(id: credentials.userId), (_) async {
      final token = unused ?? (await backend.streamCredentials(refresh: true)).token;
      unused = null;
      return token;
    });
  } catch (_) {
    await client.dispose();
    rethrow;
  }
  return client;
}
