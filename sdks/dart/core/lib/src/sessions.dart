import 'dart:convert';

import 'convert.dart';
import 'generated/api.dart' as api;
import 'models.dart';

/// How many items are read per request while unwinding.
const _itemPage = 200;

/// Conversations: the ones being held and the ones that were.
///
/// What comes back are rows rather than live handles, because reading a conversation back is
/// not holding one and most of these are over. `VisionAgents.attach` follows one that is
/// still running.
final class Sessions {
  Sessions(this._operations, {this.agent});

  final api.Operations _operations;

  /// The agent name every call here is about, or null for all of this caller's.
  final String? agent;

  /// Opens a conversation without following it, for a caller building its own state layer.
  ///
  /// Without a [callId] it is held in writing. It returns once the backend is holding the
  /// conversation, so a session that comes back is one already listening.
  Future<Session> create([SessionOptions options = const SessionOptions(), String? callId]) async {
    final request = createRequestOf(options, callId: callId, agent: agent);
    return sessionOf(await _operations.createSession(body: request));
  }

  /// This caller's conversations, newest first, the ones that ended included.
  ///
  /// Only ever the caller's own: the router owns a session by whoever opened it, and a device
  /// cannot widen that by naming somebody else.
  Future<List<Session>> query([SessionQuery query = const SessionQuery()]) async {
    final found = await _operations.listSessions(
      agent: _named(query.agent) ?? _named(agent),
      configId: query.configId,
      userId: query.userId,
      project: query.project,
      state: query.state?.name,
      custom: query.custom == null ? null : jsonEncode(query.custom),
      createdAfter: query.createdAfter,
      createdBefore: query.createdBefore,
      limit: query.limit,
      offset: query.offset,
    );
    return [for (final session in found) sessionOf(session)];
  }

  /// Finds a conversation by what it was called, best match first.
  ///
  /// It reads the title, the description, the project and the agent name, never what was
  /// said. An empty [text] is the same as [query], so a search box nobody has typed in yet
  /// shows a person their conversations rather than nothing.
  Future<List<Session>> search(String text, [SessionQuery query = const SessionQuery()]) async {
    final found = await _operations.searchSessions(
      q: text,
      agent: _named(query.agent) ?? _named(agent),
      configId: query.configId,
      userId: query.userId,
      project: query.project,
      state: query.state?.name,
      custom: query.custom == null ? null : jsonEncode(query.custom),
      createdAfter: query.createdAfter,
      createdBefore: query.createdBefore,
      limit: query.limit,
      offset: query.offset,
    );
    return [for (final session in found) sessionOf(session)];
  }

  /// One conversation, whether or not it is still being held.
  ///
  /// Somebody else's is not found rather than refused, so this is no way to learn whose an id
  /// is.
  Future<Session> get(String id) async => sessionOf(await _operations.getSession(id: id));

  /// Ends a conversation, which is how the agent leaves.
  Future<void> close(String id) => _operations.closeSession(id: id);

  /// Continues a conversation as a new one, leaving the parent as it was.
  ///
  /// Forking an incognito session is refused, since nothing was recorded to fork from.
  Future<Session> fork(String id, [ForkOptions options = const ForkOptions()]) async =>
      sessionOf(await _operations.forkSession(id: id, body: forkRequestOf(options)));

  /// A conversation's turns, whether or not this process is holding it.
  Responses responses(String id) => Responses(_operations, id);
}

/// A conversation's turns, and what each was made of.
final class Responses {
  Responses(this._operations, this.sessionId);

  final api.Operations _operations;
  final String sessionId;

  /// Asks the agent something and names the turn that answers it.
  ///
  /// It returns as soon as the turn has started, not when it has finished, so what comes back
  /// is a handle: [items] with its id reads what has been written down so far, and the
  /// session's events are what watch it arrive.
  Future<AgentResponse> create(String text, {List<AgentImage> images = const []}) async =>
      responseOf(
        await _operations.createResponse(
          id: sessionId,
          body: api.CreateResponseRequest(text: text, images: imagesOf(images)),
        ),
      );

  /// The turns so far, oldest first.
  ///
  /// A session that records nothing has none, and one rewound has none after the response
  /// it went back to.
  Future<List<AgentResponse>> list({int? limit, int? offset}) async => [
    for (final response in await _operations.listResponses(
      id: sessionId,
      limit: limit,
      offset: offset,
    ))
      responseOf(response),
  ];

  /// One page of what happened, oldest first, across every turn or within [responseId].
  Future<List<ResponseItem>> items({String? responseId, int? limit, int? offset}) async => [
    for (final item in await _operations.listResponseItems(
      id: sessionId,
      responseId: responseId,
      limit: limit,
      offset: offset,
    ))
      itemOf(item),
  ];

  /// Every item, oldest first, a page at a time.
  ///
  /// Paging is inside rather than outside because a conversation's length is not something
  /// the caller chose. A short page is the last one.
  Stream<ResponseItem> unwind({String? responseId, int pageSize = _itemPage}) async* {
    var offset = 0;
    while (true) {
      final page = await items(responseId: responseId, limit: pageSize, offset: offset);
      yield* Stream.fromIterable(page);
      if (page.length < pageSize) {
        return;
      }
      offset += page.length;
    }
  }

  /// Goes back to a response and carries on from there, as though nothing after it was said.
  ///
  /// The model forgets the later turns and they drop out of [list] and [items]. A
  /// conversation kept in Stream Chat is refused with a 400, because the channel would still
  /// hold the later turns: fork it at the response instead.
  Future<void> rewind(String responseId) {
    if (responseId.isEmpty) {
      throw ArgumentError.value(responseId, 'responseId', 'a response that was never recorded');
    }
    return _operations.rewindSession(
      id: sessionId,
      body: api.RewindSessionRequest(responseId: responseId),
    );
  }
}

/// A name left empty, which means the same as leaving it out.
String? _named(String? name) => name == null || name.isEmpty ? null : name;
