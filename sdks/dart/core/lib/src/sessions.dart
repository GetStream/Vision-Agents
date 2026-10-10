import 'dart:math';

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
  /// Held in writing unless [SessionOptions.startVoice] is set. It returns once the backend is
  /// holding the conversation, so a session that comes back is one already listening.
  Future<Session> create([SessionOptions options = const SessionOptions()]) =>
      _create(options, options.startVoice);

  /// A page of this caller's conversations, most recently updated first, the ones that ended
  /// included. Ask again with the page's [ListPage.nextCursor] as [SessionQuery.cursor] for
  /// the next one.
  ///
  /// Only ever the caller's own: the router owns a session by whoever opened it, and a device
  /// cannot widen that by naming somebody else.
  Future<ListPage<Session>> query([SessionQuery query = const SessionQuery()]) =>
      _query(null, query);

  /// Finds a conversation by what it was called, best match first, a page at a time.
  ///
  /// It reads the title, the description, the project and the agent name, never what was
  /// said. An empty [text] is the same as [query], so a search box nobody has typed in yet
  /// shows a person their conversations rather than nothing.
  Future<ListPage<Session>> search(String text, [SessionQuery query = const SessionQuery()]) =>
      _query(text, query);

  /// One conversation, whether or not it is still being held.
  ///
  /// Somebody else's is not found rather than refused, so this is no way to learn whose an id
  /// is.
  Future<Session> get(String id) async => sessionOf(await _operations.getSession(id: id));

  /// Renames or relabels a conversation, running or ended, and returns it as it now is.
  ///
  /// A field left null is left as it is. [custom] replaces the labels whole, and an empty map
  /// clears them.
  Future<Session> update(
    String id, {
    String? title,
    String? description,
    Map<String, Object?>? custom,
  }) async => sessionOf(
    await _operations.updateSession(
      id: id,
      body: api.UpdateSessionRequest(title: title, description: description, custom: custom),
    ),
  );

  /// Stops a conversation, which is how the agent leaves. What it recorded and remembered is
  /// kept; [delete] takes it away.
  Future<void> stop(String id) => _operations.stopSession(id: id);

  /// Puts the agent on the session's own call, `agent:<session id>`, and returns the session
  /// as it now is.
  Future<Session> startVoice(String id) async =>
      sessionOf(await _operations.startSessionVoice(id: id));

  /// Takes the agent off the session's call. The conversation goes on in writing.
  Future<Session> stopVoice(String id) async =>
      sessionOf(await _operations.stopSessionVoice(id: id));

  /// Deletes a conversation, running or ended: it is stopped, and its turns and what it
  /// taught memory are deleted with it.
  Future<void> delete(String id) => _operations.deleteSession(id: id);

  /// Continues a conversation as a new one, leaving the parent as it was.
  ///
  /// Forking an incognito session is refused, since nothing was recorded to fork from.
  Future<Session> fork(String id, [ForkOptions options = const ForkOptions()]) async =>
      sessionOf(await _operations.forkSession(id: id, body: forkRequestOf(options)));

  /// A conversation's turns, whether or not this process is holding it.
  Responses responses(String id) => Responses._(_operations, id);

  Future<Session> _create(SessionOptions options, bool startVoice) async {
    final request = createRequestOf(options, startVoice: startVoice, agent: agent);
    return sessionOf(await _operations.createSession(body: request));
  }

  Future<ListPage<Session>> _query(String? text, SessionQuery query) async {
    final filter = api.SessionFilter(
      agent: _named(query.agent) ?? _named(agent),
      agentId: _named(query.agentId),
      projectId: _named(query.projectId),
      userId: _named(query.userId),
      modality: query.modality?.name,
      state: query.state?.name,
      text: switch (_named(text)) {
        final String text => api.TextMatch(q: text),
        null => null,
      },
    );
    final page = await _operations.querySessions(
      body: api.SessionQuery(
        filter: filter.toJson().isEmpty ? null : filter,
        limit: query.limit,
        cursor: _named(query.cursor),
      ),
    );
    return ListPage(
      [for (final session in page.items) sessionOf(session)],
      hasMore: page.hasMore,
      nextCursor: _named(page.nextCursor),
    );
  }
}

/// The turns of a session being held here, which show each question in [asking] as it is
/// asked.
Responses heldResponses(
  Sessions sessions,
  String id, {
  required void Function(String text) asking,
}) => Responses._(sessions._operations, id, asking: asking);

/// A conversation's turns, and what each was made of.
final class Responses {
  Responses._(this._operations, this.sessionId, {void Function(String text)? asking})
    : _asking = asking;

  final api.Operations _operations;
  final String sessionId;
  final void Function(String text)? _asking;

  /// Asks the agent something and names the turn that answers it.
  ///
  /// It returns as soon as the turn has started, not when it has finished, so what comes back
  /// is a handle: [items] with its id reads what has been written down so far, and the
  /// session's events are what watch it arrive. A text question carries a fresh request id,
  /// so a retry of the same request starts no second turn.
  Future<AgentResponse> create(String text, {List<AgentImage> images = const []}) async {
    // The router refuses a request id on a question with images.
    final requestId = images.isEmpty ? _requestId() : null;
    _asking?.call(text);
    return responseOf(
      await _operations.createResponse(
        id: sessionId,
        body: api.CreateResponseRequest(text: text, images: imagesOf(images), requestId: requestId),
      ),
    );
  }

  /// A page of the turns so far, oldest first. Ask again with the page's
  /// [ListPage.nextCursor] for the next one.
  ///
  /// A session that records nothing has none, and one rewound has none after the response
  /// it went back to.
  Future<ListPage<AgentResponse>> list({int? limit, String? cursor}) async {
    final page = await _operations.listResponses(
      id: sessionId,
      limit: limit,
      cursor: _named(cursor),
    );
    return ListPage(
      [for (final response in page.items) responseOf(response)],
      hasMore: page.hasMore,
      nextCursor: _named(page.nextCursor),
    );
  }

  /// One page of what happened, oldest first, across every turn or within [responseId].
  Future<ListPage<ResponseItem>> items({String? responseId, int? limit, String? cursor}) async {
    final page = await _operations.listResponseItems(
      id: sessionId,
      responseId: responseId,
      limit: limit,
      cursor: _named(cursor),
    );
    return ListPage(
      [for (final item in page.items) itemOf(item)],
      hasMore: page.hasMore,
      nextCursor: _named(page.nextCursor),
    );
  }

  /// Every item, oldest first, a page at a time.
  ///
  /// Paging is inside rather than outside because a conversation's length is not something
  /// the caller chose.
  Stream<ResponseItem> unwind({String? responseId, int pageSize = _itemPage}) async* {
    String? cursor;
    while (true) {
      final page = await items(responseId: responseId, limit: pageSize, cursor: cursor);
      yield* Stream.fromIterable(page.items);
      if (!page.hasMore || page.nextCursor == null) {
        return;
      }
      cursor = page.nextCursor;
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

/// Opens a conversation on its call or in writing, whatever [options] say.
Future<Session> openSession(Sessions sessions, SessionOptions options, {required bool voice}) =>
    sessions._create(options, voice);

/// A name left empty, which means the same as leaving it out.
String? _named(String? name) => name == null || name.isEmpty ? null : name;

/// A fresh request id: 32 hex digits, within what the router accepts.
String _requestId() {
  final random = Random.secure();
  return [
    for (var i = 0; i < 16; i++) random.nextInt(256).toRadixString(16).padLeft(2, '0'),
  ].join();
}
