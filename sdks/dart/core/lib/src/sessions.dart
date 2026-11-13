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
  Sessions(this._operations, {this.agent, String userId = ''}) : _userId = userId;

  final api.Operations _operations;

  /// Who is asking, which is whose memories a session reads.
  final String _userId;

  /// The agent name every call here is about, or null for all of this caller's.
  final String? agent;

  /// Opens a conversation without following it, for a caller building its own state layer.
  ///
  /// Held in writing unless [SessionOptions.startVoice] is set. It returns once the backend is
  /// holding the conversation, so a session that comes back is one already listening.
  Future<Session> create([SessionOptions options = const SessionOptions()]) =>
      _create(options, options.startVoice);

  /// A page of this caller's conversations, the ones that ended included.
  ///
  /// [filter] is the router's own filter, written as it goes on the wire: field keys and
  /// `$and` / `$or`, the keys of one map all holding. A device may filter on:
  ///
  /// - `project_id`: a string, `{'$eq': 'a'}` or `{'$in': ['a', 'b']}`
  /// - `agent`: a string or `{'$eq': 'support'}`; on an agent's own [Sessions] the top-level
  ///   one is always that agent, whatever was written there
  /// - `text`: `{'$q': 'words'}`, once, at the top level or directly in the top-level `$and`,
  ///   never inside an `$or`
  ///
  /// `$and` and `$or` take 1 to 10 filters and nest at most 3 deep. `review_score` and
  /// `user_id` are a backend's to filter on, and any other key is one the router refuses, so
  /// each is an [ArgumentError] here before anything is sent.
  ///
  /// [sort] left out is most recently answered first, or best match first with `text`.
  /// Otherwise it is `last_message_at` either way, `pinned` alone or followed by
  /// `last_message_at`, or `relevance` with `text`; see [SessionSort].
  ///
  /// [limit] is up to 200 and 25 when left out. [cursor] is the [ListPage.nextCursor] of the
  /// page before, asked with the same filter and sort, since a cursor only continues the order
  /// it came from.
  ///
  /// Only ever the caller's own: the router owns a session by whoever opened it, and a device
  /// cannot widen that by naming somebody else.
  Future<ListPage<Session>> query({
    Map<String, Object?>? filter,
    List<SessionSort>? sort,
    int? limit,
    String? cursor,
  }) => _query(filter, sort: sort, limit: limit, cursor: cursor);

  /// Finds a conversation by what it was called, best match first, a page at a time.
  ///
  /// It reads the title, the description, the project and the agent name, never what was
  /// said. [filter], [limit] and [cursor] are [query]'s; [text] becomes the filter's top-level
  /// `text`, so a [filter] that already has one is an [ArgumentError]. There is no sort,
  /// because a search is ordered by how well it matched. An empty [text] is the same as
  /// [query], so a search box nobody has typed in yet shows a person their conversations
  /// rather than nothing.
  Future<ListPage<Session>> search(
    String text, {
    Map<String, Object?>? filter,
    int? limit,
    String? cursor,
  }) {
    if (filter != null && _mentions(filter, 'text')) {
      throw ArgumentError.value(filter, 'filter', 'a search sets text itself');
    }
    return _query(
      text.isEmpty
          ? filter
          : {
              ...?filter,
              'text': {r'$q': text},
            },
      limit: limit,
      cursor: cursor,
    );
  }

  /// One conversation, whether or not it is still being held.
  ///
  /// Somebody else's is not found rather than refused, so this is no way to learn whose an id
  /// is.
  Future<Session> get(String id) async => sessionOf(await _operations.getSession(id: id));

  /// Renames, relabels, pins or unpins a conversation, running or ended, and returns it as it
  /// now is.
  ///
  /// A field left null is left as it is. [custom] replaces the labels whole, and an empty map
  /// clears them. [pinned] true pins it and false unpins it; pinning does not move it in the
  /// most recently answered order.
  Future<Session> update(
    String id, {
    String? title,
    String? description,
    Map<String, Object?>? custom,
    bool? pinned,
  }) async => sessionOf(
    await _operations.updateSession(
      id: id,
      body: api.UpdateSessionRequest(
        title: title,
        description: description,
        custom: custom,
        pinned: pinned,
      ),
    ),
  );

  /// Pins a conversation, and returns it with [Session.pinnedAt] set. Sorting by `pinned`
  /// lists the pinned ones first.
  Future<Session> pin(String id) => update(id, pinned: true);

  /// Unpins a conversation, and returns it with no [Session.pinnedAt].
  Future<Session> unpin(String id) => update(id, pinned: false);

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
    final request = createRequestOf(options, startVoice: startVoice, agent: agent, userId: _userId);
    return sessionOf(await _operations.createSession(body: request));
  }

  Future<ListPage<Session>> _query(
    Map<String, Object?>? filter, {
    List<SessionSort>? sort,
    int? limit,
    String? cursor,
  }) async {
    // The agent goes on the top level beside the caller's keys rather than around them in a
    // new $and, since text is only accepted at the top or directly in the top-level $and.
    final sent = {...?filter, 'agent': ?_named(agent)};
    final page = await _operations.querySessions(
      body: api.SessionQuery(
        filter: sent.isEmpty ? null : _filterOf(sent, 'filter'),
        sort: sort == null || sort.isEmpty ? null : [for (final key in sort) _sortOf(key)],
        limit: limit,
        cursor: _named(cursor),
      ),
    );
    return ListPage(
      [for (final session in page.items) sessionOf(session)],
      hasMore: page.hasMore,
      nextCursor: _named(page.nextCursor),
    );
  }
}

/// What a backend may filter on and a device may not.
const _serverSideOnly = {'review_score', 'user_id'};

/// The caller's filter as the request carries it, refused if it holds a key a device may not
/// send or the router does not know. [at] is where in the filter this map is, for the error.
api.SessionFilter _filterOf(Map<Object?, Object?> filter, String at) {
  for (final key in filter.keys) {
    if (_serverSideOnly.contains(key)) {
      throw ArgumentError.value(key, at, 'a filter only a backend may send');
    }
    if (!const {r'$and', r'$or', 'agent', 'project_id', 'text'}.contains(key)) {
      throw ArgumentError.value(key, at, 'not a field sessions are filtered on');
    }
  }
  return api.SessionFilter(
    and: _filtersOf(filter[r'$and'], '$at.\$and'),
    or: _filtersOf(filter[r'$or'], '$at.\$or'),
    agent: filter['agent'],
    projectId: filter['project_id'],
    text: switch (filter['text']) {
      null => null,
      {r'$q': final String words} => api.TextMatch(q: words),
      final Object other => throw ArgumentError.value(other, '$at.text', r"expected {'$q': words}"),
    },
  );
}

List<api.SessionFilter>? _filtersOf(Object? filters, String at) => switch (filters) {
  null => null,
  final List<Object?> filters => [
    for (final (index, filter) in filters.indexed)
      switch (filter) {
        final Map<Object?, Object?> filter => _filterOf(filter, '$at[$index]'),
        _ => throw ArgumentError.value(filter, '$at[$index]', 'expected a filter'),
      },
  ],
  final Object other => throw ArgumentError.value(other, at, 'expected a list of filters'),
};

/// Whether a filter names [field] at any depth.
bool _mentions(Map<Object?, Object?> filter, String field) =>
    filter.containsKey(field) ||
    [filter[r'$and'], filter[r'$or']].whereType<List<Object?>>().any(
      (filters) =>
          filters.whereType<Map<Object?, Object?>>().any((inner) => _mentions(inner, field)),
    );

api.SessionSort _sortOf(SessionSort sort) {
  if (sort.field == 'review_score') {
    throw ArgumentError.value(sort.field, 'sort', 'an order only a backend may ask for');
  }
  return api.SessionSort(field: sort.field, direction: sort.direction);
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
  /// session's events are what watch it arrive. Every question carries a fresh request id,
  /// so a retry of the same request starts no second turn. A conversation kept in Stream Chat
  /// takes [images] as data URIs only, and uploads them to its channel.
  ///
  /// [maxOutputTokens] caps this answer, reasoning included, and [reasoningEffort] is how long
  /// the model may think first, one of the efforts the session's model accepts. Both cover the
  /// replies that finish it after its tools; left null, the session's own apply.
  ///
  /// [uiAction] is what the person did in the UI an earlier reply showed, which [text] says in
  /// their words, such as Stream Chat AI's `AIUIAction.toJson()`: the reply (`message_id`), the
  /// UI's format and the action (`name`), with what it carries (`payload`), at most 8 KB as
  /// JSON. Their message carries it, and the model is told it after the text. An empty one is
  /// not sent.
  Future<AgentResponse> create(
    String text, {
    List<AgentImage> images = const [],
    int? maxOutputTokens,
    String? reasoningEffort,
    Map<String, Object?>? uiAction,
  }) async {
    _asking?.call(text);
    return responseOf(
      await _operations.createResponse(
        id: sessionId,
        body: api.CreateResponseRequest(
          text: text,
          images: imagesOf(images),
          maxOutputTokens: maxOutputTokens,
          reasoningEffort: reasoningEffort,
          requestId: _requestId(),
          uiAction: uiAction == null || uiAction.isEmpty ? null : uiAction,
        ),
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
  /// conversation kept in Stream Chat is rewound there too: the later turns are hidden in
  /// the channel and kept as a branch that can be brought back.
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
