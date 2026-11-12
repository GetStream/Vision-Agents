import 'tools.dart';

/// Whether the agent is still in the conversation.
enum SessionState {
  live,
  ended,

  /// A state this SDK has never heard of.
  unknown,
}

/// How a session hears and speaks.
enum SessionMode {
  /// A transcriber, a conversation model and a voice.
  cascade,

  /// One speech-to-speech model.
  native,

  /// In writing.
  text,

  /// A mode this SDK has never heard of, or none reported.
  unknown,
}

/// How the user took part.
enum SessionModality {
  text,
  voice,
  video,

  /// A modality this SDK has never heard of.
  unknown,
}

/// A running or finished conversation.
final class Session {
  const Session({
    required this.id,
    required this.createdAt,
    this.callId = '',
    this.callType = '',
    this.userId = '',
    this.agentId = '',
    this.isText = false,
    this.state = SessionState.live,
    this.mode = SessionMode.unknown,
    this.modality = SessionModality.unknown,
    this.agent = '',
    this.configId = '',
    this.conversationId = '',
    this.instructions = '',
    this.llm = '',
    this.stt = '',
    this.tts = '',
    this.sts = '',
    this.voice = '',
    this.title = '',
    this.description = '',
    this.projectId = '',
    this.team = '',
    this.custom = const {},
    this.incognito = false,
    this.forkedFrom = '',
    this.closedAt,
    this.lastResponseAt,
    this.pinnedAt,
  });

  /// What the router holds this session by. It addresses the session and its socket, and it
  /// is not the call id.
  final String id;

  /// The Stream call the agent joined, which is what a video SDK joins. Empty for a text
  /// session.
  final String callId;
  final String callType;

  /// Who the agent is on the call, which is also who its replies are written as in chat.
  final String userId;

  /// Keys the transcript and the statistics.
  final String agentId;
  final bool isText;
  final SessionState state;
  final SessionMode mode;
  final SessionModality modality;

  /// The agent config's name, as the session was opened against it.
  final String agent;
  final String configId;

  /// The Stream Chat channel the conversation is kept in, as `type:id`. Empty for a call and
  /// for an incognito session.
  final String conversationId;
  final String instructions;

  /// The models answering, once routing has picked them.
  final String llm;
  final String stt;
  final String tts;
  final String sts;
  final String voice;

  /// The labels a person finds the conversation by again. Never shown to the model.
  final String title;
  final String description;
  final String projectId;

  /// The team the session's usage is counted under. Empty for none.
  final String team;

  /// The caller's own, handed back untouched.
  final Map<String, Object?> custom;

  /// Nothing about the session was recorded, so it cannot be listed, rewound or forked.
  final bool incognito;

  /// The session this one was forked from, empty for one opened fresh.
  final String forkedFrom;
  final DateTime createdAt;
  final DateTime? closedAt;
  final DateTime? lastResponseAt;

  /// When the user pinned this conversation, null for one that is not pinned.
  final DateTime? pinnedAt;

  @override
  bool operator ==(Object other) =>
      other is Session &&
      other.id == id &&
      other.callId == callId &&
      other.callType == callType &&
      other.userId == userId &&
      other.agentId == agentId &&
      other.isText == isText &&
      other.state == state &&
      other.mode == mode &&
      other.modality == modality &&
      other.agent == agent &&
      other.configId == configId &&
      other.conversationId == conversationId &&
      other.instructions == instructions &&
      other.llm == llm &&
      other.stt == stt &&
      other.tts == tts &&
      other.sts == sts &&
      other.voice == voice &&
      other.title == title &&
      other.description == description &&
      other.projectId == projectId &&
      other.team == team &&
      _sameJson(other.custom, custom) &&
      other.incognito == incognito &&
      other.forkedFrom == forkedFrom &&
      other.createdAt == createdAt &&
      other.closedAt == closedAt &&
      other.lastResponseAt == lastResponseAt &&
      other.pinnedAt == pinnedAt;

  @override
  int get hashCode => Object.hashAll([
    id,
    callId,
    callType,
    userId,
    agentId,
    isText,
    state,
    mode,
    modality,
    agent,
    configId,
    conversationId,
    instructions,
    llm,
    stt,
    tts,
    sts,
    voice,
    title,
    description,
    projectId,
    team,
    Object.hashAllUnordered(custom.keys),
    incognito,
    forkedFrom,
    createdAt,
    closedAt,
    lastResponseAt,
    pinnedAt,
  ]);

  @override
  String toString() => 'Session($id, ${state.name})';
}

/// How answering one turn ended.
enum ResponseStatus {
  running,
  completed,
  failed,

  /// Interrupted by the caller, which is not a failure: what was said still counts.
  cancelled,

  /// A status this SDK has never heard of.
  unknown,
}

/// One turn of a session as the router wrote it down: what was asked, and how answering it
/// ended. This is what a rewind or a fork names.
///
/// Its [id] is not the `turn_id` socket events carry.
final class AgentResponse {
  const AgentResponse({
    required this.id,
    required this.sessionId,
    required this.createdAt,
    this.said = '',
    this.status = ResponseStatus.unknown,
    this.error = '',
    this.finishedAt,
  });

  final String id;
  final String sessionId;

  /// What the person asked. Empty for a turn the agent started on its own, like a greeting.
  final String said;
  final ResponseStatus status;

  /// What went wrong, for a failed turn.
  final String error;
  final DateTime createdAt;
  final DateTime? finishedAt;

  @override
  bool operator ==(Object other) =>
      other is AgentResponse &&
      other.id == id &&
      other.sessionId == sessionId &&
      other.said == said &&
      other.status == status &&
      other.error == error &&
      other.createdAt == createdAt &&
      other.finishedAt == finishedAt;

  @override
  int get hashCode => Object.hash(id, sessionId, said, status, error, createdAt, finishedAt);

  @override
  String toString() => 'AgentResponse($id, ${status.name})';
}

/// What one step of a turn was.
enum ItemKind {
  said,
  thought,
  toolCall,
  toolResult,
  answer,
  blocked,
  error,

  /// A kind this SDK has never heard of.
  unknown,
}

/// One thing that happened inside a turn, in the order it happened.
///
/// Deltas are not items: a hundred fragments of one sentence are the sentence.
final class ResponseItem {
  const ResponseItem({
    required this.responseId,
    required this.ordinal,
    required this.at,
    this.sessionId = '',
    this.kind = ItemKind.unknown,
    this.type = '',
    this.text = '',
    this.toolName = '',
    this.payload = const {},
  });

  /// The response it belongs to, which is what a rewind or a fork names.
  final String responseId;

  /// The position within the response.
  final int ordinal;
  final String sessionId;
  final ItemKind kind;

  /// The router's own name for the kind, which is all there is for an unknown one.
  final String type;
  final String text;
  final String toolName;

  /// Whatever the kind carries that text cannot: a tool's arguments, a guardrail's reason.
  final Map<String, Object?> payload;
  final DateTime at;

  @override
  String toString() => 'ResponseItem($responseId#$ordinal, $type)';
}

/// Somebody talking to an agent before signing up.
///
/// The [token] is a Stream user token with role guest: it is what the chat and video SDKs
/// connect with, and what a router verifying tokens takes as this guest.
final class GuestUser {
  const GuestUser({
    required this.id,
    required this.token,
    this.name = '',
    this.custom = const {},
    this.expiresAt,
  });

  final String id;
  final String token;
  final String name;
  final Map<String, Object?> custom;

  /// When the token stops working. A guest coming back after it is minted a fresh one.
  final DateTime? expiresAt;

  @override
  bool operator ==(Object other) => other is GuestUser && other.id == id && other.token == token;

  @override
  int get hashCode => Object.hash(id, token);

  @override
  String toString() => 'GuestUser($id)';
}

/// How hard to reason before answering.
enum Thinking { none, minimal, low, medium, high }

/// How much detail to give.
enum Verbosity { low, medium, high }

/// What to change about the models for one conversation, over whatever its config decided.
///
/// Only the safe knobs are here. Instructions and tools are not, because a caller able to
/// rewrite those could make a session impersonate a different agent.
final class ModelOverwrites {
  const ModelOverwrites({
    this.llm,
    this.stt,
    this.tts,
    this.sts,
    this.search,
    this.thinking,
    this.temperature,
    this.maxOutputTokens,
    this.verbosity,
  });

  final String? llm;
  final String? stt;
  final String? tts;
  final String? sts;
  final String? search;
  final Thinking? thinking;

  /// Omitted leaves the provider's own default, which is not the same as zero.
  final double? temperature;
  final int? maxOutputTokens;
  final Verbosity? verbosity;
}

/// An image handed to the model alongside a question.
final class AgentImage {
  const AgentImage(this.url, {this.detail});

  /// An absolute HTTP(S) URL or a base64 data URI.
  final String url;

  /// `auto`, `low` or `high`.
  final String? detail;
}

/// What a session should be, for the cases the shorthands do not cover.
///
/// Everything is optional because everything has an answer already: a named config decides
/// what this does not say, and the router decides what the config does not. A field left
/// null is left out of the request, never sent as a copy of the server's default.
final class SessionOptions {
  const SessionOptions({
    this.id,
    this.agent,
    this.configId,
    this.startVoice = false,
    this.greeting,
    this.llm,
    this.stt,
    this.tts,
    this.voice,
    this.title,
    this.description,
    this.projectId,
    this.custom,
    this.incognito,
    this.modelOverwrites,
    this.tools = const [],
    this.tags = const {},
    this.returnUrl,
    this.memory,
    this.timezone,
  });

  /// An id to hold the session by, for a caller that wants to know it before the session
  /// exists: up to 64 letters, digits, `-` and `_`. One already taken is refused with a 409.
  /// Left out, the router generates one.
  final String? id;

  /// The agent config to start from, by the name it was stored under.
  final String? agent;

  /// The agent config to start from, by id. Naming both this and [agent] is refused.
  final String? configId;

  /// Has the agent join the session's own call, `agent:<session id>`, as soon as it opens.
  /// False holds the conversation in writing until [AgentSession.startVoice] starts it.
  final bool startVoice;

  /// What the agent opens the call with.
  final Greeting? greeting;
  final String? llm;
  final String? stt;
  final String? tts;

  /// The voice the agent speaks in: a provider's own voice id, or one of your voices by id or by
  /// the name you gave it. Prefix it with `custom:` to mean only yours.
  final String? voice;

  /// What to call the conversation, for a list a person reads. Never shown to the model.
  final String? title;
  final String? description;

  /// What the conversation belongs to, which is also its "project" cost tag.
  final String? projectId;

  /// Anything of the caller's own. Sessions can be queried by it.
  final Map<String, Object?>? custom;

  /// Hold the conversation and record nothing about it. It cannot be found, rewound or
  /// forked afterwards, which is the point.
  final bool? incognito;
  final ModelOverwrites? modelOverwrites;

  /// Functions of yours the agent may call, answered on this device.
  final List<AgentTool> tools;

  /// Cost labels, carried onto every request the session makes.
  final Map<String, String> tags;

  /// Where a sign-in the conversation asks for goes back to. Name the app's own scheme in
  /// reverse domain form, such as `com.example.app:/connected`, and the router hands the
  /// browser back to the app. It is kept with the session, so a conversation carried on later
  /// goes back there too. Null sends the sign-in back to the dashboard, which a phone cannot
  /// open.
  final String? returnUrl;

  /// Has the agent remember the signed-in user: what it learned about them before is recalled
  /// here, and what it learns here is kept for their next session. Null remembers nothing, and
  /// an [incognito] session recalls but keeps nothing new.
  final SessionMemory? memory;

  /// The user's IANA zone, such as `Europe/Amsterdam`, which the device knows and the router
  /// does not. Dart's own `DateTime.timeZoneName` is an abbreviation such as `CET`, not a
  /// zone, so read it with a plugin such as `flutter_timezone`. The agent reads dates in it,
  /// and it is remembered as the user's for the scheduled tasks that follow. Null sends none,
  /// and a zone the router does not know is refused.
  final String? timezone;
}

/// What a session remembers about the user [Backend.userId] names. It is always theirs: a
/// device recalls nobody else's. With no user id, nothing is recalled or kept.
final class SessionMemory {
  const SessionMemory({this.filter = const {}, this.appId});

  /// Your own labels, which narrow what is recalled. They cannot widen it past the user.
  final Map<String, String> filter;

  /// Separates two of your apps sharing one memory account.
  final String? appId;
}

/// What the agent says as it joins a call, before anyone speaks.
final class Greeting {
  const Greeting(this.text, {this.mode});

  /// Empty means the agent waits to be spoken to.
  final String text;

  /// Null says it word for word, as [GreetingMode.exact] does.
  final GreetingMode? mode;
}

enum GreetingMode {
  /// Word for word.
  exact,

  /// The model rewords it on every call, so callers do not hear the same opening.
  variation,
}

/// What to change about a conversation while continuing it as a new one.
///
/// Null means the fork keeps what the parent had.
final class ForkOptions {
  const ForkOptions({
    this.responseId,
    this.agent,
    this.configId,
    this.title,
    this.description,
    this.projectId,
    this.custom,
    this.incognito,
    this.withoutHistory = false,
    this.modelOverwrites,
  }) : assert(
         !(withoutHistory && responseId != null),
         'a fork at a response carries the history up to it',
       );

  /// Carry the history only up to the end of this response, and branch from there.
  final String? responseId;

  /// Another agent config to continue as, by name.
  final String? agent;

  /// Another agent config to continue as, by id.
  final String? configId;
  final String? title;
  final String? description;
  final String? projectId;
  final Map<String, Object?>? custom;
  final bool? incognito;

  /// Start the fork with none of the parent's history. Cannot be combined with [responseId],
  /// which is a point in that history.
  final bool withoutHistory;
  final ModelOverwrites? modelOverwrites;
}

/// One key to order conversations by.
///
/// [field] is the router's own name for it: `last_message_at`, when the agent last answered;
/// `pinned`, the pinned ones first, which comes first and may be followed by
/// `last_message_at`; or `relevance`, best match first, which only sorts a query whose filter
/// has `text`. Ordering by review score is a backend's, not a device's, and is refused before
/// anything is sent.
final class SessionSort {
  const SessionSort({required this.field, this.direction = -1});

  /// `last_message_at`, `pinned` or `relevance`.
  final String field;

  /// -1 is descending and 1 ascending. `pinned` and `relevance` only sort descending.
  final int direction;

  @override
  bool operator ==(Object other) =>
      other is SessionSort && other.field == field && other.direction == direction;

  @override
  int get hashCode => Object.hash(field, direction);

  @override
  String toString() => 'SessionSort($field, $direction)';
}

/// One page of a list, and where the next one starts.
final class ListPage<T> {
  const ListPage(this.items, {this.hasMore = false, this.nextCursor});

  final List<T> items;

  /// Whether there is a page after this one.
  final bool hasMore;

  /// What to ask for the next page with, null on the last one.
  final String? nextCursor;
}

/// Whether two values decoded from JSON hold the same thing, maps and lists compared by what
/// is in them.
bool _sameJson(Object? a, Object? b) => switch ((a, b)) {
  (final Map<Object?, Object?> a, final Map<Object?, Object?> b) =>
    a.length == b.length &&
        a.entries.every(
          (entry) => b.containsKey(entry.key) && _sameJson(entry.value, b[entry.key]),
        ),
  (final List<Object?> a, final List<Object?> b) =>
    a.length == b.length && a.indexed.every((item) => _sameJson(item.$2, b[item.$1])),
  _ => a == b,
};
