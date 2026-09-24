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
    this.project = '',
    this.custom = const {},
    this.incognito = false,
    this.forkedFrom = '',
    this.closedAt,
    this.lastResponseAt,
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

  /// The agent config's name, as the session was opened against it.
  final String agent;
  final String configId;

  /// The Stream Chat channel the conversation is kept in, as `type:id`. Empty unless the
  /// session was opened with `persistConversation`.
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
  final String project;

  /// The caller's own, handed back untouched.
  final Map<String, Object?> custom;

  /// Nothing about the session was recorded, so it cannot be listed, rewound or forked.
  final bool incognito;

  /// The session this one was forked from, empty for one opened fresh.
  final String forkedFrom;
  final DateTime createdAt;
  final DateTime? closedAt;
  final DateTime? lastResponseAt;

  @override
  bool operator ==(Object other) =>
      other is Session && other.id == id && other.state == state && other.title == title;

  @override
  int get hashCode => Object.hash(id, state, title);

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
      other is AgentResponse && other.id == id && other.status == status;

  @override
  int get hashCode => Object.hash(id, status);

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
    this.subagent,
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
  final String? subagent;
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
    this.agent,
    this.configId,
    this.instructions,
    this.greeting,
    this.llm,
    this.stt,
    this.tts,
    this.voice,
    this.title,
    this.description,
    this.project,
    this.custom,
    this.incognito,
    this.persistConversation,
    this.conversationId,
    this.modelOverwrites,
    this.tools = const [],
    this.tags = const {},
  });

  /// The agent config to start from, by the name it was stored under.
  final String? agent;

  /// The agent config to start from, by id. Naming both this and [agent] is refused.
  final String? configId;

  /// The system prompt.
  final String? instructions;

  /// Said on joining without going through the model.
  final String? greeting;
  final String? llm;
  final String? stt;
  final String? tts;

  /// A provider-specific voice id.
  final String? voice;

  /// What to call the conversation, for a list a person reads. Never shown to the model.
  final String? title;
  final String? description;

  /// What the conversation belongs to, which is also its "project" cost tag.
  final String? project;

  /// Anything of the caller's own. Sessions can be queried by it.
  final Map<String, Object?>? custom;

  /// Hold the conversation and record nothing about it. It cannot be found, rewound or
  /// forked afterwards, which is the point.
  final bool? incognito;

  /// Keep a text conversation in Stream Chat.
  final bool? persistConversation;

  /// A Stream Chat channel, as `type:id`, to resume. Leave it out on a first open: the
  /// channel is the backend's to name, and a resume passes the one the first open was given.
  final String? conversationId;
  final ModelOverwrites? modelOverwrites;

  /// Functions of yours the agent may call, answered on this device.
  final List<AgentTool> tools;

  /// Cost labels, carried onto every request the session makes.
  final Map<String, String> tags;
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
    this.project,
    this.custom,
    this.instructions,
    this.incognito,
    this.withoutHistory = false,
    this.modelOverwrites,
    this.callId,
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
  final String? project;
  final Map<String, Object?>? custom;
  final String? instructions;
  final bool? incognito;

  /// Start the fork with none of the parent's history. Cannot be combined with [responseId],
  /// which is a point in that history.
  final bool withoutHistory;
  final ModelOverwrites? modelOverwrites;

  /// The call the fork joins, which a voice session needs and a text session refuses.
  final String? callId;
}

/// Whether a query wants the sessions still running, the ones that ended, or both.
enum SessionFilter { running, closed }

/// Which conversations to list.
final class SessionQuery {
  const SessionQuery({
    this.agent,
    this.configId,
    this.project,
    this.userId,
    this.state,
    this.custom,
    this.createdAfter,
    this.createdBefore,
    this.limit,
    this.offset,
  });

  /// Only those opened against this agent name.
  final String? agent;
  final String? configId;
  final String? project;

  /// Only this user's. A device is narrowed to its own whatever it asks for.
  final String? userId;

  /// Omitted is both.
  final SessionFilter? state;

  /// Labels a session must carry, every one of them.
  final Map<String, Object?>? custom;
  final DateTime? createdAfter;
  final DateTime? createdBefore;

  /// Up to 200. Omitted is 25, and a page shorter than the limit is the last one.
  final int? limit;
  final int? offset;
}
