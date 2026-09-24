// Between the generated wire types and the hand-written public ones. Not exported.

import 'generated/api.dart' as api;
import 'models.dart';

Session sessionOf(api.Session wire) => Session(
  id: wire.id,
  callId: wire.callId,
  callType: wire.callType,
  userId: wire.userId,
  agentId: wire.agentId,
  isText: wire.text ?? false,
  state: _named(SessionState.values, wire.state, SessionState.unknown),
  mode: _named(SessionMode.values, wire.mode, SessionMode.unknown),
  agent: wire.agent ?? '',
  configId: wire.configId ?? '',
  conversationId: wire.conversationId ?? '',
  instructions: wire.instructions ?? '',
  llm: wire.llm ?? '',
  stt: wire.stt ?? '',
  tts: wire.tts ?? '',
  sts: wire.sts ?? '',
  voice: wire.voice ?? '',
  title: wire.title ?? '',
  description: wire.description ?? '',
  project: wire.project ?? '',
  custom: wire.custom ?? const {},
  incognito: wire.incognito ?? false,
  forkedFrom: wire.forkedFrom ?? '',
  createdAt: wire.createdAt,
  closedAt: wire.closedAt,
  lastResponseAt: wire.lastResponseAt,
);

AgentResponse responseOf(api.AgentResponse wire) => AgentResponse(
  id: wire.id,
  sessionId: wire.sessionId,
  said: wire.said ?? '',
  status: _named(ResponseStatus.values, wire.status, ResponseStatus.unknown),
  error: wire.error ?? '',
  createdAt: wire.createdAt,
  finishedAt: wire.finishedAt,
);

ResponseItem itemOf(api.AgentResponseItem wire) => ResponseItem(
  responseId: wire.responseId,
  ordinal: wire.ordinal,
  sessionId: wire.sessionId ?? '',
  kind: switch (wire.kind) {
    'said' => ItemKind.said,
    'thought' => ItemKind.thought,
    'tool_call' => ItemKind.toolCall,
    'tool_result' => ItemKind.toolResult,
    'answer' => ItemKind.answer,
    'blocked' => ItemKind.blocked,
    'error' => ItemKind.error,
    _ => ItemKind.unknown,
  },
  type: wire.kind,
  text: wire.text ?? '',
  toolName: wire.toolName ?? '',
  payload: wire.payload ?? const {},
  at: wire.at,
);

GuestUser guestOf(api.GuestUser wire) => GuestUser(
  id: wire.id,
  token: wire.token,
  name: wire.name ?? '',
  custom: wire.custom ?? const {},
  expiresAt: wire.expiresAt,
);

/// The request for a session. [agent] is the name to open it against when the options name
/// no config of their own.
api.CreateSessionRequest createRequestOf(SessionOptions options, {String? callId, String? agent}) =>
    api.CreateSessionRequest(
      callId: _set(callId),
      // Held in writing unless a call was named: a conversation with no call is one typed.
      text: _set(callId) == null ? true : null,
      agent: _set(options.agent) ?? (_set(options.configId) == null ? _set(agent) : null),
      configId: _set(options.configId),
      instructions: options.instructions,
      greeting: options.greeting,
      llm: _set(options.llm),
      stt: _set(options.stt),
      tts: _set(options.tts),
      voice: _set(options.voice),
      title: options.title,
      description: options.description,
      project: options.project,
      custom: options.custom,
      incognito: options.incognito,
      persistConversation: options.persistConversation,
      conversationId: _set(options.conversationId),
      modelOverwrites: overwritesOf(options.modelOverwrites),
      tools: options.tools.isEmpty
          ? null
          : [
              for (final tool in options.tools)
                api.SessionTool(
                  name: tool.name,
                  description: tool.description,
                  parameters: tool.parameters,
                ),
            ],
      tags: options.tags.isEmpty ? null : options.tags,
    );

api.ForkSessionRequest forkRequestOf(ForkOptions options) => api.ForkSessionRequest(
  responseId: _set(options.responseId),
  agent: _set(options.agent),
  configId: _set(options.configId),
  title: options.title,
  description: options.description,
  project: options.project,
  custom: options.custom,
  instructions: options.instructions,
  incognito: options.incognito,
  // Only ever false: true is the default, and sending the default is how a caller loses what
  // the parent was opened with.
  messages: options.withoutHistory ? false : null,
  modelOverwrites: overwritesOf(options.modelOverwrites),
  callId: _set(options.callId),
);

api.ModelOverwrites? overwritesOf(ModelOverwrites? overwrites) => overwrites == null
    ? null
    : api.ModelOverwrites(
        llm: _set(overwrites.llm),
        stt: _set(overwrites.stt),
        tts: _set(overwrites.tts),
        sts: _set(overwrites.sts),
        subagent: _set(overwrites.subagent),
        search: _set(overwrites.search),
        thinking: overwrites.thinking?.name,
        temperature: overwrites.temperature,
        maxOutputTokens: overwrites.maxOutputTokens,
        verbosity: overwrites.verbosity?.name,
      );

List<api.ImageSource>? imagesOf(List<AgentImage> images) => images.isEmpty
    ? null
    : [for (final image in images) api.ImageSource(url: image.url, detail: image.detail)];

/// A name the caller left empty, which means the same as leaving it out.
String? _set(String? value) => value == null || value.isEmpty ? null : value;

T _named<T extends Enum>(List<T> values, String? wire, T fallback) {
  for (final value in values) {
    if (value.name == wire) {
      return value;
    }
  }
  return fallback;
}
