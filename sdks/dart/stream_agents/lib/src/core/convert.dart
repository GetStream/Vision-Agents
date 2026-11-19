// Between the generated wire types and the hand-written public ones. Not exported.

import 'errors.dart';
import 'generated/api.dart' as api;
import 'models.dart';
import 'tools.dart';
import 'wire.dart' show parseRouterDate;

Session sessionOf(api.Session wire) => Session(
  id: wire.id,
  callId: wire.callId,
  callType: wire.callType,
  userId: wire.userId,
  agentId: wire.agentId,
  isText: wire.text ?? false,
  state: _named(SessionState.values, wire.state, SessionState.unknown),
  mode: _named(SessionMode.values, wire.mode, SessionMode.unknown),
  modality: _named(SessionModality.values, wire.modality, SessionModality.unknown),
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
  projectId: wire.projectId ?? '',
  team: wire.team ?? '',
  custom: wire.custom ?? const {},
  incognito: wire.incognito ?? false,
  forkedFrom: wire.forkedFrom ?? '',
  createdAt: wire.createdAt,
  closedAt: wire.closedAt,
  lastResponseAt: wire.lastResponseAt,
  pinnedAt: wire.pinnedAt,
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
/// no config of their own, and [userId] whose memories it reads.
api.CreateSessionRequest createRequestOf(
  SessionOptions options, {
  required bool startVoice,
  String? agent,
  String userId = '',
}) => api.CreateSessionRequest(
  id: _set(options.id),
  startVoice: startVoice ? true : null,
  agent: _set(options.agent) ?? (_set(options.configId) == null ? _set(agent) : null),
  configId: _set(options.configId),
  greeting: greetingOf(options.greeting),
  llm: _set(options.llm),
  stt: _set(options.stt),
  tts: _set(options.tts),
  voice: _set(options.voice),
  title: options.title,
  description: options.description,
  projectId: options.projectId,
  custom: options.custom,
  incognito: options.incognito,
  modelOverwrites: overwritesOf(options.modelOverwrites),
  tools: options.tools.isEmpty ? null : [for (final tool in options.tools) sessionToolOf(tool)],
  tags: options.tags.isEmpty ? null : options.tags,
  returnUrl: _set(options.returnUrl),
  // The router remembers only for a user id, and a device only ever means its own user.
  memory: switch (options.memory) {
    null => null,
    final memory => api.SessionMemory(
      appId: _set(memory.appId),
      filter: memory.filter.isEmpty ? null : memory.filter,
      userId: _set(userId),
    ),
  },
  timezone: _set(options.timezone),
);

api.ForkSessionRequest forkRequestOf(ForkOptions options) => api.ForkSessionRequest(
  responseId: _set(options.responseId),
  agent: _set(options.agent),
  configId: _set(options.configId),
  title: options.title,
  description: options.description,
  projectId: options.projectId,
  custom: options.custom,
  incognito: options.incognito,
  // Only ever false: true is the default, and sending the default is how a caller loses what
  // the parent was opened with.
  messages: options.withoutHistory ? false : null,
  modelOverwrites: overwritesOf(options.modelOverwrites),
);

/// A tool as the router is told of it, when a session opens or while it runs.
api.SessionTool sessionToolOf(AgentTool tool) => api.SessionTool(
  name: tool.name,
  description: tool.description,
  parameters: tool.parameters,
  displayTitle: _set(tool.displayTitle),
  executor: tool.executor?.name,
  approval: switch (tool.approval) {
    null => null,
    final approval => api.SessionToolApproval(
      title: approval.title,
      message: approval.message,
      reasonArgument: approval.reasonArgument,
      allowTitle: approval.allowTitle,
      declineTitle: approval.declineTitle,
    ),
  },
);

api.Greeting? greetingOf(Greeting? greeting) =>
    greeting == null ? null : api.Greeting(text: greeting.text, mode: greeting.mode?.name);

api.ModelOverwrites? overwritesOf(ModelOverwrites? overwrites) => overwrites == null
    ? null
    : api.ModelOverwrites(
        llm: _set(overwrites.llm),
        stt: _set(overwrites.stt),
        tts: _set(overwrites.tts),
        sts: _set(overwrites.sts),
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

ScheduledTask scheduledTaskOf(api.ScheduledTask wire) => ScheduledTask(
  id: wire.id,
  title: wire.title,
  instructions: wire.instructions,
  schedule: ScheduledTaskSchedule(
    at: wire.schedule.at,
    after: wire.schedule.inValue,
    repeat: wire.schedule.repeat,
    timezone: wire.schedule.timezone,
    until: wire.schedule.until,
    description: wire.schedule.description ?? '',
  ),
  deliverTo: wire.deliverTo,
  quietHours: switch (wire.quietHours) {
    null => null,
    final hours => ScheduledTaskQuietHours(start: hours.start, end: hours.end),
  },
  state: _named(ScheduledTaskState.values, wire.state, ScheduledTaskState.unknown),
  nextRunAt: _dateOrNull(wire.nextRunAt, 'ScheduledTask.next_run_at'),
  upcoming: [for (final at in wire.upcoming) _date(at, 'ScheduledTask.upcoming')],
  lastRunAt: _dateOrNull(wire.lastRunAt, 'ScheduledTask.last_run_at'),
  consecutiveFailures: wire.consecutiveFailures,
  createdBy: _named(ScheduledTaskCreatedBy.values, wire.createdBy, ScheduledTaskCreatedBy.unknown),
  sessionId: wire.sessionId,
  configId: wire.configId,
  custom: wire.custom ?? const {},
  createdAt: wire.createdAt,
  updatedAt: wire.updatedAt,
);

ScheduledTaskRun scheduledTaskRunOf(api.ScheduledTaskRun wire) => ScheduledTaskRun(
  id: wire.id,
  taskId: wire.taskId,
  scheduledFor: _date(wire.scheduledFor, 'ScheduledTaskRun.scheduled_for'),
  trigger: _named(ScheduledTaskRunTrigger.values, wire.trigger, ScheduledTaskRunTrigger.unknown),
  state: _named(ScheduledTaskRunState.values, wire.state, ScheduledTaskRunState.unknown),
  deliveredTo: wire.deliveredTo,
  sessionId: wire.sessionId,
  attempts: wire.attempts,
  startedAt: wire.startedAt,
  finishedAt: wire.finishedAt,
  error: wire.error ?? '',
);

/// A change to a task as the router takes one. Its instructions are a backend's to change.
api.UpdateScheduledTaskRequest updateTaskRequestOf({
  String? title,
  ScheduledTaskSchedule? schedule,
  ScheduledTaskNext? next,
  List<String>? deliverTo,
  ScheduledTaskQuietHours? quietHours,
}) => api.UpdateScheduledTaskRequest(
  title: title,
  schedule: switch (schedule) {
    null => null,
    final schedule => api.ScheduledTaskSchedule(
      at: schedule.at,
      inValue: schedule.after,
      repeat: schedule.repeat,
      timezone: schedule.timezone,
      until: schedule.until,
    ),
  },
  next: switch (next) {
    null => null,
    ScheduledTaskNextAt(:final at) => api.ScheduledTaskNext(at: at),
    ScheduledTaskNextAfter(:final duration) => api.ScheduledTaskNext(inValue: duration),
  },
  deliverTo: deliverTo,
  quietHours: switch (quietHours) {
    null => null,
    final hours => api.ScheduledTaskQuietHours(start: hours.start, end: hours.end),
  },
);

/// A timestamp the router wrote as a plain string, refused by name when it is not one.
DateTime _date(String value, String at) =>
    parseRouterDate(value) ?? (throw UnreadableException('$at is not a timestamp'));

/// The same, where empty or absent means there is none.
DateTime? _dateOrNull(String? value, String at) =>
    value == null || value.isEmpty ? null : _date(value, at);
