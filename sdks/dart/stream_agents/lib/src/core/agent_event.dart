import 'dart:convert';

import 'wire.dart';

/// Who said something.
final class Participant {
  const Participant({this.id = '', this.userId = '', this.name = ''});

  factory Participant.fromJson(Map<String, Object?> fields) => Participant(
    id: _text(fields['id']),
    userId: _text(fields['user_id']),
    name: _text(fields['name']),
  );

  final String id;
  final String userId;
  final String name;

  /// What to show for this participant: their name, or their user id when they have none.
  String get display => name.isEmpty ? userId : name;

  @override
  bool operator ==(Object other) =>
      other is Participant && other.id == id && other.userId == userId && other.name == name;

  @override
  int get hashCode => Object.hash(id, userId, name);
}

/// The events the router documents. A frame outside this set is still delivered.
enum AgentEventKind {
  joined('joined'),
  participantJoined('participant_joined'),
  participantLeft('participant_left'),
  hearing('hearing'),
  heard('heard'),
  decision('decision'),
  responding('responding'),
  responseDelta('response_delta'),
  uiDelta('ui_delta'),
  uiCompleted('ui_completed'),
  responded('responded'),
  blocked('blocked'),
  spoke('spoke'),
  turn('turn'),
  delegated('delegated'),
  taskSettled('task_settled'),
  taskCancelled('task_cancelled'),
  toolCall('tool_call'),
  toolCancel('tool_cancel'),
  toolHeld('tool_held'),
  toolStarted('tool_started'),
  toolUpdated('tool_updated'),
  toolRan('tool_ran'),
  toolCallStarted('tool_call.started'),
  toolCallProgress('tool_call.progress'),
  toolCallFinished('tool_call.finished'),
  transferred('transferred'),
  pressed('pressed'),
  lookedUp('looked_up'),
  backchannel('backchannel'),
  interrupted('interrupted'),
  overlapDecided('overlap_decided'),
  conversationCompacted('conversation_compacted'),
  conversationUpdated('conversation_updated'),
  connectorScopeRequired('connector_scope_required'),
  commandAccepted('command_accepted'),
  commandStopped('command_stopped'),
  modelsChanged('models_changed'),
  error('error'),
  left('left');

  const AgentEventKind(this.wire);

  /// The router's name for it.
  final String wire;

  static final Map<String, AgentEventKind> _byWire = {for (final kind in values) kind.wire: kind};
}

/// One event on a session's socket.
///
/// The fields are kept as they arrived, so an event added to the router after this SDK
/// shipped still reaches the caller: [kind] is null and [type] names it. Switching on [kind]
/// covers what is known; `event['whatever']` covers the rest.
final class AgentEvent {
  const AgentEvent(this.type, [this.fields = const {}]);

  /// Reads one frame, or returns null for one that is not a JSON object with a type.
  ///
  /// Null rather than a throw, because an unreadable frame is skipped rather than ending the
  /// conversation, which is what the router does with a command it cannot read.
  static AgentEvent? tryParse(String frame) {
    final Object? decoded;
    try {
      decoded = jsonDecode(frame);
    } on FormatException {
      return null;
    }
    if (decoded is! Map || decoded['type'] is! String) {
      return null;
    }
    final fields = decoded.cast<String, Object?>();
    return AgentEvent(fields['type'] as String, {
      for (final MapEntry(:key, :value) in fields.entries)
        if (key != 'type') key: value,
    });
  }

  /// The router's own name for this event, always present.
  final String type;

  /// The event's fields, flattened as the router sends them.
  final Map<String, Object?> fields;

  /// This event as one of the kinds the SDK knows, or null for one it does not.
  AgentEventKind? get kind => AgentEventKind._byWire[type];

  Object? operator [](String key) => fields[key];

  /// What was said, transcribed or generated, depending on the event.
  String get text => _text(fields['text']);

  /// The turn this belongs to, or the empty string for events outside a turn.
  String get turnId => _text(fields['turn_id']);

  /// Who this is about, or null for the events that are about nobody.
  Participant? get participant => switch (fields['participant']) {
    final Map<Object?, Object?> who => Participant.fromJson(who.cast<String, Object?>()),
    _ => null,
  };

  /// What went wrong, for `error` and for the events that carry a failure of their own.
  String get errorText => _text(fields['error']);

  /// A tool the model wants run, or null when this event is not a tool call.
  ToolCall? get toolCall => kind == AgentEventKind.toolCall
      ? ToolCall(
          id: _text(fields['id']),
          name: _text(fields['name']),
          arguments: _text(fields['arguments']),
          requestId: _text(fields['request_id']),
          turnId: turnId,
          approved: fields['approved'] == true,
        )
      : null;

  /// The call's whole state on a `tool_call.started`, `tool_call.progress` or
  /// `tool_call.finished`, or null for any other event.
  ToolCallState? get toolCallState => switch (kind) {
    AgentEventKind.toolCallStarted ||
    AgentEventKind.toolCallProgress ||
    AgentEventKind.toolCallFinished when _text(fields['tool_call_id']).isNotEmpty => ToolCallState(
      id: _text(fields['tool_call_id']),
      tool: _text(fields['tool']),
      turnId: turnId,
      status: ToolCallStatus._byWire[_text(fields['status'])] ?? ToolCallStatus.unknown,
      progress: _text(fields['progress']),
      startedAt: parseRouterDate(_text(fields['started_at'])),
      duration: Duration(
        milliseconds: switch (fields['duration_ms']) {
          final num ms => ms.toInt(),
          _ => 0,
        },
      ),
      approvedAt: parseRouterDate(_text(fields['approved_at'])),
      rejectedAt: parseRouterDate(_text(fields['rejected_at'])),
      error: _text(fields['error']),
    ),
    _ => null,
  };

  /// A connector tool call the provider refused because the person's own connection lacks
  /// access it asked for, and the step-up consent the router began for it, or null when this
  /// event is not one. The old grant keeps working until the step-up succeeds, and the same
  /// call works afterwards in the same session.
  ///
  /// An app cannot finish the step-up yet: `launchUrl` takes `handoffToken` only from a page on
  /// the dashboard's origin, as for a `connector_authorization` attachment.
  ConnectorScopeRequired? get connectorScopeRequired =>
      kind == AgentEventKind.connectorScopeRequired
      ? ConnectorScopeRequired(
          name: _text(fields['name']),
          connectorId: _text(fields['connector_id']),
          connectionId: _text(fields['connection_id']),
          scopes: [
            if (fields['scopes'] case final List<Object?> scopes) ...scopes.whereType<String>(),
          ],
          authorizationId: _text(fields['authorization_id']),
          launchUrl: _text(fields['launch_url']),
          handoffToken: _text(fields['handoff_token']),
          expiresAt: parseRouterDate(_text(fields['expires_at'])),
        )
      : null;

  @override
  String toString() => 'AgentEvent($type)';
}

/// The access a connector tool call needs, and where the person grants it.
final class ConnectorScopeRequired {
  const ConnectorScopeRequired({
    required this.name,
    required this.connectorId,
    required this.connectionId,
    required this.scopes,
    required this.authorizationId,
    required this.launchUrl,
    required this.handoffToken,
    this.expiresAt,
  });

  /// The binding's alias.
  final String name;
  final String connectorId;
  final String connectionId;

  /// What the provider asked for, empty for a claims challenge.
  final List<String> scopes;
  final String authorizationId;
  final String launchUrl;
  final String handoffToken;
  final DateTime? expiresAt;
}

/// A request from the model to run one of the caller's functions.
final class ToolCall {
  const ToolCall({
    required this.id,
    required this.name,
    required this.arguments,
    this.requestId = '',
    this.turnId = '',
    this.approved = false,
  });

  final String id;
  final String name;

  /// Whether the person already said yes to it, on a voice call, so nobody asks again.
  final bool approved;

  /// The arguments as the model wrote them, which is a JSON object encoded as a string.
  final String arguments;

  /// The request that made this call, which its result has to repeat. Empty for
  /// most calls.
  final String requestId;
  final String turnId;

  /// The arguments decoded, or an empty map if the model wrote something else.
  Map<String, Object?> get argumentValues {
    try {
      final decoded = jsonDecode(arguments);
      return decoded is Map ? decoded.cast<String, Object?>() : const {};
    } on FormatException {
      return const {};
    }
  }
}

/// Where a tool call stands. [unknown] is a status the router added after this SDK shipped.
enum ToolCallStatus {
  inProgress('in_progress'),

  /// Held for a person's approval.
  waiting('waiting'),
  success('success'),
  failed('failed'),
  cancelled('cancelled'),
  unknown('');

  const ToolCallStatus(this.wire);

  /// The router's name for it.
  final String wire;

  /// Whether the call is over.
  bool get isFinished => this == success || this == failed || this == cancelled;

  static final Map<String, ToolCallStatus> _byWire = {
    for (final status in values) status.wire: status,
  };
}

/// One tool call as the session last reported it, whichever process runs it.
final class ToolCallState {
  const ToolCallState({
    required this.id,
    required this.tool,
    required this.status,
    this.turnId = '',
    this.progress = '',
    this.startedAt,
    this.duration = Duration.zero,
    this.approvedAt,
    this.rejectedAt,
    this.error = '',
  });

  /// The call's `tool_call_id`.
  final String id;
  final String tool;
  final String turnId;
  final ToolCallStatus status;

  /// The last thing said about the call while it runs, such as "Searching for flights", or the
  /// question it waits on.
  final String progress;
  final DateTime? startedAt;

  /// From [startedAt] to this report; once finished, how long the call took.
  final Duration duration;

  /// When a person allowed the call.
  final DateTime? approvedAt;

  /// When a person declined the call, or its wait for them passed.
  final DateTime? rejectedAt;

  /// Why a failed call failed, or what a declined one shows.
  final String error;

  @override
  bool operator ==(Object other) =>
      other is ToolCallState &&
      other.id == id &&
      other.tool == tool &&
      other.turnId == turnId &&
      other.status == status &&
      other.progress == progress &&
      other.startedAt == startedAt &&
      other.duration == duration &&
      other.approvedAt == approvedAt &&
      other.rejectedAt == rejectedAt &&
      other.error == error;

  @override
  int get hashCode => Object.hash(
    id,
    tool,
    turnId,
    status,
    progress,
    startedAt,
    duration,
    approvedAt,
    rejectedAt,
    error,
  );
}

String _text(Object? value) => value is String ? value : '';
