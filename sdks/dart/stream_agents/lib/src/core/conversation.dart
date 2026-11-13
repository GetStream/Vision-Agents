import 'dart:math';

import 'agent_event.dart';

/// Who said a turn.
sealed class Speaker {
  const Speaker();

  static const agent = AgentSpeaker._();

  bool get isAgent => this is AgentSpeaker;
}

/// The agent.
final class AgentSpeaker extends Speaker {
  const AgentSpeaker._();
}

/// A person, on the call or typing.
final class ParticipantSpeaker extends Speaker {
  const ParticipantSpeaker([this.participant]);

  /// Null when the router did not say who, which is the case for what this device typed.
  final Participant? participant;

  @override
  bool operator ==(Object other) => other is ParticipantSpeaker && other.participant == participant;

  @override
  int get hashCode => participant.hashCode;
}

/// Interactive UI streamed alongside an agent turn.
final class TurnUI {
  const TurnUI({
    required this.format,
    this.version,
    required this.status,
    required this.content,
    this.raw = '',
  });

  final String format;
  final String? version;

  /// Either `streaming` or `complete`.
  final String status;
  final String content;

  /// Unmerged turn-local source accumulated before patch merging.
  final String raw;

  bool get isStreaming => status == 'streaming';

  /// Wire representation matching Stream Chat's `custom.ai_ui` envelope.
  Map<String, Object?> toWireJson() => {
    'v': 1,
    'format': format,
    if (version != null && version!.isNotEmpty) 'version': version,
    'status': status,
    'content': content,
  };

  @override
  bool operator ==(Object other) =>
      other is TurnUI &&
      other.format == format &&
      other.version == version &&
      other.status == status &&
      other.content == content;

  @override
  int get hashCode => Object.hash(format, version, status, content);

  @override
  String toString() => 'TurnUI($format, $status, ${content.length} chars)';
}

/// One line of a conversation.
final class Turn {
  const Turn({
    required this.id,
    required this.speaker,
    required this.text,
    required this.at,
    this.ui,
  });

  /// The router's turn id for an agent turn, and one of our own for a participant's. Stable
  /// while the turn grows, so a list keyed by it keeps the row rather than rebuilding it.
  final String id;
  final Speaker speaker;
  final String text;
  final DateTime at;
  final TurnUI? ui;

  Turn _withText(String text) => Turn(id: id, speaker: speaker, text: text, at: at, ui: ui);

  Turn _withUI(TurnUI ui) => Turn(id: id, speaker: speaker, text: text, at: at, ui: ui);

  @override
  bool operator ==(Object other) =>
      other is Turn &&
      other.id == id &&
      other.speaker == speaker &&
      other.text == text &&
      other.at == at &&
      other.ui == ui;

  @override
  int get hashCode => Object.hash(id, speaker, text, at, ui);

  @override
  String toString() => 'Turn($id, ${speaker.isAgent ? 'agent' : 'participant'}: $text)';
}

/// What the agent is doing.
sealed class ConversationState {
  const ConversationState();

  static const idle = Idle._();
  static const listening = Listening._();
  static const responding = Responding._();
  static const ended = Ended._();
}

/// Waiting to be spoken to.
final class Idle extends ConversationState {
  const Idle._();
}

/// Somebody is talking and being transcribed.
final class Listening extends ConversationState {
  const Listening._();
}

/// The model is answering.
final class Responding extends ConversationState {
  const Responding._();
}

/// Skills are thinking, named so a view can say what about.
final class Working extends ConversationState {
  const Working(this.skills);

  final List<String> skills;

  @override
  bool operator ==(Object other) => other is Working && _same(other.skills, skills);

  @override
  int get hashCode => Object.hashAll(skills);

  @override
  String toString() => 'Working($skills)';
}

/// The conversation is over.
final class Ended extends ConversationState {
  const Ended._();
}

/// A conversation, as the events so far have left it.
///
/// An immutable value with no network in it, which is the whole design: what a stream of
/// frames means for a transcript is decided here, testably, and `AgentSession` is only the
/// socket around it. Folding in an event returns a new conversation, or this one untouched
/// when the event changes nothing, so `identical` is a cheap test for "did anything happen".
final class Conversation {
  const Conversation({this.turns = const [], this.state = ConversationState.idle, this.failure});

  /// Oldest first. The agent's turn in flight is the last entry and grows as deltas arrive.
  final List<Turn> turns;
  final ConversationState state;

  /// What the agent reported going wrong, or null. Reported rather than thrown, because
  /// nobody is awaiting the socket.
  final String? failure;

  /// Whatever the caller typed, shown before the router has confirmed hearing it.
  Conversation said(String text, {DateTime? at}) => _with(
    turns: [
      ...turns,
      Turn(
        id: _localId(),
        speaker: const ParticipantSpeaker(),
        text: text,
        at: at ?? DateTime.now(),
      ),
    ],
  );

  /// The conversation with this state instead.
  Conversation withState(ConversationState state) =>
      state == this.state ? this : _with(state: state);

  /// Folds one event in.
  ///
  /// An event with no bearing on the transcript, and one this SDK has never heard of, both
  /// return this conversation unchanged.
  Conversation apply(AgentEvent event, {DateTime? at}) {
    final now = at ?? DateTime.now();
    switch (event.kind) {
      case AgentEventKind.heard:
        // A text session echoes what was typed, which is already here. A call transcribes
        // what was spoken, which is the first anyone hears of it.
        if (_matchesLastParticipantTurn(event.text)) {
          return failure == null
              ? withState(ConversationState.idle)
              : _with(state: ConversationState.idle, failure: null);
        }
        return _with(
          turns: [
            ...turns,
            Turn(
              id: _localId(),
              speaker: ParticipantSpeaker(event.participant),
              text: event.text,
              at: now,
            ),
          ],
          state: ConversationState.idle,
          failure: null,
        );

      case AgentEventKind.hearing:
        return withState(ConversationState.listening);

      case AgentEventKind.responding:
        return _with(
          turns: [
            ...turns,
            Turn(id: event.turnId, speaker: Speaker.agent, text: '', at: now),
          ],
          state: ConversationState.responding,
          failure: null,
        );

      case AgentEventKind.responseDelta:
        return _write(
          event.turnId,
          now,
          (text) => text + event.text,
          event.text,
        ).withState(ConversationState.responding);

      case AgentEventKind.uiDelta:
        final delta = _text(event['delta']);
        final format = _text(event['format']);
        if (delta.isEmpty || format.isEmpty) {
          return this;
        }
        final version = _optionalText(event['version']);
        final mode = _nonEmptyOr(_text(event['mode']), 'replace');
        final updated = _writeUI(event.turnId, now, (previous) {
          final raw = '${previous?.raw ?? ''}$delta';
          final content = _resolveUIContent(event.turnId, format, mode, raw);
          return TurnUI(
            format: format,
            version: version,
            status: 'streaming',
            content: content,
            raw: raw,
          );
        });
        final nextState = state is Working ? state : ConversationState.responding;
        return failure == null
            ? updated.withState(nextState)
            : updated._with(state: nextState, failure: null);

      case AgentEventKind.uiCompleted:
        final format = _text(event['format']);
        if (format.isEmpty) {
          return this;
        }
        final version = _optionalText(event['version']);
        final mode = _nonEmptyOr(_text(event['mode']), 'replace');
        final explicit = _text(event['content']);
        return _writeUI(event.turnId, now, (previous) {
          final raw = explicit.isNotEmpty ? explicit : (previous?.raw ?? '');
          final content = _resolveUIContent(event.turnId, format, mode, raw);
          return TurnUI(format: format, version: version, status: 'complete', content: content);
        });

      case AgentEventKind.responded:
        // The final text is authoritative: the deltas are what was being written, this is
        // what was said. An empty one adds nothing, which is what a spoken-only turn is.
        final written = event.text.isEmpty
            ? this
            : _write(event.turnId, now, (_) => event.text, event.text);
        return written.withState(ConversationState.idle);

      case AgentEventKind.delegated:
        return _with(state: Working([..._working, _text(event['skill'])]));

      case AgentEventKind.taskSettled || AgentEventKind.taskCancelled:
        // A task settling after the turn finished, or the conversation ended, changes nothing.
        if (state is! Working) {
          return this;
        }
        final left = [..._working]..remove(_text(event['skill']));
        return _with(state: left.isEmpty ? ConversationState.responding : Working(left));

      case AgentEventKind.interrupted:
        return withState(ConversationState.idle);

      case AgentEventKind.error:
        return _with(failure: event.errorText);

      case AgentEventKind.left:
        return withState(ConversationState.ended);

      default:
        return this;
    }
  }

  List<String> get _working => switch (state) {
    Working(:final skills) => skills,
    _ => const [],
  };

  /// What was typed is shown trimmed, so the echo is compared trimmed too.
  bool _matchesLastParticipantTurn(String text) =>
      turns.isNotEmpty && !turns.last.speaker.isAgent && turns.last.text == text.trim();

  /// Changes the agent turn this event belongs to, starting one if the router sent a delta
  /// for a turn we never saw begin.
  Conversation _write(String turnId, DateTime now, String Function(String) change, String fresh) {
    final index = turns.lastIndexWhere((turn) => turn.id == turnId && turn.speaker.isAgent);
    if (index == -1) {
      return _with(
        turns: [
          ...turns,
          Turn(id: turnId, speaker: Speaker.agent, text: fresh, at: now),
        ],
      );
    }
    final changed = [...turns];
    changed[index] = turns[index]._withText(change(turns[index].text));
    return _with(turns: changed);
  }

  Conversation _writeUI(String turnId, DateTime now, TurnUI Function(TurnUI?) change) {
    final index = turns.lastIndexWhere((turn) => turn.id == turnId && turn.speaker.isAgent);
    if (index == -1) {
      return _with(
        turns: [
          ...turns,
          Turn(id: turnId, speaker: Speaker.agent, text: '', at: now, ui: change(null)),
        ],
      );
    }
    final nextUI = change(turns[index].ui);
    if (turns[index].ui == nextUI && turns[index].ui?.raw == nextUI.raw) {
      return this;
    }
    final changed = [...turns];
    changed[index] = turns[index]._withUI(nextUI);
    return _with(turns: changed);
  }

  String _resolveUIContent(String turnId, String format, String mode, String raw) {
    if (mode != 'patch' || format != 'openui') {
      return raw;
    }
    for (var i = turns.length - 1; i >= 0; i--) {
      final turn = turns[i];
      if (turn.speaker.isAgent &&
          turn.id != turnId &&
          turn.ui?.format == 'openui' &&
          (turn.ui?.content.isNotEmpty ?? false)) {
        return mergeOpenUIStatements(turn.ui!.content, raw);
      }
    }
    return raw;
  }

  Conversation _with({List<Turn>? turns, ConversationState? state, Object? failure = _keep}) =>
      Conversation(
        turns: turns ?? this.turns,
        state: state ?? this.state,
        failure: identical(failure, _keep) ? this.failure : failure as String?,
      );

  @override
  bool operator ==(Object other) =>
      other is Conversation &&
      other.state == state &&
      other.failure == failure &&
      _same(other.turns, turns);

  @override
  int get hashCode => Object.hash(Object.hashAll(turns), state, failure);
}

final _openUIIdent = RegExp(r'^\$?[A-Za-z_][A-Za-z0-9_]*$');

/// Merges an OpenUI statement patch onto an earlier OpenUI program, updating matching
/// identifiers in place, removing component identifiers assigned `null`, and appending newly
/// introduced identifiers at the end.
String mergeOpenUIStatements(String base, String patch) {
  if (base.trim().isEmpty) return patch;
  if (patch.trim().isEmpty) return base;

  final order = <String>[];
  final byIdent = <String, String>{};

  for (final raw in _splitOpenUIStatements(base)) {
    final parsed = _parseOpenUIAssignment(raw);
    if (parsed == null) continue;
    if (!byIdent.containsKey(parsed.ident)) {
      order.add(parsed.ident);
    }
    byIdent[parsed.ident] = raw;
  }

  for (final raw in _splitOpenUIStatements(patch)) {
    final parsed = _parseOpenUIAssignment(raw);
    if (parsed == null) continue;
    if (parsed.rhs == 'null' && !parsed.ident.startsWith(r'$')) {
      byIdent.remove(parsed.ident);
      order.remove(parsed.ident);
      continue;
    }
    if (!byIdent.containsKey(parsed.ident)) {
      order.add(parsed.ident);
    }
    byIdent[parsed.ident] = '${parsed.ident} = ${parsed.rhs}';
  }

  return [
    for (final ident in order)
      if (byIdent[ident] case final String stmt) stmt,
  ].join('\n');
}

List<String> _splitOpenUIStatements(String program) {
  final stmts = <String>[];
  final current = StringBuffer();
  var quote = '';
  var escaped = false;
  var depth = 0;

  for (final line in program.split('\n')) {
    if (quote.isEmpty && depth == 0) {
      final trimmed = line.trim();
      if (trimmed.isEmpty || trimmed.startsWith('//') || trimmed.startsWith('#')) {
        continue;
      }
    }
    if (current.isNotEmpty) {
      current.write('\n');
    }
    current.write(line);

    for (var i = 0; i < line.length; i++) {
      final ch = line[i];
      if (escaped) {
        escaped = false;
        continue;
      }
      if (quote.isNotEmpty) {
        if (ch == r'\') {
          escaped = true;
        } else if (ch == quote) {
          quote = '';
        }
        continue;
      }
      if (ch == '/' && i + 1 < line.length && line[i + 1] == '/') {
        break;
      }
      if (ch == '"' || ch == "'") {
        quote = ch;
      } else if (ch == '(' || ch == '[' || ch == '{') {
        depth++;
      } else if ((ch == ')' || ch == ']' || ch == '}') && depth > 0) {
        depth--;
      }
    }

    if (quote.isEmpty && depth == 0) {
      final finished = current.toString().trim();
      if (finished.isNotEmpty) {
        stmts.add(finished);
      }
      current.clear();
    }
  }

  final tail = current.toString().trim();
  if (tail.isNotEmpty) {
    stmts.add(tail);
  }
  return stmts;
}

({String ident, String rhs})? _parseOpenUIAssignment(String stmt) {
  for (var i = 0; i < stmt.length; i++) {
    if (stmt[i] != '=') continue;
    final prev = i > 0 ? stmt[i - 1] : '';
    if (prev == '!' || prev == '<' || prev == '>' || prev == '=') continue;
    if (i + 1 < stmt.length && stmt[i + 1] == '=') {
      i++;
      continue;
    }
    final ident = stmt.substring(0, i).trim();
    if (!_openUIIdent.hasMatch(ident)) return null;
    return (ident: ident, rhs: stmt.substring(i + 1).trim());
  }
  return null;
}

/// Leaves [Conversation.failure] as it is, where null clears it.
const _keep = Object();

final _random = Random();

String _localId() =>
    'local-${DateTime.now().microsecondsSinceEpoch}-${_random.nextInt(0x7fffffff)}';

String _text(Object? value) => value is String ? value : '';

String? _optionalText(Object? value) => value is String && value.isNotEmpty ? value : null;

String _nonEmptyOr(String value, String fallback) => value.isNotEmpty ? value : fallback;

bool _same<T>(List<T> a, List<T> b) {
  if (identical(a, b)) {
    return true;
  }
  if (a.length != b.length) {
    return false;
  }
  // From the end: what streams in changes the last turn.
  for (var i = a.length - 1; i >= 0; i--) {
    if (!identical(a[i], b[i]) && a[i] != b[i]) {
      return false;
    }
  }
  return true;
}
