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

  /// The UI to show. While OpenUI streams, that is its finished statements only: the one no
  /// newline has ended yet is held back, so a card grows a statement at a time rather than
  /// reflowing with every delta (a gauge reading 8 before 86). A turn has no UI until one is
  /// finished.
  final String content;

  /// Unmerged turn-local source accumulated before patch merging. While OpenUI streams, it can
  /// lag behind what was written: a delta that ends no line is kept, but leaves this as it was
  /// until the next one that does.
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

  /// Compares what shows, not [raw]: the text written so far is kept by [Conversation], which
  /// counts it, so no delta is lost to a holder that skips equal values.
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
  /// A conversation built here has no UI being written; derive from an existing one
  /// ([withoutTurn], [withState]) to keep it.
  const Conversation({this.turns = const [], this.state = ConversationState.idle, this.failure})
    : _uiWritten = const {};

  const Conversation._(this.turns, this.state, this.failure, this._uiWritten);

  /// Oldest first. The agent's turn in flight is the last entry and grows as deltas arrive.
  final List<Turn> turns;
  final ConversationState state;

  /// What the agent reported going wrong, or null. Reported rather than thrown, because
  /// nobody is awaiting the socket.
  final String? failure;

  /// What each turn's UI block has written so far, unmerged, until the turn's UI is settled.
  final Map<String, _PendingUI> _uiWritten;

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

  /// The conversation without the turn [id], such as a question the router refused. A card the
  /// agent is still writing keeps what it has written so far.
  Conversation withoutTurn(String id) => _with(
    turns: [
      for (final turn in turns)
        if (turn.id != id) turn,
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
          uiWritten: _uiWritten.containsKey(event.turnId) ? _without(event.turnId) : null,
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
        final pending = (
          format: format,
          version: _optionalText(event['version']),
          mode: _nonEmptyOr(_text(event['mode']), 'replace'),
          raw: '${_uiWritten[event.turnId]?.raw ?? ''}$delta',
        );
        final written = _with(uiWritten: {..._uiWritten, event.turnId: pending});
        // Only the line still being written grew, so no statement finished: nothing shown changes.
        final updated = format == 'openui' && !delta.contains('\n')
            ? written
            : written._showUI(event.turnId, now, pending, _Keep.finished);
        final nextState = state is Working ? state : ConversationState.responding;
        return failure == null
            ? updated.withState(nextState)
            : updated._with(state: nextState, failure: null);

      case AgentEventKind.uiCompleted:
        final format = _text(event['format']);
        if (format.isEmpty) {
          return this;
        }
        final explicit = _text(event['content']);
        final pending = (
          format: format,
          version: _optionalText(event['version']),
          mode: _nonEmptyOr(_text(event['mode']), 'replace'),
          raw: explicit.isNotEmpty ? explicit : (_uiWritten[event.turnId]?.raw ?? ''),
        );
        final settled = _uiWritten.containsKey(event.turnId)
            ? _with(uiWritten: _without(event.turnId))
            : this;
        return settled._showUI(event.turnId, now, pending, _Keep.all);

      case AgentEventKind.responded:
        // The final text is authoritative: the deltas are what was being written, this is
        // what was said. An empty one adds nothing, which is what a spoken-only turn is.
        final settled = _settleUI(event.turnId, now);
        final written = event.text.isEmpty
            ? settled
            : settled._write(event.turnId, now, (_) => event.text, event.text);
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
        return _settleUI(event.turnId, now).withState(ConversationState.idle);

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

  /// The turn ended with no `ui_completed`, as an interrupted reply does: the router dropped its
  /// UI block with the turn. What was written shows, but for a statement it ended in the middle
  /// of, and the UI is complete, since nothing more of it will come.
  Conversation _settleUI(String turnId, DateTime now) {
    final pending = _uiWritten[turnId];
    if (pending == null) return this;
    return _with(uiWritten: _without(turnId))._showUI(turnId, now, pending, _Keep.closed);
  }

  /// Shows [pending] as the UI of the turn [turnId]: streaming while only its finished
  /// statements count, complete otherwise. A UI with nothing to show yet sets none.
  Conversation _showUI(String turnId, DateTime now, _PendingUI pending, _Keep keep) {
    final content = _resolveUIContent(turnId, pending, keep);
    if (content == null) return this;
    final streaming = keep == _Keep.finished;
    return _writeUI(
      turnId,
      now,
      TurnUI(
        format: pending.format,
        version: pending.version,
        status: streaming ? 'streaming' : 'complete',
        content: content,
        raw: streaming ? pending.raw : '',
      ),
    );
  }

  /// Sets the agent turn [turnId]'s UI, starting the turn if this never saw it begin.
  Conversation _writeUI(String turnId, DateTime now, TurnUI nextUI) {
    final index = turns.lastIndexWhere((turn) => turn.id == turnId && turn.speaker.isAgent);
    if (index == -1) {
      return _with(
        turns: [
          ...turns,
          Turn(id: turnId, speaker: Speaker.agent, text: '', at: now, ui: nextUI),
        ],
      );
    }
    if (turns[index].ui == nextUI) {
      return this;
    }
    final changed = [...turns];
    changed[index] = turns[index]._withUI(nextUI);
    return _with(turns: changed);
  }

  /// A turn's UI as it shows: what was written, or a patch merged onto the UI before it. Of
  /// OpenUI, only the statements [keep] says count. Null when there is nothing to show yet, and
  /// for a patch with no UI before it here, such as after a reload: Stream Chat's copy, which
  /// the router merged, is the one that shows.
  String? _resolveUIContent(String turnId, _PendingUI pending, _Keep keep) {
    if (pending.format != 'openui') {
      return pending.raw;
    }
    final written = keep == _Keep.all
        ? pending.raw
        : _splitOpenUIStatements(pending.raw, keep: keep).join('\n');
    if (keep != _Keep.all && written.isEmpty) {
      return null;
    }
    if (pending.mode != 'patch') {
      return written;
    }
    for (var i = turns.length - 1; i >= 0; i--) {
      final turn = turns[i];
      if (turn.speaker.isAgent &&
          turn.id != turnId &&
          turn.ui?.format == 'openui' &&
          (turn.ui?.content.isNotEmpty ?? false)) {
        return mergeOpenUIStatements(turn.ui!.content, written);
      }
    }
    return null;
  }

  Map<String, _PendingUI> _without(String turnId) => {..._uiWritten}..remove(turnId);

  Conversation _with({
    List<Turn>? turns,
    ConversationState? state,
    Object? failure = _keep,
    Map<String, _PendingUI>? uiWritten,
  }) => Conversation._(
    turns ?? this.turns,
    state ?? this.state,
    identical(failure, _keep) ? this.failure : failure as String?,
    uiWritten ?? _uiWritten,
  );

  /// The UI written so far counts, or a delta that shows nothing new would be lost to a holder
  /// that skips equal values, and the next one would land on stale text.
  @override
  bool operator ==(Object other) =>
      other is Conversation &&
      other.state == state &&
      other.failure == failure &&
      _same(other.turns, turns) &&
      _sameWritten(other._uiWritten, _uiWritten);

  @override
  int get hashCode => Object.hash(
    Object.hashAll(turns),
    state,
    failure,
    Object.hashAllUnordered(_uiWritten.values),
  );
}

/// What a turn's UI was written to.
typedef _PendingUI = ({String format, String? version, String mode, String raw});

/// Which statements of an OpenUI program count: [all] of them; those [closed], leaving out one a
/// bracket or a string left open; or those [finished], also leaving out the line no newline has
/// ended, which is still being written.
enum _Keep { all, closed, finished }

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

/// A program's statements: each line, or the lines a bracket or a string left open joined to the
/// one that closes it. Blank and comment lines are left out, and so is what [keep] leaves out.
List<String> _splitOpenUIStatements(String program, {_Keep keep = _Keep.all}) {
  final stmts = <String>[];
  final current = StringBuffer();
  var quote = '';
  var escaped = false;
  var depth = 0;

  final lines = program.split('\n');
  if (keep == _Keep.finished) {
    // What follows the last newline is still being written.
    lines.removeLast();
  }
  for (final line in lines) {
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
  if (tail.isNotEmpty && keep == _Keep.all) {
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

bool _sameWritten(Map<String, _PendingUI> a, Map<String, _PendingUI> b) {
  if (identical(a, b)) {
    return true;
  }
  if (a.length != b.length) {
    return false;
  }
  for (final MapEntry(:key, :value) in a.entries) {
    if (b[key] != value) {
      return false;
    }
  }
  return true;
}

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
