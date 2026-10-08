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

/// One line of a conversation.
final class Turn {
  const Turn({required this.id, required this.speaker, required this.text, required this.at});

  /// The router's turn id for an agent turn, and one of our own for a participant's. Stable
  /// while the turn grows, so a list keyed by it keeps the row rather than rebuilding it.
  final String id;
  final Speaker speaker;
  final String text;
  final DateTime at;

  Turn _withText(String text) => Turn(id: id, speaker: speaker, text: text, at: at);

  @override
  bool operator ==(Object other) =>
      other is Turn &&
      other.id == id &&
      other.speaker == speaker &&
      other.text == text &&
      other.at == at;

  @override
  int get hashCode => Object.hash(id, speaker, text, at);

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
          return withState(ConversationState.idle);
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
        );

      case AgentEventKind.responseDelta:
        return _write(
          event.turnId,
          now,
          (text) => text + event.text,
          event.text,
        ).withState(ConversationState.responding);

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

  bool _matchesLastParticipantTurn(String text) =>
      turns.isNotEmpty && !turns.last.speaker.isAgent && turns.last.text == text;

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

  Conversation _with({List<Turn>? turns, ConversationState? state, String? failure}) =>
      Conversation(
        turns: turns ?? this.turns,
        state: state ?? this.state,
        failure: failure ?? this.failure,
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

final _random = Random();

String _localId() =>
    'local-${DateTime.now().microsecondsSinceEpoch}-${_random.nextInt(0x7fffffff)}';

String _text(Object? value) => value is String ? value : '';

bool _same<T>(List<T> a, List<T> b) {
  if (a.length != b.length) {
    return false;
  }
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}
