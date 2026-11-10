import 'package:flutter/painting.dart';
import 'package:flutter_assistant/captions.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

void main() {
  test('Captions reveals the reply at the pace of the voice, whole words only', () {
    final captions = Captions();

    _hear(captions, _reply(), seconds: 0.6);

    // 0.6 s at 15 characters a second is 9 characters: "Your orde", held back to "Your".
    expect(captions.caption, const Caption('Your', isPerson: false));
  });

  test('Captions shows the reply while the agent is heard', () {
    final captions = Captions();

    _hear(captions, _reply(), seconds: 0.6);

    expect(captions.isVisible, isTrue);
  });

  test('Captions reveals nothing of a reply the agent has not started to say', () {
    final captions = Captions();

    captions.update(
      turn: _reply(),
      agentIsAudible: false,
      agentIsAnswering: true,
      now: _tick,
      elapsed: _tick,
    );

    expect(captions.caption, isNull);
  });

  test('Captions starts the pacing over for the next reply', () {
    final captions = Captions();
    _hear(captions, _reply(), seconds: 0.6);

    _hear(
      captions,
      _reply(id: 'next', text: 'Second reply here.'),
      seconds: 0.6,
      from: const Duration(milliseconds: 600),
    );

    // 9 characters again, "Second re", rather than 18 counted from the first reply.
    expect(captions.caption, const Caption('Second', isPerson: false));
  });

  test('Captions shows the rest of the reply once the agent has stopped for good', () {
    final captions = Captions();
    final reply = _reply();
    captions.update(
      turn: reply,
      agentIsAudible: true,
      agentIsAnswering: false,
      now: _tick,
      elapsed: _tick,
    );

    // Quiet for longer than the 1.5 s that ends a reply, with nothing more on its way.
    captions.update(
      turn: reply,
      agentIsAudible: false,
      agentIsAnswering: false,
      now: const Duration(seconds: 2),
      elapsed: const Duration(milliseconds: 1900),
    );

    expect(captions.caption?.text, 'It arrives on Friday.');
  });

  test('Captions puts up what the person said', () {
    final captions = Captions();

    _say(captions, _question(), now: _tick);

    expect(captions.caption, const Caption('Where is my order?', isPerson: true));
  });

  test('Captions takes what the person said down after a moment', () {
    final captions = Captions();
    final question = _question();
    _say(captions, question, now: _tick);

    // Longer than the 2.5 s a caption stays up once nothing new is said.
    _say(captions, question, now: const Duration(seconds: 3));

    expect(captions.isVisible, isFalse);
  });

  test('Captions.sentence shows only the sentence being said', () {
    const text = 'Your order is on its way. It arrives on Friday.';

    expect(Captions.sentence(text, text.length), 'It arrives on Friday.');
  });

  test('Captions.sentence holds back a word until all of it has been said', () {
    const text = 'Your order is on its way. It arrives on Friday.';
    final count = text.indexOf('arrives') + 2;

    expect(Captions.sentence(text, count), 'It');
  });

  test('Captions.sentence starts a sentence after a blank line', () {
    const text = 'Here is the list:\n\nMilk';

    expect(Captions.sentence(text, text.length), 'Milk');
  });

  test('Captions.sentence shows nothing before anything is said', () {
    expect(Captions.sentence('Your order is on its way.', 0), isEmpty);
  });

  test('CaptionView.tail shows a caption that fits as it is', () {
    final shown = CaptionView.tail('It arrives on Friday.', style: _style, maxWidth: 400);

    expect(shown, 'It arrives on Friday.');
  });

  test('CaptionView.tail keeps the end of a caption too long for two lines', () {
    const said = 'Your order left the warehouse this morning and arrives on Friday afternoon.';

    final shown = CaptionView.tail(said, style: _style, maxWidth: 200);

    expect(shown, '…${said.substring(said.length - shown.length + 1)}');
  });

  test('CaptionView.tail shows the last word alone when nothing shorter fits', () {
    final shown = CaptionView.tail(
      'A supercalifragilisticexpialidocious',
      style: _style,
      maxWidth: 10,
    );

    expect(shown, '…supercalifragilisticexpialidocious');
  });
}

/// How often the voice screen moves the captions on, rounded.
const _tick = Duration(milliseconds: 100);

const _style = TextStyle(fontSize: 10);

Turn _reply({
  String id = 'reply',
  String text = 'Your order is on its way. It arrives on Friday.',
}) => Turn(id: id, speaker: Speaker.agent, text: text, at: DateTime(2026));

Turn _question() => Turn(
  id: 'question',
  speaker: const ParticipantSpeaker(),
  text: 'Where is my order?',
  at: DateTime(2026),
);

/// Hears the agent say [reply] for [seconds], a tick at a time, starting [from].
void _hear(
  Captions captions,
  Turn reply, {
  required double seconds,
  Duration from = Duration.zero,
}) {
  var now = from;
  for (var i = 0; i < (seconds * 1000) ~/ _tick.inMilliseconds; i++) {
    now += _tick;
    captions.update(
      turn: reply,
      agentIsAudible: true,
      agentIsAnswering: true,
      now: now,
      elapsed: _tick,
    );
  }
}

/// Moves the captions on to [now] with the person's [question], the agent quiet.
void _say(Captions captions, Turn question, {required Duration now}) => captions.update(
  turn: question,
  agentIsAudible: false,
  agentIsAnswering: false,
  now: now,
  elapsed: _tick,
);
