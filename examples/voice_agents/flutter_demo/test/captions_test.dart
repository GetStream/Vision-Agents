import 'package:flutter/material.dart';
import 'package:flutter_demo/captions.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

void main() {
  test(
    'Captions reveals the reply at the pace of the voice, whole words only',
    () {
      final captions = Captions();
      final reply = _reply();
      var now = Duration.zero;
      for (var i = 0; i < 6; i += 1) {
        now += _tick;
        captions.update(
          turn: reply,
          agentIsAudible: true,
          agentIsAnswering: true,
          now: now,
          elapsed: _tick,
        );
      }

      // 0.6 s at 15 characters a second is 9 characters: "Your orde", held back to "Your".
      expect(captions.caption, const Caption('Your', isPerson: false));
      expect(captions.isVisible, isTrue);
    },
  );

  test('Captions.sentence shows only the sentence being said', () {
    final text = _reply().text;

    expect(Captions.sentence(text, text.length), 'It arrives on Friday.');
  });

  test('Captions.sentence holds back a word until all of it has been said', () {
    final text = _reply().text;
    // The reply up to "It arr", partway into "arrives".
    final count = text.indexOf('arrives') + 3;

    expect(Captions.sentence(text, count), 'It');
  });

  test(
    'Captions keeps pacing the reply through a pause shorter than finished',
    () {
      final captions = Captions()
        ..update(
          turn: _reply(),
          agentIsAudible: true,
          agentIsAnswering: false,
          now: _tick,
          elapsed: _tick,
        );
      final pause = Captions.finished - _tick;

      captions.update(
        turn: _reply(),
        agentIsAudible: false,
        agentIsAnswering: false,
        now: _tick + pause,
        elapsed: pause,
      );

      // 0.1 s heard is 1 character, short of a whole word.
      expect(captions.caption, isNull);
    },
  );

  test(
    'Captions shows the rest of the reply once the agent has stopped for good',
    () {
      final captions = Captions()
        ..update(
          turn: _reply(),
          agentIsAudible: true,
          agentIsAnswering: false,
          now: _tick,
          elapsed: _tick,
        );
      final silence = Captions.finished + _tick;

      captions.update(
        turn: _reply(),
        agentIsAudible: false,
        agentIsAnswering: false,
        now: _tick + silence,
        elapsed: silence,
      );

      expect(captions.caption?.text, 'It arrives on Friday.');
    },
  );

  test(
    'Captions puts up what the person said as soon as it is transcribed',
    () {
      final captions = Captions()
        ..update(
          turn: _question(),
          agentIsAudible: false,
          agentIsAnswering: false,
          now: _tick,
          elapsed: _tick,
        );

      expect(
        captions.caption,
        const Caption('Where is my order?', isPerson: true),
      );
      expect(captions.isVisible, isTrue);
    },
  );

  test('Captions takes a caption down once it has lingered', () {
    final captions = Captions()
      ..update(
        turn: _question(),
        agentIsAudible: false,
        agentIsAnswering: false,
        now: _tick,
        elapsed: _tick,
      );

    captions.update(
      turn: _question(),
      agentIsAudible: false,
      agentIsAnswering: false,
      now: _tick + Captions.linger,
      elapsed: Captions.linger,
    );

    expect(captions.isVisible, isFalse);
  });

  testWidgets(
    'CaptionView keeps the newest words of a caption too long for two lines',
    (tester) async {
      const caption = Caption(
        'The refund for your two wool throws goes back to the card ending 4242 and shows '
        'within five working days, which is the last thing said',
        isPerson: false,
      );

      await tester.pumpWidget(_captionApp(caption));

      final shown = tester.widget<Text>(find.byType(Text)).data!;
      expect(shown, startsWith('…'));
      expect(shown, endsWith('the last thing said'));
      expect(shown, isNot(contains('The refund')));
    },
  );

  testWidgets('CaptionView shows a caption that fits in full', (tester) async {
    const caption = Caption('Your order is on its way.', isPerson: false);

    await tester.pumpWidget(_captionApp(caption));

    expect(find.text('Your order is on its way.'), findsOneWidget);
  });
}

const _tick = Duration(milliseconds: 100);

Turn _reply() => Turn(
  id: 'reply',
  speaker: Speaker.agent,
  text: 'Your order is on its way. It arrives on Friday.',
  at: DateTime(2026),
);

Turn _question() => Turn(
  id: 'question',
  speaker: const ParticipantSpeaker(),
  text: 'Where is my order?',
  at: DateTime(2026),
);

// The test font draws every character a square as wide as the font size, so at 16 a line
// across this width holds 33 characters.
Widget _captionApp(Caption caption) => MaterialApp(
  home: Center(
    child: SizedBox(
      width: 600.0,
      child: CaptionView(caption: caption, isVisible: true),
    ),
  ),
);
