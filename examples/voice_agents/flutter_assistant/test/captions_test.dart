import 'package:flutter/painting.dart';
import 'package:flutter_assistant/captions.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

void main() {
  group('Captions', () {
    const tick = Duration(milliseconds: 100);
    final reply = Turn(
      id: 'reply',
      speaker: Speaker.agent,
      text: 'Your order is on its way. It arrives on Friday.',
      at: DateTime(2026),
    );

    test('reveals the reply at the pace of the voice, whole words only', () {
      final captions = Captions();
      var now = Duration.zero;
      for (var i = 0; i < 6; i++) {
        now += tick;
        captions.update(
          turn: reply,
          agentIsAudible: true,
          agentIsAnswering: true,
          now: now,
          elapsed: tick,
        );
      }
      // 0.6 s at 15 characters a second is 9 characters: "Your orde", held back to "Your".
      expect(captions.caption, const Caption('Your', isPerson: false));
      expect(captions.isVisible, isTrue);
    });

    test('shows only the sentence being said', () {
      expect(
        Captions.sentence(reply.text, reply.text.length),
        'It arrives on Friday.',
      );
      expect(Captions.sentence(reply.text, 30), 'It');
    });

    test('shows the rest of the reply once the agent has stopped for good', () {
      final captions = Captions();
      captions.update(
        turn: reply,
        agentIsAudible: true,
        agentIsAnswering: false,
        now: tick,
        elapsed: tick,
      );
      captions.update(
        turn: reply,
        agentIsAudible: false,
        agentIsAnswering: false,
        now: const Duration(seconds: 2),
        elapsed: const Duration(milliseconds: 1900),
      );
      expect(captions.caption?.text, 'It arrives on Friday.');
    });

    test("puts up what the person said and takes it down after a moment", () {
      final captions = Captions();
      final question = Turn(
        id: 'question',
        speaker: const ParticipantSpeaker(),
        text: 'Where is my order?',
        at: DateTime(2026),
      );
      captions.update(
        turn: question,
        agentIsAudible: false,
        agentIsAnswering: false,
        now: tick,
        elapsed: tick,
      );
      expect(
        captions.caption,
        const Caption('Where is my order?', isPerson: true),
      );
      expect(captions.isVisible, isTrue);

      captions.update(
        turn: question,
        agentIsAudible: false,
        agentIsAnswering: false,
        now: const Duration(seconds: 3),
        elapsed: const Duration(milliseconds: 2900),
      );
      expect(captions.isVisible, isFalse);
    });
  });

  group('CaptionView', () {
    const style = TextStyle(fontSize: 10);

    test('shows a caption that fits as it is', () {
      expect(
        CaptionView.tail('It arrives on Friday.', style: style, maxWidth: 400),
        'It arrives on Friday.',
      );
    });

    test('keeps the end of a caption too long for two lines', () {
      const said =
          'Your order left the warehouse this morning and arrives on Friday afternoon.';
      final shown = CaptionView.tail(said, style: style, maxWidth: 200);

      expect(shown, startsWith('…'));
      expect(shown, endsWith('Friday afternoon.'));
      expect(said, endsWith(shown.substring(1)));
    });
  });
}
