import 'package:flutter_demo/talking.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('WhoIsTalking hears the agent while it speaks, at its level', () {
    final talking = WhoIsTalking();

    final heard = talking.hear(
      agentIsSpeaking: true,
      agentLevel: 0.6,
      personIsSpeaking: false,
      personLevel: 0.0,
      now: Duration.zero,
    );

    expect(heard, (isPerson: false, level: 0.6));
  });

  test('WhoIsTalking hears the agent over the person when both are heard', () {
    final talking = WhoIsTalking();

    final heard = _hearPersonFor(
      talking,
      WhoIsTalking.onset * 2,
      agentIsSpeaking: true,
    );

    expect(heard?.isPerson, isFalse);
  });

  test(
    'WhoIsTalking ignores the person until they have talked for the onset',
    () {
      final talking = WhoIsTalking();

      final heard = _hearPersonFor(talking, WhoIsTalking.onset - _frame);

      expect(heard, isNull);
    },
  );

  test('WhoIsTalking hears the person once they have talked for the onset', () {
    final talking = WhoIsTalking();

    final heard = _hearPersonFor(talking, WhoIsTalking.onset);

    expect(heard, (isPerson: true, level: 0.4));
  });

  test("WhoIsTalking ignores the microphone during the agent's echo tail", () {
    final talking = WhoIsTalking()
      ..hear(
        agentIsSpeaking: true,
        agentLevel: 0.6,
        personIsSpeaking: true,
        personLevel: 0.4,
        now: Duration.zero,
      );

    final heard = talking.hear(
      agentIsSpeaking: false,
      agentLevel: 0.0,
      personIsSpeaking: true,
      personLevel: 0.4,
      now: WhoIsTalking.echoTail,
    );

    expect(heard, isNull);
  });

  test("WhoIsTalking hears the person once the agent's echo tail is over", () {
    final talking = WhoIsTalking()
      ..hear(
        agentIsSpeaking: true,
        agentLevel: 0.6,
        personIsSpeaking: true,
        personLevel: 0.4,
        now: Duration.zero,
      );

    final heard = talking.hear(
      agentIsSpeaking: false,
      agentLevel: 0.0,
      personIsSpeaking: true,
      personLevel: 0.4,
      now: WhoIsTalking.echoTail + _frame,
    );

    expect(heard, (isPerson: true, level: 0.4));
  });

  test('WhoIsTalking starts the onset over when the person stops', () {
    final talking = WhoIsTalking();
    _hearPersonFor(talking, WhoIsTalking.onset - _frame);
    talking.hear(
      agentIsSpeaking: false,
      agentLevel: 0.0,
      personIsSpeaking: false,
      personLevel: 0.0,
      now: WhoIsTalking.onset,
    );

    final heard = talking.hear(
      agentIsSpeaking: false,
      agentLevel: 0.0,
      personIsSpeaking: true,
      personLevel: 0.4,
      now: WhoIsTalking.onset + _frame,
    );

    expect(heard, isNull);
  });
}

const _frame = Duration(milliseconds: 33);

/// Hears the person talking at 0.4 from zero until [duration], and returns what is heard
/// then.
Heard? _hearPersonFor(
  WhoIsTalking talking,
  Duration duration, {
  bool agentIsSpeaking = false,
}) {
  for (var now = Duration.zero; now < duration; now += _frame) {
    talking.hear(
      agentIsSpeaking: agentIsSpeaking,
      agentLevel: 0.6,
      personIsSpeaking: true,
      personLevel: 0.4,
      now: now,
    );
  }
  return talking.hear(
    agentIsSpeaking: agentIsSpeaking,
    agentLevel: 0.6,
    personIsSpeaking: true,
    personLevel: 0.4,
    now: duration,
  );
}
