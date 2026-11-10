import 'package:flutter_demo/voice_view.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

void main() {
  test('voicePresence is idle with no word before a call is started', () {
    expect(
      voicePresence(
        isStarting: false,
        session: null,
        isJoined: false,
        heard: null,
      ),
      (state: AIVoiceAgentState.idle, status: ' '),
    );
  });

  test('voicePresence is connecting while a call starts and joins', () {
    const connecting = (
      state: AIVoiceAgentState.connecting,
      status: 'Connecting',
    );
    expect(
      voicePresence(
        isStarting: true,
        session: null,
        isJoined: false,
        heard: null,
      ),
      connecting,
    );
    expect(
      voicePresence(
        isStarting: false,
        session: ConversationState.idle,
        isJoined: false,
        heard: null,
      ),
      connecting,
    );
  });

  test(
    'voicePresence follows whoever is heard before what the agent is doing',
    () {
      expect(
        voicePresence(
          isStarting: false,
          session: ConversationState.responding,
          isJoined: true,
          heard: (isPerson: true, level: 0.4),
        ),
        (state: AIVoiceAgentState.listening, status: 'Listening'),
      );
      expect(
        voicePresence(
          isStarting: false,
          session: ConversationState.idle,
          isJoined: true,
          heard: (isPerson: false, level: 0.4),
        ),
        (state: AIVoiceAgentState.speaking, status: 'Speaking'),
      );
    },
  );

  test('voicePresence says what the agent is doing while nobody is heard', () {
    expect(
      voicePresence(
        isStarting: false,
        session: ConversationState.responding,
        isJoined: true,
        heard: null,
      ),
      (state: AIVoiceAgentState.thinking, status: 'Thinking'),
    );
    expect(
      voicePresence(
        isStarting: false,
        session: ConversationState.idle,
        isJoined: true,
        heard: null,
      ),
      (state: AIVoiceAgentState.idle, status: 'Ready'),
    );
  });
}
