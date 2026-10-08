import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:stream_chat_flutter_core/stream_chat_flutter_core.dart' as chat;
import 'package:vision_agents_core/vision_agents_core.dart';
import 'package:vision_agents_ui/vision_agents_ui.dart';

Widget _app(Widget child) => MaterialApp(home: Scaffold(body: child));

Turn _turn(String id, String text, {Speaker speaker = Speaker.agent}) =>
    Turn(id: id, speaker: speaker, text: text, at: DateTime.utc(2026, 9, 24));

void main() {
  group('TranscriptView', () {
    testWidgets('draws every turn, the newest lowest', (tester) async {
      await tester.pumpWidget(
        _app(
          TranscriptView(
            turns: [
              _turn('1', 'Hello', speaker: const ParticipantSpeaker()),
              _turn('2', 'Hi, how can I help?'),
            ],
          ),
        ),
      );

      final question = tester.getTopLeft(find.text('Hello')).dy;
      final answer = tester.getTopLeft(find.text('Hi, how can I help?')).dy;
      expect(answer, greaterThan(question));
    });

    testWidgets('draws each turn with the bubble it is given', (tester) async {
      await tester.pumpWidget(
        _app(
          TranscriptView(
            turns: [_turn('1', 'Hello')],
            bubble: (context, turn) => Text('> ${turn.text}'),
          ),
        ),
      );

      expect(find.text('> Hello'), findsOneWidget);
      expect(find.byType(TurnBubble), findsNothing);
    });
  });

  group('TurnBubble', () {
    testWidgets('names the person speaking, and not the agent', (tester) async {
      await tester.pumpWidget(
        _app(
          Column(
            children: [
              TurnBubble(
                turn: _turn(
                  '1',
                  'Where is my order?',
                  speaker: const ParticipantSpeaker(Participant(userId: 'u1', name: 'Ada')),
                ),
              ),
              TurnBubble(turn: _turn('2', 'On its way.')),
            ],
          ),
        ),
      );

      expect(find.text('Ada'), findsOneWidget);
      expect(find.text('On its way.'), findsOneWidget);
    });
  });

  group('Composer', () {
    testWidgets('sends what was typed and empties itself', (tester) async {
      final sent = <String>[];
      await tester.pumpWidget(_app(Composer(send: (text) async => sent.add(text))));

      await tester.enterText(find.byType(TextField), 'What are your hours?');
      await tester.pump();
      await tester.tap(find.byTooltip('Send'));
      await tester.pump();

      expect(sent, ['What are your hours?']);
      expect(find.text('What are your hours?'), findsNothing);
    });

    testWidgets('sends nothing that is only blank', (tester) async {
      final sent = <String>[];
      await tester.pumpWidget(_app(Composer(send: (text) async => sent.add(text))));

      await tester.enterText(find.byType(TextField), '   ');
      await tester.pump();
      await tester.tap(find.byTooltip('Send'));
      await tester.testTextInput.receiveAction(TextInputAction.send);
      await tester.pump();

      expect(sent, isEmpty);
    });

    testWidgets('takes no text while disabled', (tester) async {
      await tester.pumpWidget(_app(Composer(enabled: false, send: (_) async {})));

      expect(tester.widget<TextField>(find.byType(TextField)).enabled, isFalse);
    });
  });

  group('AgentStatusView', () {
    testWidgets('says what the agent is doing', (tester) async {
      await tester.pumpWidget(_app(const AgentStatusView(state: Working(['search']))));

      expect(find.text('thinking (search)'), findsOneWidget);
      expect(find.byType(CircularProgressIndicator), findsOneWidget);
    });

    test('labels each state', () {
      expect(AgentStatusView.label(ConversationState.idle), 'ready');
      expect(AgentStatusView.label(ConversationState.listening), 'listening');
      expect(AgentStatusView.label(ConversationState.responding), 'answering');
      expect(AgentStatusView.label(const Working([])), 'thinking');
      expect(AgentStatusView.label(ConversationState.ended), 'the conversation ended');
    });
  });

  group('LiveValueBuilder', () {
    testWidgets('draws the current value, then each change', (tester) async {
      final count = LiveValueController(1);
      addTearDown(count.close);
      await tester.pumpWidget(
        _app(LiveValueBuilder(value: count, builder: (context, value) => Text('count $value'))),
      );
      expect(find.text('count 1'), findsOneWidget);

      count.value = 2;
      await tester.pump();

      expect(find.text('count 2'), findsOneWidget);
    });
  });

  group('LiveValueNotifier', () {
    testWidgets('follows the live value until disposed', (tester) async {
      final count = LiveValueController(1);
      addTearDown(count.close);
      final notifier = LiveValueNotifier(count);

      count.value = 2;
      await tester.pump();
      expect(notifier.value, 2);

      notifier.dispose();
      count.value = 3;
      await tester.pump();
      expect(notifier.value, 2);
    });
  });

  group('turnsOfMessages', () {
    test("tells the agent's turns by the role the router wrote", () {
      final turns = turnsOfMessages([
        chat.Message(
          id: 'm1',
          text: 'Where is my order?',
          user: chat.User(id: 'ada', name: 'Ada'),
          extraData: const {
            'support_message': {'role': 'user', 'text': 'Where is my order?'},
          },
        ),
        chat.Message(
          id: 'm2',
          text: 'On its way.',
          user: chat.User(id: 'support-agent'),
          extraData: const {
            'support_message': {'role': 'assistant', 'text': 'On its way.'},
          },
        ),
      ]);

      expect(turns.map((turn) => turn.speaker.isAgent), [false, true]);
      expect(turns.map((turn) => turn.text), ['Where is my order?', 'On its way.']);
      expect((turns.first.speaker as ParticipantSpeaker).participant?.display, 'Ada');
    });

    test('tells a turn without a role by who sent it', () {
      final turns = turnsOfMessages([
        chat.Message(
          id: 'm1',
          text: 'Spoken answer',
          user: chat.User(id: 'agent-1'),
        ),
        chat.Message(
          id: 'm2',
          text: 'Spoken question',
          user: chat.User(id: 'ada'),
        ),
      ], agentUserId: 'agent-1');

      expect(turns.map((turn) => turn.speaker.isAgent), [true, false]);
    });

    test('leaves out deleted turns and turns not written yet', () {
      final turns = turnsOfMessages([
        chat.Message(id: 'm1', text: 'gone', type: chat.MessageType.deleted),
        chat.Message(
          id: 'm2',
          extraData: const {
            'support_message': {'role': 'assistant', 'text': '', 'state': 'thinking'},
          },
        ),
        chat.Message(id: 'm3', text: 'kept'),
      ]);

      expect(turns.map((turn) => turn.id), ['m3']);
    });
  });

  group('conversationChannel', () {
    test('opens the channel a session names, and none for one without', () {
      final client = chat.StreamChatClient('key');
      addTearDown(client.dispose);
      final kept = Session(
        id: 's1',
        createdAt: DateTime.utc(2026),
        conversationId: 'agent:support-123',
      );

      final channel = conversationChannel(client, kept);

      expect(channel?.type, 'agent');
      expect(channel?.id, 'support-123');
      expect(conversationChannel(client, Session(id: 's2', createdAt: DateTime.utc(2026))), isNull);
    });
  });
}
