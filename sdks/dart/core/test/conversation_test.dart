import 'package:test/test.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

/// Frames exactly as `frameOf` in the router writes them, so a change to the wire format on
/// that side fails here rather than in somebody's app.
AgentEvent frame(String json) => AgentEvent.tryParse(json)!;

void main() {
  group('Conversation', () {
    test('a reply arrives one delta at a time', () {
      var conversation = const Conversation();

      conversation = conversation.apply(
        frame('{"type":"responding","turn_id":"t1","prompt":"hi"}'),
      );
      expect(conversation.state, ConversationState.responding);
      expect(conversation.turns, hasLength(1));

      conversation = conversation
          .apply(frame('{"type":"response_delta","turn_id":"t1","text":"Hel"}'))
          .apply(frame('{"type":"response_delta","turn_id":"t1","text":"lo"}'));
      expect(conversation.turns.last.text, 'Hello');

      conversation = conversation.apply(
        frame(
          '{"pending_work":false,"type":"responded","turn_id":"t1","text":"Hello there.",'
          '"time_to_first_token_ms":90}',
        ),
      );
      expect(conversation.turns, hasLength(1));
      expect(conversation.turns.last.text, 'Hello there.');
      expect(conversation.turns.last.speaker.isAgent, isTrue);
      expect(conversation.state, ConversationState.idle);
    });

    test('keeps the turn id while the turn grows, so a list keyed by it keeps the row', () {
      final conversation = const Conversation()
          .apply(frame('{"type":"responding","turn_id":"t1"}'))
          .apply(frame('{"type":"response_delta","turn_id":"t1","text":"a"}'))
          .apply(frame('{"type":"response_delta","turn_id":"t1","text":"b"}'));

      expect(conversation.turns.map((turn) => turn.id), ['t1']);
    });

    test('a delta for a turn nobody saw begin still lands', () {
      final conversation = const Conversation().apply(
        frame('{"type":"response_delta","turn_id":"t9","text":"mid"}'),
      );

      expect(conversation.turns.map((turn) => turn.text), ['mid']);
      expect(conversation.turns.last.speaker.isAgent, isTrue);
    });

    test('a spoken turn with no final text keeps what was streamed', () {
      final conversation = const Conversation()
          .apply(frame('{"type":"responding","turn_id":"t1"}'))
          .apply(frame('{"type":"response_delta","turn_id":"t1","text":"Sure."}'))
          .apply(frame('{"type":"responded","turn_id":"t1","text":""}'));

      expect(conversation.turns.map((turn) => turn.text), ['Sure.']);
    });

    test('what was heard on a call becomes a participant turn', () {
      final conversation = const Conversation().apply(
        frame(
          '{"type":"heard","participant":{"id":"p1","user_id":"u1","name":"Alice"},'
          '"text":"what are your hours","language":"en"}',
        ),
      );

      expect(conversation.turns, hasLength(1));
      expect(conversation.turns.single.text, 'what are your hours');
      final speaker = conversation.turns.single.speaker;
      expect(speaker, isA<ParticipantSpeaker>());
      expect((speaker as ParticipantSpeaker).participant?.display, 'Alice');
    });

    test('typing something is not shown twice when the router echoes it', () {
      final conversation = const Conversation()
          .said('what are your hours')
          .apply(frame('{"type":"heard","participant":{},"text":"what are your hours"}'));

      expect(conversation.turns, hasLength(1));
    });

    test('a delegated skill is named while it runs', () {
      var conversation = const Conversation().apply(
        frame('{"type":"delegated","task_id":"k1","skill":"lookup_order"}'),
      );
      expect(conversation.state, const Working(['lookup_order']));

      conversation = conversation.apply(
        frame(
          '{"type":"task_settled","evidence":null,"task_id":"k1","skill":"lookup_order",'
          '"text":"done","question":"","elapsed_ms":12,"error":""}',
        ),
      );
      expect(conversation.state, ConversationState.responding);
    });

    test('two skills at once both show until both settle', () {
      var conversation = const Conversation()
          .apply(frame('{"type":"delegated","task_id":"k1","skill":"think"}'))
          .apply(frame('{"type":"delegated","task_id":"k2","skill":"recall"}'));
      expect(conversation.state, const Working(['think', 'recall']));

      conversation = conversation.apply(
        frame('{"type":"task_settled","task_id":"k1","skill":"think"}'),
      );
      expect(conversation.state, const Working(['recall']));
    });

    test('the conversation ends when the agent leaves', () {
      final conversation = const Conversation().apply(
        frame('{"type":"left","at":"2026-09-02T17:05:00Z"}'),
      );

      expect(conversation.state, ConversationState.ended);
    });

    test('a reported failure is kept rather than thrown', () {
      final conversation = const Conversation().apply(
        frame('{"type":"error","context":"tts","error":"the voice is unknown"}'),
      );

      expect(conversation.failure, 'the voice is unknown');
    });

    test('an event this SDK has never heard of changes nothing', () {
      final before = const Conversation().apply(frame('{"type":"responding","turn_id":"t1"}'));

      final after = before.apply(frame('{"type":"astonished","turn_id":"t1","degree":11}'));

      expect(identical(after, before), isTrue);
    });
  });

  group('AgentEvent', () {
    test('keeps an unknown event whole, with no kind and every field', () {
      final event = frame('{"type":"astonished","degree":11,"nested":{"a":[1,2]}}');

      expect(event.kind, isNull);
      expect(event.type, 'astonished');
      expect(event['degree'], 11);
      expect(event['nested'], {
        'a': [1, 2],
      });
    });

    test('reads nothing out of a frame that is not an object with a type', () {
      expect(AgentEvent.tryParse('not json'), isNull);
      expect(AgentEvent.tryParse('[1,2]'), isNull);
      expect(AgentEvent.tryParse('{"text":"no type"}'), isNull);
    });

    test('a tool call decodes its arguments, which arrive as a JSON string', () {
      final call = frame(
        '{"type":"tool_call","id":"c1","name":"lookup_order",'
        r'"arguments":"{\"order_id\":\"A-1042\"}","command_id":"","turn_id":"t1"}',
      ).toolCall!;

      expect(call.name, 'lookup_order');
      expect(call.argumentValues, {'order_id': 'A-1042'});
      expect(call.turnId, 't1');
    });

    test('arguments that are not an object decode to nothing rather than throwing', () {
      const call = ToolCall(id: 'c1', name: 'x', arguments: 'not json');

      expect(call.argumentValues, isEmpty);
    });
  });
}
