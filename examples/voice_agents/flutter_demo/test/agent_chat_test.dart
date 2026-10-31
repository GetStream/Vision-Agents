import 'package:flutter_demo/agent_chat.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:stream_chat/stream_chat.dart' as stream;
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';
import 'package:vision_agents_chat/vision_agents_chat.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

void main() {
  test('describeWork names each skill in words', () {
    expect(
      describeWork(['refund_decision', 'order_status']),
      'Working on refund decision and order status',
    );
  });

  test('describeWork says nothing when no skill is named', () {
    expect(describeWork(const []), isNull);
  });

  test(
    'isThinkingAbout is false when nothing was asked and the session is idle',
    () {
      expect(
        isThinkingAbout(
          state: ConversationState.idle,
          messages: [
            _question('q1'),
            _reply('a1', text: 'It is on its way.'),
          ],
          askedAfter: null,
        ),
        isFalse,
      );
    },
  );

  test('isThinkingAbout is true as soon as the first question is asked', () {
    expect(
      isThinkingAbout(
        state: ConversationState.idle,
        messages: const [],
        askedAfter: '',
      ),
      isTrue,
    );
  });

  test('isThinkingAbout is true before the question reaches the channel', () {
    expect(
      isThinkingAbout(
        state: ConversationState.idle,
        messages: [_reply('a1', text: 'Hello.')],
        askedAfter: 'a1',
      ),
      isTrue,
    );
  });

  test(
    'isThinkingAbout is true while the reply is generating with nothing to show',
    () {
      expect(
        isThinkingAbout(
          state: ConversationState.idle,
          messages: [_question('q1'), _reply('a1', generating: true)],
          askedAfter: '',
        ),
        isTrue,
      );
    },
  );

  test('isThinkingAbout is false once the reply shows text', () {
    expect(
      isThinkingAbout(
        state: ConversationState.responding,
        messages: [
          _question('q1'),
          _reply('a1', text: 'Your order', generating: true),
        ],
        askedAfter: '',
      ),
      isFalse,
    );
  });

  test('isThinkingAbout is false once the reply shows a step', () {
    expect(
      isThinkingAbout(
        state: ConversationState.responding,
        messages: [
          _question('q1'),
          _reply('a1', generating: true, attachments: [_toolCall('call_1')]),
        ],
        askedAfter: '',
      ),
      isFalse,
    );
  });

  test(
    'isThinkingAbout is false while skills say what they are working on',
    () {
      expect(
        isThinkingAbout(
          state: const Working(['refund_decision']),
          messages: [_question('q1')],
          askedAfter: '',
        ),
        isFalse,
      );
    },
  );

  test(
    'isThinkingAbout is true while the model responds to a question asked elsewhere',
    () {
      expect(
        isThinkingAbout(
          state: ConversationState.responding,
          messages: [_question('q1')],
          askedAfter: null,
        ),
        isTrue,
      );
    },
  );

  test(
    'AgentReply.isEmptyReply is true for a reply with no text and no steps',
    () {
      expect(_reply('a1', generating: true).isEmptyReply, isTrue);
    },
  );

  test("AgentReply.isEmptyReply is false for the person's own message", () {
    expect(_question('q1').isEmptyReply, isFalse);
  });

  test('AgentReply.steps reads the ai_ attachments in order, by their ids', () {
    final reply = _reply(
      'a1',
      attachments: [
        stream.Attachment(
          type: 'ai_reasoning',
          id: 'r1',
          extraData: const {'v': 1, 'status': 'completed'},
        ),
        stream.Attachment(type: 'image', imageUrl: 'https://example.com/a.png'),
        _toolCall('call_1'),
      ],
    );

    expect([for (final step in reply.steps) step.id], ['r1', 'call_1']);
  });

  test(
    'AgentMessage.attachments shows images and files and leaves out steps',
    () {
      final message = AgentMessage(
        _question(
          'q1',
          attachments: [
            stream.Attachment(
              type: 'image',
              imageUrl: 'https://example.com/a.png',
              title: 'a.png',
            ),
            stream.Attachment(
              type: 'file',
              assetUrl: 'https://example.com/b.pdf',
              title: 'b.pdf',
              extraData: const {'mime_type': 'application/pdf'},
            ),
            _toolCall('call_1'),
            stream.Attachment(type: 'image'),
          ],
        ),
        LiveReasoning(),
      );

      expect(message.attachments, [
        AIMessageAttachment.image(
          Uri.parse('https://example.com/a.png'),
          name: 'a.png',
        ),
        AIMessageAttachment.file(
          Uri.parse('https://example.com/b.pdf'),
          name: 'b.pdf',
          mimeType: 'application/pdf',
        ),
      ]);
    },
  );

  test(
    "AgentMessage.role is the assistant's for a reply and the user's otherwise",
    () {
      final reasoning = LiveReasoning();

      expect(
        AgentMessage(_reply('a1'), reasoning).role,
        AIMessageRole.assistant,
      );
      expect(AgentMessage(_question('q1'), reasoning).role, AIMessageRole.user);
    },
  );
}

stream.Message _question(
  String id, {
  List<stream.Attachment> attachments = const [],
}) => stream.Message(
  id: id,
  text: 'Where is order A-1042?',
  attachments: attachments,
);

stream.Message _reply(
  String id, {
  String text = '',
  bool generating = false,
  List<stream.Attachment> attachments = const [],
}) => stream.Message(
  id: id,
  text: text,
  attachments: attachments,
  extraData: {'ai_generated': true, 'generating': generating},
);

stream.Attachment _toolCall(String id) => stream.Attachment(
  type: 'ai_tool_call',
  id: id,
  extraData: const {'v': 1, 'status': 'running', 'name': 'lookup_order'},
);
