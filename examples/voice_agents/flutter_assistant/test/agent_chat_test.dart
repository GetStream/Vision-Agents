import 'package:flutter_assistant/agent_chat.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:stream_chat/stream_chat.dart' as stream;
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';
import 'package:vision_agents_chat/vision_agents_chat.dart';

void main() {
  test('AgentChat.workingOnSkills names a skill in words', () {
    expect(AgentChat.workingOnSkills(['web_search']), 'Working on web search');
  });

  test('AgentChat.workingOnSkills names every skill being worked on', () {
    expect(AgentChat.workingOnSkills(['web_search', 'recall']), 'Working on web search and recall');
  });

  test('AgentReply.isAgentReply is true for a message the agent wrote', () {
    expect(_agentReply(text: 'Hello').isAgentReply, isTrue);
  });

  test("AgentReply.isAgentReply is false for the person's message", () {
    expect(stream.Message(id: 'q', text: 'Hi').isAgentReply, isFalse);
  });

  test('AgentReply.isGenerating is true while the agent is still writing', () {
    final reply = _agentReply(text: 'Hel', extraData: {'generating': true});

    expect(reply.isGenerating, isTrue);
  });

  test('AgentReply.isEmptyReply is true for a reply with no text and no steps yet', () {
    expect(_agentReply().isEmptyReply, isTrue);
  });

  test('AgentReply.isEmptyReply is false once the reply has text', () {
    expect(_agentReply(text: 'Hello').isEmptyReply, isFalse);
  });

  test("AgentMessage.role is the assistant's for the agent's reply", () {
    final message = AgentMessage(_agentReply(text: 'Hello'), LiveReasoning());

    expect(message.role, AIMessageRole.assistant);
  });

  test("AgentMessage.role is the user's for the person's message", () {
    final message = AgentMessage(stream.Message(id: 'q', text: 'Hi'), LiveReasoning());

    expect(message.role, AIMessageRole.user);
  });

  test('AgentMessage.attachments shows a photo by its URL', () {
    final question = stream.Message(
      id: 'q',
      attachments: [
        stream.Attachment(type: 'image', imageUrl: 'https://cdn.example.com/photo.jpg'),
      ],
    );

    final attachments = AgentMessage(question, LiveReasoning()).attachments;

    expect(attachments.single.url, Uri.parse('https://cdn.example.com/photo.jpg'));
  });

  test('AgentMessage.attachments leaves out a photo not uploaded yet', () {
    final question = stream.Message(
      id: 'q',
      attachments: [stream.Attachment(type: 'image')],
    );

    final attachments = AgentMessage(question, LiveReasoning()).attachments;

    expect(attachments, isEmpty);
  });
}

stream.Message _agentReply({String? text, Map<String, Object?> extraData = const {}}) =>
    stream.Message(id: 'r', text: text, extraData: {'ai_generated': true, ...extraData});
