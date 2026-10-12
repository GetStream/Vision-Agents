import 'package:stream_chat/stream_chat.dart' as stream;
import 'package:test/test.dart';
import 'package:vision_agents_chat/vision_agents_chat.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

void main() {
  late VisionAgents agents;

  AgentSession sessionOf(VisionAgents agents, {String conversationId = 'messaging:s1'}) =>
      AgentSession(
        Session(id: 's1', createdAt: DateTime.utc(2026), conversationId: conversationId),
        backend: agents.backend,
        sessions: agents.sessions,
      );

  setUp(() {
    agents = VisionAgents(
      url: Uri.parse('http://127.0.0.1:9'),
      apiKey: 'stream-key',
      userId: 'ada',
      token: () async => 't',
    );
  });

  tearDown(() => agents.close());

  group('chat', () {
    test('opens the channel the conversation is kept in, on the app\'s own client', () async {
      final client = stream.StreamChatClient('stream-key');
      agents.useChat(client);

      final channel = await sessionOf(agents).chat();

      expect(channel.type, 'messaging');
      expect(channel.id, 's1');
      expect(channel.client, same(client));
      await agents.disconnect();
      expect(await sessionOf(agents).chat().then((channel) => channel.client), same(client));
      await client.dispose();
    });

    test('every session shares the one client', () async {
      final client = stream.StreamChatClient('stream-key');
      agents.useChat(client);

      final first = await sessionOf(agents).chat();
      final second = await sessionOf(agents, conversationId: 'messaging:s2').chat();

      expect(second.client, same(first.client));
      await client.dispose();
    });

    test('a session that keeps no transcript has no channel', () async {
      expect(sessionOf(agents, conversationId: '').chat(), throwsStateError);
    });

    test('needs the Stream app\'s key, which a customer id alone is not', () async {
      final byCustomer = VisionAgents(url: Uri.parse('http://127.0.0.1:9'), customerId: 'acme');

      expect(sessionOf(byCustomer).chat(), throwsArgumentError);
      byCustomer.close();
    });
  });
}
