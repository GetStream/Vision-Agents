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

  // The windows of thinking the router puts on a reply's live updates, as `reasoningWindow` in
  // `acceleration/internal/conversation/reasoning.go` writes them.
  group('LiveReasoning', () {
    late LiveReasoning reasoning;

    void window(String text, int offset, [String id = 'r1']) => reasoning.read(
      stream.Message(
        text: 'ignored',
        extraData: {
          'reasoning': {'id': id, 'offset': offset, 'text': text, 'length': offset + text.length},
        },
      ),
    );

    setUp(() => reasoning = LiveReasoning());

    test('windows in order make the whole thinking', () {
      window('The order was ', 0);
      window('delivered unopened.', 14);

      expect(reasoning['r1'], 'The order was delivered unopened.');
    });

    test('a repeat of recent thinking adds only what is new', () {
      window('Checking the policy', 0);

      window('the policy, which allows 30 days', 9);
      window('policy', 13);

      expect(reasoning['r1'], 'Checking the policy, which allows 30 days');
    });

    test('a missed window leaves a paragraph break rather than running on', () {
      window('First thought.', 0);

      window('Much later.', 40);

      expect(reasoning['r1'], 'First thought.\n\nMuch later.');
    });

    test('somebody joining midway starts from the window they get', () {
      window('the rest of it', 300, 'r2');

      expect(reasoning['r2'], 'the rest of it');
      expect(reasoning['r1'], isNull);
    });

    test('positions count code points as the router does', () {
      window('café ', 0);

      window(' 🌸 bloom', 4);

      expect(reasoning['r1'], 'café 🌸 bloom');
    });

    test('a message with no window changes nothing', () {
      reasoning.read(stream.Message(text: 'Hello'));

      expect(reasoning['r1'], isNull);
    });
  });
}
