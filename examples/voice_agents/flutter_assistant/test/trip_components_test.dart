// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:flutter_assistant/agent_chat.dart';
import 'package:flutter_assistant/brand_theme.dart';
import 'package:flutter_assistant/openui_renderer.dart';
import 'package:flutter_assistant/trip_components.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';

import 'support/ui_message.dart';

void main() {
  setUpAll(() async {
    TestWidgetsFlutterBinding.ensureInitialized();
    await TripComponents.load();
  });

  group('SeatSelectorComponent', () {
    const offered = '''
root = SeatSelector([{id: "T1", zone: "Cliff Terrace", seats: 2, available: true}, {id: "T4", zone: "Harbor Window", seats: 4, available: true}])
''';
    const others = '''
root = SeatSelector([{id: "T7", zone: "Garden", seats: 6, available: true}])
''';

    Future<void> show(WidgetTester tester, AgentChat chat, List<UIMessage> replies) async {
      final host = AIUIHost(
        components: TripComponents.all,
        renderers: [OpenUIRenderer()],
        send: (_) async {},
      );
      addTearDown(host.dispose);
      host.receive(replies);
      await tester.pumpWidget(
        MaterialApp(
          theme: ThemeData(extensions: [StreamTheme.light()]),
          home: Scaffold(
            body: AgentChatScope(
              chat: chat,
              child: BrandThemeScope(
                child: Column(
                  children: [
                    for (final reply in replies)
                      AIUIView(host: host, document: reply.uiDocument, messageId: reply.id),
                  ],
                ),
              ),
            ),
          ),
        ),
      );
      await tester.pumpAndSettle();
    }

    testWidgets('can\'t be tapped while the agent is still writing it', (tester) async {
      final chat = AgentChat(null);
      addTearDown(chat.dispose);
      await show(tester, chat, [
        UIMessage(
          'm1',
          'root = SeatSelector([{id: "T1", zone: "Cliff Terrace", seats: 2, available: true}, '
              '{id: "T4", zone: "Harbor Window", seats: 4, available: true}])',
          isStreaming: true,
        ),
      ]);

      final chip = tester.widget<ChoiceChip>(
        find.ancestor(of: find.textContaining('T4 · Harbor'), matching: find.byType(ChoiceChip)),
      );
      expect(chip.onSelected, isNull);
    });

    testWidgets('shows the booking of one of its tables in place of the tables', (tester) async {
      final chat = AgentChat(null);
      addTearDown(chat.dispose);
      await show(tester, chat, [UIMessage('m1', offered), UIMessage('m2', others)]);
      expect(find.text('Select a table zone'), findsNWidgets(2));
      expect(find.textContaining('T4 · Harbor Window'), findsOneWidget);

      final reservation = chat.reserve('T4', 'tonight at 8pm');
      await tester.pumpAndSettle();

      expect(find.text('Table T4 reserved'), findsOneWidget);
      expect(find.text('Harbor Window · 4 people · tonight at 8pm'), findsOneWidget);
      expect(find.text('Confirmation ${reservation.code}'), findsOneWidget);
      // The booked card no longer offers its tables.
      expect(find.textContaining('T1 · Cliff Terrace'), findsNothing);
      expect(find.textContaining('T4 · Harbor Window'), findsNothing);
      // The card of other tables still offers them.
      expect(find.text('Select a table zone'), findsOneWidget);
      expect(find.textContaining('T7 · Garden'), findsOneWidget);
    });
  });
}
