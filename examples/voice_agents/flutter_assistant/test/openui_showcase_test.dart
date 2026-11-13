// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:flutter_assistant/brand_theme.dart';
import 'package:flutter_assistant/openui_renderer.dart';
import 'package:flutter_assistant/trip_components.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';

class _TestMessage with AIMessage {
  _TestMessage(this.id, this.uiDocument);

  @override
  final String id;

  @override
  final AIUIDocument uiDocument;

  @override
  AIMessageRole get role => AIMessageRole.assistant;

  @override
  String get text => 'Kaneo has a cliff terrace table at 19:15.';

  @override
  bool get isGenerating => uiDocument.isStreaming;

  @override
  List<AIMessagePart> get parts => const [];

  @override
  List<AIMessageAttachment> get attachments => const [];

  @override
  String? reasoningText(AIReasoningPart step) => null;
}

void main() {
  setUpAll(() async {
    TestWidgetsFlutterBinding.ensureInitialized();
    await TripComponents.load();
  });

  tearDown(() {
    BrandProfile.active.value = BrandProfile.stream;
  });

  test('TripComponents loads all 6 portable trip and hospitality booking widgets', () {
    expect(TripComponents.all.map((c) => c.name).toList(), [
      'place_card',
      'trip_map',
      'calendar_check',
      'restaurant_card',
      'availability_gauge',
      'seat_selector',
    ]);
  });

  testWidgets(
    'OpenUIRenderer progressively streams and interacts with RestaurantCard, AvailabilityGauge, SeatSelector, and BrandThemeScope',
    (tester) async {
      final actions = <AIUIAction>[];
      final host = AIUIHost(
        components: TripComponents.all,
        renderers: [OpenUIRenderer()],
        send: (action) async => actions.add(action),
      );
      addTearDown(host.dispose);

      const partialProgram = '''
root = Card([venue, gauge, tables])
venue = RestaurantCard("Kaneo Cliffside Kitchen", "Macedonian Grill · 1.2 km", 4.9, "\$\$\$")
''';

      final streamingDoc = AIUIDocument.fromCustomData({
        'ai_ui': {'v': 1, 'format': 'openui', 'status': 'streaming', 'content': partialProgram},
      })!;

      BrandProfile.active.value = BrandProfile.ember;
      host.receive([_TestMessage('m1', streamingDoc)]);

      await tester.pumpWidget(
        MaterialApp(
          theme: ThemeData(extensions: [StreamTheme.light()]),
          home: Scaffold(
            body: SingleChildScrollView(
              child: BrandThemeScope(
                child: AIUIView(host: host, document: streamingDoc, messageId: 'm1'),
              ),
            ),
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('Kaneo Cliffside Kitchen'), findsOneWidget);
      expect(find.textContaining('4.9'), findsOneWidget);

      const fullProgram = '''
root = Card([venue, gauge, tables])
venue = RestaurantCard("Kaneo Cliffside Kitchen", "Macedonian Grill · 1.2 km", 4.9, "\$\$\$")
gauge = AvailabilityGauge(86, "12 min wait")
tables = SeatSelector([{id: "T1", zone: "Cliff Terrace", seats: 2, available: true}, {id: "T4", zone: "Harbor Window", seats: 4, available: true}])
''';

      final completedDoc = AIUIDocument.fromCustomData({
        'ai_ui': {'v': 1, 'format': 'openui', 'status': 'complete', 'content': fullProgram},
      })!;
      host.receive([_TestMessage('m1', completedDoc)]);

      await tester.pumpWidget(
        MaterialApp(
          theme: ThemeData(extensions: [StreamTheme.light()]),
          home: Scaffold(
            body: SingleChildScrollView(
              child: BrandThemeScope(
                child: AIUIView(host: host, document: completedDoc, messageId: 'm1'),
              ),
            ),
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('86%'), findsOneWidget);
      expect(find.text('12 min wait'), findsOneWidget);
      expect(find.textContaining('Harbor Window'), findsOneWidget);

      await tester.tap(find.textContaining('Harbor Window'));
      await tester.pumpAndSettle();

      expect(actions, hasLength(1));
      expect(actions.single.name, 'select_table');
      expect(actions.single.payload['zone'], 'Harbor Window');
    },
  );
}
