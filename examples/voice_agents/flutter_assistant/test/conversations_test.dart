import 'package:flutter_assistant/conversations.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

void main() {
  group('Conversations', () {
    final now = DateTime(2026, 10, 27, 15);

    test('lists a conversation under the day it started', () {
      expect(
        Conversations.dayTitle(DateTime(2026, 10, 27, 9), now: now),
        'Today',
      );
      expect(
        Conversations.dayTitle(DateTime(2026, 10, 26, 23), now: now),
        'Yesterday',
      );
      expect(
        Conversations.dayTitle(DateTime(2026, 10, 22, 16), now: now),
        'This week',
      );
      expect(
        Conversations.dayTitle(DateTime(2026, 9, 3), now: now),
        'September 2026',
      );
    });

    test('keeps the router order, starting a section when the day changes', () {
      Session session(String id, DateTime createdAt) =>
          Session(id: id, createdAt: createdAt);
      final sections = Conversations.sections([
        session('a', DateTime(2026, 10, 27, 9)),
        session('b', DateTime(2026, 10, 27, 8)),
        session('c', DateTime(2026, 10, 26, 8)),
        // Active again, though it started today: the list is in order of activity.
        session('d', DateTime(2026, 10, 27, 7)),
      ], now: now);

      expect(
        [
          for (final section in sections)
            '${section.title}: ${section.items.map((item) => item.id).join(', ')}',
        ],
        ['Today: a, b', 'Yesterday: c', 'Today: d'],
      );
    });

    test('names a conversation the router has not titled yet', () {
      expect(
        Session(id: 'a', createdAt: now).displayTitle,
        'New conversation',
      );
      expect(
        Session(id: 'a', createdAt: now, title: 'Lisbon weekend').displayTitle,
        'Lisbon weekend',
      );
    });
  });
}
