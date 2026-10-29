import 'package:flutter_assistant/conversations.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

void main() {
  test('Conversations.dayTitle lists one started earlier today under Today', () {
    final title = Conversations.dayTitle(DateTime(2026, 10, 27, 9), now: _now);

    expect(title, 'Today');
  });

  test('Conversations.dayTitle lists one started late yesterday under Yesterday', () {
    final title = Conversations.dayTitle(DateTime(2026, 10, 26, 23), now: _now);

    expect(title, 'Yesterday');
  });

  test('Conversations.dayTitle lists one started six days ago under This week', () {
    final title = Conversations.dayTitle(DateTime(2026, 10, 21, 9), now: _now);

    expect(title, 'This week');
  });

  test('Conversations.dayTitle lists one started seven days ago under its month, '
      'though less than a week has passed since', () {
    // Later in the day than now: 6 days and 22 hours ago, but a week ago by the calendar.
    final title = Conversations.dayTitle(DateTime(2026, 10, 20, 17), now: _now);

    expect(title, 'October 2026');
  });

  test('Conversations.dayTitle lists one from an earlier month under that month', () {
    final title = Conversations.dayTitle(DateTime(2026, 9, 3), now: _now);

    expect(title, 'September 2026');
  });

  test(
    'Conversations.sections keeps the router order, starting a section when the day changes',
    () {
      final sections = Conversations.sections([
        _session('a', DateTime(2026, 10, 27, 9)),
        _session('b', DateTime(2026, 10, 27, 8)),
        _session('c', DateTime(2026, 10, 26, 8)),
        // Active again, though it started today: the list is in order of activity.
        _session('d', DateTime(2026, 10, 27, 7)),
      ], now: _now);

      expect(
        [
          for (final section in sections)
            '${section.title}: ${section.items.map((item) => item.id).join(', ')}',
        ],
        ['Today: a, b', 'Yesterday: c', 'Today: d'],
      );
    },
  );

  test('Session.displayTitle names a conversation the router has not titled yet', () {
    expect(_session('a', _now).displayTitle, 'New conversation');
  });

  test("Session.displayTitle is the router's title once it has one", () {
    final session = Session(id: 'a', createdAt: _now, title: 'Lisbon weekend');

    expect(session.displayTitle, 'Lisbon weekend');
  });
}

/// The moment the list is seen at in these tests: 27 October 2026, 3pm.
final _now = DateTime(2026, 10, 27, 15);

Session _session(String id, DateTime createdAt) => Session(id: id, createdAt: createdAt);
