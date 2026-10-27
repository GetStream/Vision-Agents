import 'package:flutter_assistant/device_tools.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('DeviceTools', () {
    test('tells the model the date, time, timezone and language', () {
      final now = DateTime(2026, 10, 16, 18, 5);
      final said = DeviceTools.localContext(
        now,
        zone: 'Europe/Amsterdam',
        language: 'nl_NL',
      );

      expect(
        said,
        'It is Friday 16 October 2026, 6:05\u202fPM in Europe/Amsterdam '
        '(${now.timeZoneName}). Today as a date is 2026-10-16. '
        'Their language is nl_NL.',
      );
    });

    test('takes a due time only with its date, time and UTC offset', () {
      expect(
        DeviceTools.parseDue('2026-10-16T18:00:00+02:00'),
        DateTime.utc(2026, 10, 16, 16),
      );
      expect(
        DeviceTools.parseDue('2026-10-16T18:00Z'),
        DateTime.utc(2026, 10, 16, 18),
      );
      expect(DeviceTools.parseDue('2026-10-16'), isNull);
      expect(DeviceTools.parseDue('2026-10-16T18:00:00'), isNull);
      expect(DeviceTools.parseDue('tomorrow at 6'), isNull);
    });

    test('reads a day only as YYYY-MM-DD', () {
      expect(DeviceTools.parseDay('2026-10-16'), DateTime(2026, 10, 16));
      expect(DeviceTools.parseDay('2026-02-30'), isNull);
      expect(DeviceTools.parseDay('16 October'), isNull);
      expect(DeviceTools.parseDay(''), isNull);
    });
  });
}
