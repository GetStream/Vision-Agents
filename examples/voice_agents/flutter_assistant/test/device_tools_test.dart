import 'package:flutter_assistant/device_tools.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('DeviceTools.localContext tells the model the date, time, timezone and language', () {
    final now = DateTime(2026, 10, 16, 18, 5);

    final said = DeviceTools.localContext(now, zone: 'Europe/Amsterdam', language: 'nl_NL');

    expect(
      said,
      // intl puts a narrow no-break space before the AM/PM marker.
      'It is Friday 16 October 2026, 6:05 PM in Europe/Amsterdam '
      '(${now.timeZoneName}). Today as a date is 2026-10-16. '
      'Their language is nl_NL.',
    );
  });

  test('DeviceTools.parseDue reads a time with its UTC offset as that moment', () {
    expect(DeviceTools.parseDue('2026-10-16T18:00:00+02:00'), DateTime.utc(2026, 10, 16, 16));
  });

  test('DeviceTools.parseDue reads a time in UTC', () {
    expect(DeviceTools.parseDue('2026-10-16T18:00Z'), DateTime.utc(2026, 10, 16, 18));
  });

  test('DeviceTools.parseDue keeps fractions of a second', () {
    expect(
      DeviceTools.parseDue('2026-10-16T18:00:00.5Z'),
      DateTime.utc(2026, 10, 16, 18, 0, 0, 500),
    );
  });

  test('DeviceTools.parseDue takes no date without a time', () {
    expect(DeviceTools.parseDue('2026-10-16'), isNull);
  });

  test('DeviceTools.parseDue takes no time without a UTC offset', () {
    expect(DeviceTools.parseDue('2026-10-16T18:00:00'), isNull);
  });

  test('DeviceTools.parseDue takes no time in words', () {
    expect(DeviceTools.parseDue('tomorrow at 6'), isNull);
  });

  test('DeviceTools.parseDue takes no date that does not exist', () {
    expect(DeviceTools.parseDue('2026-02-30T18:00Z'), isNull);
  });

  test('DeviceTools.parseDue takes no hour past 23', () {
    expect(DeviceTools.parseDue('2026-10-16T24:00Z'), isNull);
  });

  test('DeviceTools.parseDay reads a day as YYYY-MM-DD', () {
    expect(DeviceTools.parseDay('2026-10-16'), DateTime(2026, 10, 16));
  });

  test('DeviceTools.parseDay takes no date that does not exist', () {
    expect(DeviceTools.parseDay('2026-02-30'), isNull);
  });

  test('DeviceTools.parseDay takes no day in words', () {
    expect(DeviceTools.parseDay('16 October'), isNull);
  });

  test('DeviceTools.parseDay takes no empty day', () {
    expect(DeviceTools.parseDay(''), isNull);
  });
}
