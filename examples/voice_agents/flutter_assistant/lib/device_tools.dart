import 'dart:async';
import 'dart:io' show Platform;

import 'package:flutter/services.dart';
import 'package:flutter_timezone/flutter_timezone.dart';
import 'package:geocoding/geocoding.dart';
import 'package:geolocator/geolocator.dart';
import 'package:intl/intl.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

import 'gmail.dart';
import 'permissions.dart';

/// What the agent can ask this phone: the time and place, the calendar and reminders, and to
/// send an email from the person's Gmail.
///
/// The agent runs in the backend and knows none of this. Each tool runs here, when the model
/// calls it, and only its answer leaves the phone. The ones that share or do something personal
/// ask first: the call waits in the session's `approvals` until the person answers, on the
/// reply's step in a chat or on a card over a call. A permission the phone needs is asked for
/// the first time a tool needs it.
abstract final class DeviceTools {
  static final all = [
    _localContext,
    _location,
    _events,
    _reminder,
    _sendEmail,
  ];

  /// The calendar and the reminders, which this app's own iOS and Android code read and
  /// write: EventKit on iOS, the calendar provider on Android.
  static const _calendar = MethodChannel(
    'io.getstream.visionagents.flutterassistant/calendar',
  );

  /// The date, time and timezone, which a model in a data centre has no way to know.
  ///
  /// It takes no arguments, and still declares an empty object of them: a tool sent with no
  /// parameters at all is refused by xAI, which fails the whole reply.
  static final _localContext = AgentTool(
    name: 'get_local_context',
    description:
        "The person's current date, local time, timezone and language. Call it "
        'before answering anything that depends on today or the time.',
    parameters: AgentTool.strings({}),
    run: (_) async {
      final now = DateTime.now();
      final zone = (await FlutterTimezone.getLocalTimezone()).identifier;
      return localContext(
        now,
        zone: zone,
        language: Platform.localeName,
      );
    },
  );

  /// Where the person is, to the street. It asks first, since it is where they are.
  static final _location = AgentTool(
    name: 'get_location',
    description:
        'Where the person is: their street, neighbourhood and city. Only call it '
        'when the answer depends on where they are, such as the weather or places nearby.',
    parameters: AgentTool.strings(
      {'reason': 'why you need it, in a few words they would recognise'},
      required: ['reason'],
    ),
    displayTitle: 'Checking where you are',
    approval: const ToolApproval(
      title: 'Share your location?',
      message: 'Juno gets your street and city, once.',
      reasonArgument: 'reason',
      allowTitle: 'Share',
      declineTitle: 'Not now',
    ),
    run: (_) async {
      final here = await _currentPosition();
      final List<Placemark> places;
      try {
        places = await Geocoding().placemarkFromCoordinates(
          here.latitude,
          here.longitude,
        );
      } on Exception {
        return _noAddress(here);
      }
      final place = places.firstOrNull;
      if (place == null) return _noAddress(here);
      final parts = [
        place.thoroughfare,
        place.subLocality,
        place.locality,
        place.country,
      ].where((part) => part != null && part.isNotEmpty);
      return 'They are on ${parts.join(', ')}.';
    },
  );

  /// What is on the calendar for one day.
  static final _events = AgentTool(
    name: 'list_events',
    description: "The events on the person's calendar for one day.",
    parameters: AgentTool.strings(
      {'day': 'the day as YYYY-MM-DD, in their timezone'},
      required: ['day'],
    ),
    displayTitle: 'Checking your calendar',
    run: (arguments) async {
      final day = '${arguments['day'] ?? ''}';
      if (parseDay(day) == null) {
        throw const ToolFailure('The day has to be a date like 2026-10-16.');
      }
      if (!await Permissions.allOf([Permission.calendarFullAccess])) {
        throw const ToolFailure(
          'They have not let Juno see their calendar. It is in Settings.',
        );
      }
      final lines = await _ask<List<Object?>>('listEvents', {'day': day});
      if (lines.isEmpty) return 'Nothing is on their calendar that day.';
      return lines.join('\n');
    },
  );

  /// Adds a reminder, once the person says it may.
  ///
  /// On iOS it goes in their Reminders. Android has no Reminders of its own for an app to add
  /// to, so there it is an event in their calendar at the time it is due, with an alert then,
  /// and it needs a due time, as the Android app's does.
  static final _reminder = AgentTool(
    name: 'add_reminder',
    description: Platform.isIOS
        ? "Adds a reminder to the person's Reminders, with an alert when it is due."
        : "Adds a reminder to the person's calendar, with an alert when it is due.",
    parameters: {
      'type': 'object',
      'properties': {
        'title': {
          'type': 'string',
          'description': 'what to remind them of, such as Call mum',
        },
        'due': {
          'type': 'string',
          'description': Platform.isIOS
              ? 'when, as an ISO 8601 date and time with their UTC offset, such as '
                    '2026-10-16T18:00:00+02:00. Leave it out for no due time.'
              : 'when, as an ISO 8601 date and time with their UTC offset, such as '
                    '2026-10-16T18:00:00+02:00.',
        },
        'summary': {
          'type': 'string',
          'description':
              'the reminder and its time as they would read it, such as Call mum, '
              'today at 6pm',
        },
      },
      'required': [
        'title',
        if (!Platform.isIOS) 'due',
        'summary',
      ],
    },
    displayTitle: 'Adding a reminder',
    approval: ToolApproval(
      title: 'Add this reminder?',
      message: Platform.isIOS
          ? 'It goes in your Reminders, with an alert when it is due.'
          : 'It goes in your calendar, with an alert when it is due.',
      reasonArgument: 'summary',
      allowTitle: 'Add reminder',
      declineTitle: 'Not now',
    ),
    run: (arguments) async {
      final title = '${arguments['title'] ?? ''}';
      if (title.isEmpty) throw const ToolFailure('A reminder needs a title.');
      final due = parseDue('${arguments['due'] ?? ''}');
      if (due == null && !Platform.isIOS) {
        throw const ToolFailure(
          'A reminder needs a due time, such as 2026-10-16T18:00:00+02:00.',
        );
      }
      if (!await Permissions.allOf([Permission.calendarFullAccess])) {
        throw const ToolFailure(
          'They have not let Juno add reminders. It is in Settings.',
        );
      }
      await _ask<void>('addReminder', {
        'title': title,
        'due': due?.millisecondsSinceEpoch,
      });
      return 'Added the reminder "$title".';
    },
  );

  /// Sends an email from the person's Gmail, once they say it may.
  ///
  /// Google's Gmail MCP server writes drafts but cannot send, so sending is a tool here: the
  /// approval is asked on the phone, and the backend sends through the router's proxy for the
  /// person's Gmail connection, which holds the credential. The question shows who it goes to,
  /// read from the call itself rather than from anything the model wrote about it.
  static final _sendEmail = AgentTool(
    name: 'send_email',
    description:
        "Sends an email from the person's Gmail. Only call it once they have asked "
        'for it to be sent, after you have shown them the subject and the body.',
    parameters: AgentTool.strings(
      {
        'to': 'who it is to: one or more email addresses, separated by commas',
        'subject': 'the subject line',
        'body': 'the email itself, in plain text',
      },
      required: ['to', 'subject', 'body'],
    ),
    displayTitle: 'Sending your email',
    approval: const ToolApproval(
      title: 'Send this email?',
      message: 'Juno sends it from your Gmail, as you.',
      reasonArgument: 'to',
      allowTitle: 'Send',
      declineTitle: 'Not now',
    ),
    run: (arguments) async {
      final to = '${arguments['to'] ?? ''}';
      final subject = '${arguments['subject'] ?? ''}';
      final body = '${arguments['body'] ?? ''}';
      if (to.isEmpty || body.isEmpty) {
        throw const ToolFailure('An email needs someone to go to and a body.');
      }
      await Gmail.send(to: to, subject: subject, body: body);
      return 'Sent "$subject" to $to from their Gmail.';
    },
  );

  /// What get_local_context answers at [now], in [zone], for someone speaking [language].
  static String localContext(
    DateTime now, {
    required String zone,
    required String language,
  }) {
    final day = DateFormat('EEEE d MMMM y').format(now);
    final time = DateFormat.jm().format(now);
    return 'It is $day, $time in $zone (${now.timeZoneName}). '
        'Today as a date is ${DateFormat('yyyy-MM-dd').format(now)}. '
        'Their language is $language.';
  }

  /// When a reminder is due, from an ISO 8601 date and time with its UTC offset, or null when
  /// [text] is not one. A date alone, or a time with no offset, is not taken as midnight or as
  /// the phone's own time: iOS then adds the reminder with no due time, and Android asks for one,
  /// as the iOS and Android apps do.
  static DateTime? parseDue(String text) =>
      RegExp(
        r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})$',
      ).hasMatch(text)
      ? DateTime.tryParse(text)
      : null;

  /// A day as YYYY-MM-DD, or null when [text] is not one.
  static DateTime? parseDay(String text) {
    try {
      return DateFormat('yyyy-MM-dd').parseStrict(text);
    } on FormatException {
      return null;
    }
  }

  /// One reading of where the phone is, within 15 seconds.
  static Future<Position> _currentPosition() async {
    // On Android, asked in turn with the calendar, since only one question shows at a time.
    if (!await Permissions.allOf([Permission.locationWhenInUse])) {
      throw const ToolFailure(
        'They have not let Juno see where they are. It is in Settings.',
      );
    }
    var permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }
    if (permission == LocationPermission.denied ||
        permission == LocationPermission.deniedForever) {
      throw const ToolFailure(
        'They have not let Juno see where they are. It is in Settings.',
      );
    }
    try {
      return await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          timeLimit: Duration(seconds: 15),
        ),
      );
    } on TimeoutException {
      throw const ToolFailure('The phone could not tell where it is just now.');
    } on LocationServiceDisabledException {
      throw const ToolFailure(
        'Location is turned off on the phone. It is in Settings.',
      );
    }
  }

  static String _noAddress(Position here) =>
      'They are at latitude ${here.latitude}, longitude ${here.longitude}, '
      'which has no address.';

  /// Calls the app's calendar code. What it could not do comes back as words for the model.
  static Future<T> _ask<T>(String method, Map<String, Object?> arguments) async {
    try {
      return (await _calendar.invokeMethod<T>(method, arguments)) as T;
    } on PlatformException catch (error) {
      throw ToolFailure(error.message ?? 'The phone could not do that just now.');
    }
  }
}

/// Why a tool could not answer, in words the model passes on.
class ToolFailure implements Exception {
  const ToolFailure(this.message);

  final String message;

  @override
  String toString() => message;
}
