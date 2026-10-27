import 'dart:async';
import 'dart:io' show Platform;

import 'package:permission_handler/permission_handler.dart';

/// Asks Android for a permission the first time something needs it, one question at a time.
///
/// A tool runs while the model waits, so it asks from inside its handler and carries on with
/// the answer. Two tools can be called at once, and Android shows one request at a time, so
/// each waits for the one before. iOS asks by itself the first time the calendar, the
/// reminders, the camera or the microphone is used, so there this answers yes and leaves the
/// question to the system.
abstract final class Permissions {
  static Future<void> _asking = Future.value();

  /// Whether every one of [permissions] is granted, asking for those that are not.
  static Future<bool> allOf(List<Permission> permissions) {
    if (!Platform.isAndroid) return Future.value(true);
    final answer = _asking.then(
      (_) async => (await permissions.request()).values.every(
        (status) => status.isGranted,
      ),
    );
    _asking = answer.then<void>((_) {}, onError: (_) {});
    return answer;
  }
}
