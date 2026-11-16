import 'dart:async';
import 'dart:io' show Platform;

import 'package:flutter/foundation.dart';
import 'package:permission_handler/permission_handler.dart';

/// The permission questions Android asks the first time something needs them, one at a time.
///
/// A tool runs while the model waits, so it asks from inside its handler and carries on with
/// the answer. Two tools can be called at once, and Android shows one request at a time, so
/// each waits for the one before. iOS asks by itself the first time the calendar, the
/// reminders, the camera or the microphone is used, so there this answers yes and leaves the
/// question to the system. All it says no to there is a permission the person already
/// refused, when the caller [needed] it: that shows with a way to Settings.
abstract final class Permissions {
  static Future<void> _asking = Future.value();

  /// Whether every one of [permissions] is granted, asking for those that are not.
  ///
  /// On iOS it answers yes without asking, and the system asks when the permission is first
  /// used, unless [needed] and the person already refused one.
  static Future<bool> allOf(List<Permission> permissions, {bool needed = false}) {
    if (Platform.isIOS && needed) return has(permissions);
    if (!Platform.isAndroid) return Future.value(true);
    final answer = _asking.then(
      (_) async => (await permissions.request()).values.every((status) => status.isGranted),
    );
    _asking = answer.then<void>((_) {}, onError: (_) {});
    return answer;
  }

  /// Whether every one of [permissions] is there to use, without asking: granted on Android,
  /// and not refused on iOS, where one not asked for yet is asked for by the system when it
  /// is first used.
  static Future<bool> has(List<Permission> permissions) async {
    if (!Platform.isAndroid && !Platform.isIOS) return true;
    for (final permission in permissions) {
      final PermissionStatus status;
      try {
        status = await permission.status;
      } on Exception {
        // Nothing to read it with, so it is left to the system.
        if (Platform.isIOS) continue;
        return false;
      }
      if (Platform.isIOS ? isRefused(status) : !status.isGranted) return false;
    }
    return true;
  }

  /// Whether [status], read on iOS without asking, says the person refused the permission.
  ///
  /// Nothing else there does. One not asked for yet reads as denied, and so does every one
  /// when the plugin was built without it, as it is under SwiftPM when it does not find the
  /// app's `Info.plist`. Asking through the plugin then answers "permanently denied" without
  /// the system having asked anything, which is why iOS is not asked through it.
  @visibleForTesting
  static bool isRefused(PermissionStatus status) =>
      status == PermissionStatus.permanentlyDenied || status == PermissionStatus.restricted;

  /// Opens the app's page in the system's Settings, where a denied permission is given.
  static Future<bool> openSettings() => openAppSettings();
}
