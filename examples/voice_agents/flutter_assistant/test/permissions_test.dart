import 'package:flutter_assistant/permissions.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:permission_handler/permission_handler.dart';

void main() {
  group('Permissions.isRefused', () {
    test('is true for a permission the person or the device refused', () {
      expect(Permissions.isRefused(PermissionStatus.permanentlyDenied), isTrue);
      expect(Permissions.isRefused(PermissionStatus.restricted), isTrue);
    });

    test('is false for one iOS has not asked for yet, which reads as denied', () {
      expect(Permissions.isRefused(PermissionStatus.denied), isFalse);
    });

    test('is false for one that is granted', () {
      expect(Permissions.isRefused(PermissionStatus.granted), isFalse);
      expect(Permissions.isRefused(PermissionStatus.limited), isFalse);
    });
  });
}
