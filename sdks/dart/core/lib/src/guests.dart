import 'dart:convert';

import 'models.dart';
import 'wire.dart';

/// Somewhere to remember a guest between launches.
///
/// Narrow on purpose, so an app passes whatever it already persists with. With Flutter's
/// `shared_preferences` it is three lines:
///
///     final class PreferencesGuestStore implements GuestStore {
///       PreferencesGuestStore(this.prefs);
///       final SharedPreferencesAsync prefs;
///       Future<String?> read() => prefs.getString(guestStorageKey);
///       Future<void> write(String value) => prefs.setString(guestStorageKey, value);
///       Future<void> clear() => prefs.remove(guestStorageKey);
///     }
abstract interface class GuestStore {
  Future<String?> read();
  Future<void> write(String value);
  Future<void> clear();
}

/// The key a guest is kept under, the same one the JavaScript SDK uses.
const guestStorageKey = 'stream-vision-agents-guest';

/// A guest held in memory, which is remembered for as long as the process runs.
final class MemoryGuestStore implements GuestStore {
  String? _value;

  @override
  Future<String?> read() async => _value;

  @override
  Future<void> write(String value) async => _value = value;

  @override
  Future<void> clear() async => _value = null;
}

/// What a store holds: the guest, and the refresh secret that brings them back once their
/// token has expired.
///
/// The secret is kept here and not on [GuestUser]: it is a credential for coming back as this
/// guest, and nothing an app does with a guest needs it.
typedef RememberedGuest = ({GuestUser guest, String? refreshSecret});

/// The guest a store holds, or null for nothing or something unreadable under the key.
///
/// Unreadable is not an error: minting a fresh guest is the recoverable answer, and throwing
/// would leave the app unable to ask anything at all.
Future<RememberedGuest?> readGuest(GuestStore store) async {
  final held = await store.read();
  if (held == null) {
    return null;
  }
  try {
    final decoded = jsonDecode(held);
    if (decoded is! Map || decoded['id'] is! String || decoded['token'] is! String) {
      return null;
    }
    final expires = decoded['expires_at'];
    final secret = decoded['refresh_secret'];
    return (
      guest: GuestUser(
        id: decoded['id'] as String,
        token: decoded['token'] as String,
        name: decoded['name'] is String ? decoded['name'] as String : '',
        custom: decoded['custom'] is Map
            ? (decoded['custom'] as Map).cast<String, Object?>()
            : const {},
        expiresAt: expires is String ? parseRouterDate(expires) : null,
      ),
      refreshSecret: secret is String && secret.isNotEmpty ? secret : null,
    );
  } on FormatException {
    return null;
  }
}

/// Remembers [guest] with the refresh secret the router handed out with it, which replaces
/// the one held before: each works once.
Future<void> writeGuest(GuestStore store, GuestUser guest, {String? refreshSecret}) => store.write(
  jsonEncode({
    'id': guest.id,
    'token': guest.token,
    if (guest.name.isNotEmpty) 'name': guest.name,
    if (guest.custom.isNotEmpty) 'custom': guest.custom,
    if (guest.expiresAt case final expires?) 'expires_at': expires.toUtc().toIso8601String(),
    'refresh_secret': ?refreshSecret,
  }),
);
