import 'package:flutter/services.dart';
import 'package:flutter_web_auth_2/flutter_web_auth_2.dart';

import 'assistant.dart';

/// The browser half of a connector's sign-in, such as Linear's or Gmail's.
///
/// The page is the browser's, so the provider's credentials never pass through the app: a web
/// authentication sheet on iOS, and on Android an Auth Tab, or a Custom Tab where the browser
/// has none. The router sends the browser back to [Assistant.returnUrl] once the provider has
/// answered.
abstract final class SignIn {
  /// Opens [url], the sign-in's launch URL with its handoff token, and completes with whether
  /// the person connected.
  ///
  /// Completes with false when the page is closed before the provider answers. Throws a
  /// [PlatformException] when the page cannot be opened, such as when the phone has no browser.
  static Future<bool> open(Uri url) async {
    try {
      final back = await FlutterWebAuth2.authenticate(
        url: '$url',
        callbackUrlScheme: Assistant.callbackScheme,
      );
      return connected(back);
    } on PlatformException catch (error) {
      if (error.code == 'CANCELED') return false;
      rethrow;
    }
  }

  /// Whether [callback], where the router sent the browser back to, says the account is
  /// connected.
  ///
  /// The router adds `status=connected` once it holds the grant, and another status when the
  /// person refused at the provider or the sign-in failed.
  static bool connected(String callback) =>
      Uri.tryParse(callback)?.queryParameters['status'] == 'connected';
}
