import 'package:flutter_assistant/sign_in.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('SignIn.connected is true when the router says the account is connected', () {
    const callback =
        'io.getstream.visionagents.flutterassistant:/connected?connection_id=c1&status=connected';

    expect(SignIn.connected(callback), isTrue);
  });

  test('SignIn.connected is false when the person refused at the provider', () {
    const callback = 'io.getstream.visionagents.flutterassistant:/connected?status=denied';

    expect(SignIn.connected(callback), isFalse);
  });

  test('SignIn.connected is false when the callback says nothing of the account', () {
    const callback = 'io.getstream.visionagents.flutterassistant:/connected';

    expect(SignIn.connected(callback), isFalse);
  });

  test('SignIn.connected is false for a callback that is not a URL', () {
    expect(SignIn.connected('http://[::1'), isFalse);
  });
}
