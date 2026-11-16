import 'package:flutter_assistant/main.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('isLocalNetwork', () {
    test('is the Mac on the Wi-Fi or the emulator host', () {
      for (final host in ['192.168.1.20', '10.0.2.2', '172.16.0.4', '172.31.255.1', 'mac.local']) {
        expect(isLocalNetwork(host), isTrue, reason: host);
      }
    });

    test('is not the phone itself, a hosted router or a public address', () {
      for (final host in [
        'localhost',
        '127.0.0.1',
        '::1',
        'accelerate-us-east4.gcp.stream-io-api.com',
        '172.32.0.1',
        '8.8.8.8',
      ]) {
        expect(isLocalNetwork(host), isFalse, reason: host);
      }
    });
  });
}
