import 'package:test/test.dart';
import 'package:vision_agents_core/vision_agents_core.dart';

/// Stands in for a Stream client: something with an identity of its own, connected as a user.
final class _Peer {
  _Peer(this.user);

  final String user;
  bool connected = true;
}

/// What a chat or video package opens and closes: every peer it built, and whether it is
/// still connected.
final class _Peers {
  final built = <_Peer>[];

  Future<_Peer> open(StreamCredentials credentials) async {
    await Future<void>.delayed(const Duration(milliseconds: 10));
    final peer = _Peer(credentials.userId);
    built.add(peer);
    return peer;
  }

  Future<void> disconnect(_Peer peer) async => peer.connected = false;
}

Backend _signedIn(String user) => Backend(
  url: Uri.parse('http://localhost'),
  apiKey: 'key',
  userId: user,
  token: () async => 'token-for-$user',
);

void main() {
  group('Stream clients', () {
    test('the key, the user and a fresh token are what Stream connects with', () async {
      var asked = 0;
      final backend = Backend(apiKey: 'key', userId: 'john', token: () async => 'token-${++asked}');

      final first = await backend.streamCredentials();
      final again = await backend.streamCredentials();

      expect([first.apiKey, first.userId, first.token], ['key', 'john', 'token-1']);
      expect(again.token, 'token-2');
    });

    test('refuses without a Stream key or a user, before asking for anything', () {
      final local = Backend(url: Uri.parse('http://localhost'), customerId: 'acme', userId: 'john');
      final nobody = Backend(apiKey: 'key', token: () async => 'token');

      expect(local.streamCredentials, throwsArgumentError);
      expect(nobody.streamCredentials, throwsArgumentError);
    });

    test("beside a customer id the key is Stream's", () async {
      final local = Backend(
        url: Uri.parse('http://localhost'),
        customerId: 'acme',
        apiKey: 'stream-key',
        userId: 'john',
        token: () async => 'token-for-john',
      );

      final credentials = await local.streamCredentials();

      expect([credentials.apiKey, credentials.token], ['stream-key', 'token-for-john']);
    });

    test('sessions asking at once share one client', () async {
      final peers = _Peers();
      final backend = _signedIn('john');

      final clients = await Future.wait([
        for (var i = 0; i < 3; i++)
          backend.shared('chat', open: peers.open, disconnect: peers.disconnect),
      ]);

      expect(peers.built, hasLength(1));
      expect(clients.every((client) => identical(client, clients.first)), isTrue);
    });

    test('chat and video are a client each', () async {
      final peers = _Peers();
      final backend = _signedIn('john');

      final chat = await backend.shared('chat', open: peers.open, disconnect: peers.disconnect);
      final video = await backend.shared('video', open: peers.open, disconnect: peers.disconnect);

      expect(identical(chat, video), isFalse);
      expect(peers.built, hasLength(2));
    });

    test('a client that failed to open is opened again', () async {
      var tries = 0;
      final backend = _signedIn('john');
      Future<_Peer> open(StreamCredentials credentials) async {
        if (++tries == 1) {
          throw StateError('offline');
        }
        return _Peer(credentials.userId);
      }

      await expectLater(
        backend.shared('chat', open: open, disconnect: (_) async {}),
        throwsStateError,
      );
      final peer = await backend.shared('chat', open: open, disconnect: (_) async {});

      expect(peer.user, 'john');
      expect(tries, 2);
    });

    test('another user gets a client of their own', () async {
      final peers = _Peers();
      final backend = _signedIn('john');

      final john = await backend.shared('chat', open: peers.open, disconnect: peers.disconnect);
      final jane = await backend
          .withUser('jane', token: () async => 'token-for-jane')
          .shared('chat', open: peers.open, disconnect: peers.disconnect);

      expect([john.user, jane.user], ['john', 'jane']);
    });

    test("the app's own client is used, kept, and left connected", () async {
      final peers = _Peers();
      final agents = VisionAgents.withBackend(_signedIn('john'));
      final mine = _Peer('john');
      agents.backend.give('chat', mine);

      final first = await agents.backend.shared(
        'chat',
        open: peers.open,
        disconnect: peers.disconnect,
      );
      final second = await agents.backend.shared(
        'chat',
        open: peers.open,
        disconnect: peers.disconnect,
      );
      await agents.disconnect();
      agents.close();

      expect(identical(first, mine) && identical(second, mine), isTrue);
      expect(peers.built, isEmpty);
      expect(mine.connected, isTrue);
    });

    test("the app's own client is not handed to another user", () async {
      final peers = _Peers();
      final backend = _signedIn('john');
      backend.give('chat', _Peer('john'));

      final jane = await backend
          .withUser('jane', token: () async => 'token-for-jane')
          .shared('chat', open: peers.open, disconnect: peers.disconnect);

      expect(jane.user, 'jane');
      expect(peers.built, [jane]);
    });

    test('disconnecting closes what was built, and the next session opens anew', () async {
      final peers = _Peers();
      final agents = VisionAgents.withBackend(_signedIn('john'));

      final first = await agents.backend.shared(
        'chat',
        open: peers.open,
        disconnect: peers.disconnect,
      );
      await agents.disconnect();
      final second = await agents.backend.shared(
        'chat',
        open: peers.open,
        disconnect: peers.disconnect,
      );
      agents.close();

      expect(identical(first, second), isFalse);
      expect([for (final peer in peers.built) peer.connected], [false, true]);
    });
  });
}
