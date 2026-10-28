/// Hands over the token to present, asked for on every request and socket handshake.
///
/// A function rather than a value so a long-lived client never wakes up holding an expired
/// token. Caching one until it is about to expire is the provider's business.
typedef TokenProvider = Future<String> Function();

/// What Stream's own Chat and Video clients connect with: the app's key, the user and their
/// token.
final class StreamCredentials {
  const StreamCredentials({required this.apiKey, required this.userId, required this.token});

  final String apiKey;
  final String userId;
  final String token;
}

/// Where the router is and who is asking.
///
/// A device holds no secret worth having, so there is no API secret here. There are two ways
/// to be let in, and which one a deployment takes is its `ROUTER_AUTH_MODE`, not a choice
/// made per request:
///
/// * `customerId`, for a router in `noauth` mode, which believes the header. `userId` names
///   the end user, since there is no token to name them.
/// * `apiKey` with a `token` a backend minted for this user, or a guest's token, for a router
///   verifying tokens. With no token yet, the one thing it may do is ask for a guest.
///
/// The key, the user and the token are also what Stream Chat and Stream Video connect with,
/// so Stream is set up once. Beside a `customerId` the key is Stream's alone, and the router
/// is still reached by customer id.
///
/// Every request and handshake is sent as a device's: `Stream-Auth-Type: jwt` on requests,
/// and no auth type at all on a socket, which the router reads the same way.
final class Backend {
  Backend({
    Uri? url,
    String customerId = '',
    String apiKey = '',
    String userId = '',
    TokenProvider? token,
  }) : this._(url ?? defaultUrl, customerId, apiKey, userId, token, _StreamClients());

  Backend._(this.url, this.customerId, this.apiKey, this.userId, this.token, this._clients) {
    if (customerId.isEmpty && apiKey.isEmpty) {
      throw ArgumentError('a backend needs a customerId or an apiKey');
    }
  }

  /// Stream's hosted router, which is where a client goes when nothing says otherwise.
  static final Uri defaultUrl = Uri.parse('https://accelerate-us-east4.gcp.stream-io-api.com');

  /// The shape of a region name, such as us-east4 or europe-west4. Only the shape is checked:
  /// which regions are served is the hosted router's to say, so offering another needs no
  /// release of this SDK.
  static final _region = RegExp(r'^[a-z]+-[a-z]+[0-9]+$');

  /// Stream's hosted router held to one [region], such as us-east4: requests to it are served
  /// there and nowhere else, even when that region is full or down.
  ///
  /// Throws an [ArgumentError] for a name not shaped like a region.
  static Uri regionUrl(String region) {
    if (!_region.hasMatch(region)) {
      throw ArgumentError.value(region, 'region', 'is not a region, such as us-east4');
    }
    return Uri.parse('https://accelerate-$region.gcp.stream-io-api.com');
  }

  /// The router's base URL. Stream's hosted router unless another is named.
  final Uri url;

  /// Which tenant's agents these are, for a router that believes the header.
  final String customerId;

  /// The public half of a credential, for a router verifying tokens, and the Stream app's key.
  final String apiKey;

  /// Who is asking. A router verifying tokens reads it from the token, but Stream Chat and
  /// Stream Video need it named.
  final String userId;

  final TokenProvider? token;

  final _StreamClients _clients;

  bool get _byCustomer => customerId.isNotEmpty;

  /// Whether this holds the app's key and no token yet, which is how a guest is asked for
  /// before there is one: `Stream-Auth-Type: anonymous` and no token.
  bool get anonymous => !_byCustomer && token == null;

  /// The same router, asked by somebody else. Shares this one's Stream clients, which are
  /// kept per user.
  Backend withUser(String userId, {TokenProvider? token}) =>
      Backend._(url, customerId, apiKey, userId, token ?? this.token, _clients);

  /// What every request carries.
  ///
  /// `Stream-Auth-Type: jwt` declares a device, and is sent even to an open router, which
  /// would otherwise take a caller with no proxy in front of it for a backend. The key travels
  /// in the query instead, where both Stream's proxy and the router read it.
  Future<Map<String, String>> headers() async {
    if (_byCustomer) {
      return {
        'X-Customer-Id': customerId,
        'Stream-Auth-Type': 'jwt',
        if (userId.isNotEmpty) 'X-Stream-User-Id': userId,
      };
    }
    return {'Authorization': 'Bearer ${await _token()}', 'Stream-Auth-Type': 'jwt'};
  }

  /// The URL for a request path, with the query values that are set and the API key, the
  /// public half of the credential.
  Uri requestUri(String path, [Map<String, String?> query = const {}]) {
    final present = {
      if (!_byCustomer) 'api_key': apiKey,
      for (final MapEntry(:key, :value) in query.entries) key: ?value,
    };
    return url.replace(
      path: _join(url.path, path),
      queryParameters: present.isEmpty ? null : present,
    );
  }

  /// The URL of a socket under the router, credentials included.
  ///
  /// In the query because a browser WebSocket cannot set headers, and one URL shape for every
  /// platform is worth more than keeping a token out of a URL the VM could have avoided.
  /// There is deliberately no auth type here: the spec gives it no query counterpart, so a
  /// socket cannot claim to be a backend.
  Future<Uri> socketUri(String path, [Map<String, String> query = const {}]) async {
    final credentials = _byCustomer
        ? {'customer_id': customerId, if (userId.isNotEmpty) 'user_id': userId}
        : {'api_key': apiKey, 'token': await _token()};
    return url.replace(
      scheme: url.scheme == 'https' ? 'wss' : 'ws',
      path: _join(url.path, path),
      queryParameters: {...query, ...credentials},
    );
  }

  /// The key, user and token Stream's own clients connect with, the token asked for afresh.
  Future<StreamCredentials> streamCredentials() async {
    _requireStreamIdentity();
    return StreamCredentials(apiKey: apiKey, userId: userId, token: await token!());
  }

  /// Uses [client], the app's own Stream client of this [kind], rather than building one.
  /// It is never disconnected here.
  ///
  /// For the chat and rtc packages, whose `useChat` and `useVideo` are what an app calls.
  void give(String kind, Object client) {
    _requireStreamIdentity();
    _clients.given[_key(kind)] = client;
  }

  /// The one Stream client of this [kind] for this key and user: the one the app gave, the
  /// one already built, or one [open] builds now. Sessions asking at once share one opening,
  /// and one that fails to open is opened again by the next to ask.
  ///
  /// For the chat and rtc packages.
  Future<T> shared<T extends Object>(
    String kind, {
    required Future<T> Function(StreamCredentials credentials) open,
    required Future<void> Function(T client) disconnect,
  }) async {
    _requireStreamIdentity();
    final key = _key(kind);
    if (_clients.given[key] case final T client) {
      return client;
    }
    final entry = _clients.built.putIfAbsent(
      key,
      () => _Built(
        Future(() async => open(await streamCredentials())),
        (client) => disconnect(client as T),
      ),
    );
    try {
      return await entry.opening as T;
    } catch (_) {
      if (identical(_clients.built[key], entry)) {
        _clients.built.remove(key);
      }
      rethrow;
    }
  }

  /// Disconnects every Stream client built here, for every user, and none the app gave.
  Future<void> disconnectStream() async {
    final built = [..._clients.built.values];
    _clients.built.clear();
    for (final entry in built) {
      final Object client;
      try {
        client = await entry.opening;
      } catch (_) {
        // It never opened, so there is nothing to disconnect.
        continue;
      }
      await entry.disconnect(client);
    }
  }

  String _key(String kind) => '$kind:$apiKey:$userId';

  Future<String> _token() {
    final token = this.token;
    if (token == null) {
      throw ArgumentError(
        'an apiKey needs a token to go with it: pass token, or use withUser or withGuest',
      );
    }
    return token();
  }

  void _requireStreamIdentity() {
    if (apiKey.isEmpty) {
      throw ArgumentError('Stream Chat and Video need the Stream app\'s key: pass apiKey');
    }
    if (userId.isEmpty) {
      throw ArgumentError('Stream Chat and Video need to know who is connecting: pass userId');
    }
    if (token == null) {
      throw ArgumentError('Stream Chat and Video need the user\'s token: pass token');
    }
  }
}

final class _StreamClients {
  final given = <String, Object>{};
  final built = <String, _Built>{};
}

final class _Built {
  _Built(this.opening, this.disconnect);

  final Future<Object> opening;
  final Future<void> Function(Object client) disconnect;
}

String _join(String base, String path) =>
    base.endsWith('/') ? '${base.substring(0, base.length - 1)}$path' : '$base$path';
