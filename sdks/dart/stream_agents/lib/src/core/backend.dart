/// Hands over a token for the user, and a fresh one when asked again.
///
/// A function rather than a string, because a token expires and a long conversation should
/// not end when it does. It is asked once, then again after a 401 or when Stream says the
/// one held has expired.
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
/// and `stream-auth-type=jwt` in a socket's query.
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

  Future<String>? _pendingToken;
  bool _tokenFetched = false;

  bool get _byCustomer => customerId.isNotEmpty;

  /// Whether this holds the app's key and no token yet, which is how a guest is asked for
  /// before there is one: `Stream-Auth-Type: anonymous` and no token.
  bool get anonymous => !_byCustomer && token == null;

  /// The same router, asked by somebody else. Shares this one's Stream clients, which are
  /// kept per user.
  Backend withUser(String userId, {TokenProvider? token}) {
    final next = Backend._(url, customerId, apiKey, userId, token ?? this.token, _clients);
    if (token == null && userId == this.userId) {
      next
        .._pendingToken = _pendingToken
        .._tokenFetched = _tokenFetched;
    }
    return next;
  }

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
  ///
  /// By customer id the socket declares a device as `stream-auth-type=jwt`, without which a
  /// router with no proxy in front takes it for a backend and never asks it to run this
  /// device's tools. By key the token also goes as `authorization` with `stream-auth-type=jwt`,
  /// which is how Stream's proxy reads a socket's credential: it refuses one carrying only
  /// `token`. The router itself reads `token` and ignores both.
  Future<Uri> socketUri(String path, [Map<String, String> query = const {}]) async {
    final Map<String, String> credentials;
    if (_byCustomer) {
      credentials = {
        'customer_id': customerId,
        if (userId.isNotEmpty) 'user_id': userId,
        'stream-auth-type': 'jwt',
      };
    } else {
      final token = await _token();
      credentials = {
        'api_key': apiKey,
        'token': token,
        'authorization': token,
        'stream-auth-type': 'jwt',
      };
    }
    return url.replace(
      scheme: url.scheme == 'https' ? 'wss' : 'ws',
      path: _join(url.path, path),
      queryParameters: {...query, ...credentials},
    );
  }

  /// The key, user and token Stream's own clients connect with.
  ///
  /// [refresh] is for a token the caller was told has expired: the one held is dropped and
  /// the provider asked again, unless a fresh one is already on its way, so two clients
  /// refreshing at once fetch one token.
  Future<StreamCredentials> streamCredentials({bool refresh = false}) async {
    _requireStreamIdentity();
    if (refresh && _tokenFetched) {
      _pendingToken = null;
      _tokenFetched = false;
    }
    return StreamCredentials(apiKey: apiKey, userId: userId, token: await _token());
  }

  /// Uses [client], the app's own Stream client of this [kind], rather than building one.
  /// It is never disconnected here.
  ///
  /// For chat and rtc, whose `useChat` and `useVideo` are what an app calls.
  void give(String kind, Object client) {
    _requireStreamIdentity();
    _clients.given[_key(kind)] = client;
  }

  /// The one Stream client of this [kind] for this key and user: the one the app gave, the
  /// one already built, or one [open] builds now. Sessions asking at once share one opening,
  /// and one that fails to open is opened again by the next to ask.
  ///
  /// For chat and rtc.
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
    final T client;
    try {
      client = await entry.opening as T;
    } catch (_) {
      if (identical(_clients.built[key], entry)) {
        _clients.built.remove(key);
      }
      rethrow;
    }
    if (entry.disconnected) {
      throw StateError('the Stream clients were disconnected while this one was opening');
    }
    return client;
  }

  /// Disconnects every Stream client built here, for every user, and none the app gave.
  Future<void> disconnectStream() async {
    final built = [..._clients.built.values];
    _clients.built.clear();
    for (final entry in built) {
      entry.disconnected = true;
    }
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

  Future<String> _token() async {
    if (_pendingToken case final pending?) {
      return pending;
    }
    final provider = token;
    if (provider == null) {
      throw ArgumentError(
        'an apiKey needs a token to go with it: pass token, or use withUser or withGuest',
      );
    }
    final pending = Future<String>(provider);
    _pendingToken = pending;
    _tokenFetched = false;
    try {
      final minted = await pending;
      if (identical(_pendingToken, pending)) {
        _tokenFetched = true;
      }
      return minted;
    } catch (_) {
      // A token that could not be had is asked for again next time, not remembered.
      if (identical(_pendingToken, pending)) {
        _pendingToken = null;
      }
      rethrow;
    }
  }

  bool _expireToken() {
    if (_byCustomer || token == null) {
      return false;
    }
    if (_tokenFetched) {
      _pendingToken = null;
      _tokenFetched = false;
    }
    return true;
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

  /// Set by [Backend.disconnectStream], so a session still waiting on the opening is not
  /// handed a client being disconnected.
  bool disconnected = false;
}

/// Drops the token [backend] holds, so the next request asks the provider again, and says
/// whether there is a provider to ask.
bool expireBackendToken(Backend backend) => backend._expireToken();

String _join(String base, String path) =>
    base.endsWith('/') ? '${base.substring(0, base.length - 1)}$path' : '$base$path';
