import 'dart:convert';

/// Hands over the token to present, from the app's backend.
///
/// Asked once and its answer kept. It is asked again when the router refuses the token with a
/// 401, when a session socket is refused, when the token is within a minute of the expiry it
/// carries, and when Stream Chat or Stream Video say theirs expired. Requests that need a token
/// while it is being fetched wait for the same fetch rather than asking again.
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
  }) : this._(
         url ?? defaultUrl,
         customerId,
         apiKey,
         userId,
         token,
         token == null ? null : _Tokens(token),
         _StreamClients(),
       );

  Backend._(
    this.url,
    this.customerId,
    this.apiKey,
    this.userId,
    this.token,
    this._tokens,
    this._clients,
  ) {
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

  /// What [token] answered, kept for every copy that asks with the same provider.
  final _Tokens? _tokens;

  final _StreamClients _clients;

  bool get _byCustomer => customerId.isNotEmpty;

  /// Whether this holds the app's key and no token yet, which is how a guest is asked for
  /// before there is one: `Stream-Auth-Type: anonymous` and no token.
  bool get anonymous => !_byCustomer && token == null;

  /// The same router, asked by somebody else. Shares this one's Stream clients, which are
  /// kept per user. A new [token] provider starts with no token kept; without one, the token
  /// this one holds is shared.
  Backend withUser(String userId, {TokenProvider? token}) => Backend._(
    url,
    customerId,
    apiKey,
    userId,
    token ?? this.token,
    token == null ? _tokens : _Tokens(token),
    _clients,
  );

  /// Drops the token the provider last answered, so the next request asks it again.
  ///
  /// [refused] is the token the router turned down. Only that one is forgotten: when a newer
  /// one is already held or being fetched, because another refused request got there first,
  /// it is kept. Otherwise requests refused together would each throw away the token the one
  /// before had fetched, and a refusal arriving late would throw away a good one.
  ///
  /// Answers whether a request sent again would carry a token asked for afresh, which is what
  /// a retry after a 401 is for. It would not by customer id, where the router is never shown
  /// the token, nor with no provider at all.
  bool expireToken([String? refused]) {
    final tokens = _tokens;
    if (_byCustomer || tokens == null) {
      return false;
    }
    tokens.expire(refused);
    return true;
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
  /// Stream's clients ask again only when the token they hold has expired, so the one kept
  /// here is no answer to them: [refresh] asks the provider for a new one. A fetch already
  /// under way is that new one, so it is waited for rather than started over.
  Future<StreamCredentials> streamCredentials({bool refresh = false}) async {
    _requireStreamIdentity();
    if (refresh) {
      _tokens!.refresh();
    }
    return StreamCredentials(apiKey: apiKey, userId: userId, token: await _token());
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
    final tokens = _tokens;
    if (tokens == null) {
      throw ArgumentError(
        'an apiKey needs a token to go with it: pass token, or use withUser or withGuest',
      );
    }
    return tokens.get();
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

/// The token one provider answered, and the fetch under way, if any.
final class _Tokens {
  _Tokens(this.provider);

  final TokenProvider provider;
  String? _fetched;
  Future<String>? _fetching;

  Future<String> get() {
    if (_fetched case final token? when !_expiresSoon(token)) {
      return Future.value(token);
    }
    return _fetching ??= _fetch();
  }

  void expire(String? refused) {
    final newer = refused != null && (_fetched != null ? _fetched != refused : _fetching != null);
    if (!newer) {
      _fetched = null;
      _fetching = null;
    }
  }

  void refresh() {
    if (_fetching == null) {
      _fetched = null;
    }
  }

  Future<String> _fetch() {
    late final Future<String> fetching;
    // Kept only if nothing expired the token while it was fetched: otherwise it is the one
    // that was just refused.
    fetching = Future(provider).then(
      (token) {
        if (identical(_fetching, fetching)) {
          _fetched = token;
          _fetching = null;
        }
        return token;
      },
      onError: (Object error, StackTrace stack) {
        if (identical(_fetching, fetching)) {
          _fetching = null;
        }
        Error.throwWithStackTrace(error, stack);
      },
    );
    return fetching;
  }
}

/// How long before the expiry a token carries it is fetched again rather than sent.
const _expiryMargin = Duration(seconds: 60);

/// Whether [token] says it expires within the margin.
///
/// Read rather than trusted: nothing is verified here, the point is only to not open a socket
/// with a token the router is about to refuse, since a refused socket does not say why. A
/// token that is not a JWT, or carries no expiry, never expires as far as this is concerned.
bool _expiresSoon(String token) {
  final parts = token.split('.');
  if (parts.length != 3) {
    return false;
  }
  try {
    final claims = jsonDecode(utf8.decode(base64Url.decode(base64Url.normalize(parts[1]))));
    if (claims case {'exp': final num exp}) {
      final expiry = DateTime.fromMillisecondsSinceEpoch((exp * 1000).round());
      return expiry.difference(DateTime.now()) < _expiryMargin;
    }
  } on FormatException {
    // Not a JWT this can read.
  }
  return false;
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
