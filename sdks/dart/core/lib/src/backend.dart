/// Hands over the token to present, asked for on every request and socket handshake.
///
/// A function rather than a value so a long-lived client never wakes up holding an expired
/// token. Caching one until it is about to expire is the provider's business.
typedef TokenProvider = Future<String> Function();

/// Where the router is and who is asking.
///
/// A device holds no secret worth having, so there is no API secret here. There are two ways
/// to be let in, and which one a deployment takes is its `ROUTER_AUTH_MODE`, not a choice
/// made per request:
///
/// * `customerId`, for a router in `noauth` mode, which believes the header. `userId` names
///   the end user, since there is no token to name them.
/// * `apiKey` with a `token` a backend minted for this user, or a guest's token, for a router
///   verifying tokens.
///
/// Every request and handshake is sent as a device's: `Stream-Auth-Type: jwt` on requests,
/// and no auth type at all on a socket, which the router reads the same way.
final class Backend {
  Backend({
    required this.url,
    this.customerId = '',
    this.apiKey = '',
    this.userId = '',
    this.token,
  }) {
    if (customerId.isEmpty && apiKey.isEmpty) {
      throw ArgumentError('a backend needs a customerId, or an apiKey with a token');
    }
    if (apiKey.isNotEmpty && token == null) {
      throw ArgumentError('an apiKey is only half a credential: pass the token as well');
    }
  }

  /// The router's base URL.
  final Uri url;

  /// Which tenant's agents these are, for a router that believes the header.
  final String customerId;

  /// The public half of a credential, for a router verifying tokens.
  final String apiKey;

  /// Who is asking, for a router that believes the header. With a token the token says so.
  final String userId;

  final TokenProvider? token;

  /// The same router, asked by somebody else.
  Backend withUser(String userId, {TokenProvider? token}) => Backend(
    url: url,
    customerId: customerId,
    apiKey: apiKey,
    userId: userId,
    token: token ?? this.token,
  );

  /// What every request carries.
  ///
  /// `Stream-Auth-Type: jwt` declares a device, and is sent even to an open router, which
  /// would otherwise take a caller with no proxy in front of it for a backend.
  Future<Map<String, String>> headers() async {
    if (apiKey.isEmpty) {
      return {
        'X-Customer-Id': customerId,
        'Stream-Auth-Type': 'jwt',
        if (userId.isNotEmpty) 'X-Stream-User-Id': userId,
      };
    }
    return {
      'X-Api-Key': apiKey,
      'Authorization': 'Bearer ${await token!()}',
      'Stream-Auth-Type': 'jwt',
    };
  }

  /// The URL for a request path, with the query values that are set.
  Uri requestUri(String path, [Map<String, String?> query = const {}]) {
    final present = {for (final MapEntry(:key, :value) in query.entries) key: ?value};
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
    final credentials = apiKey.isEmpty
        ? {'customer_id': customerId, if (userId.isNotEmpty) 'user_id': userId}
        : {'api_key': apiKey, 'token': await token!()};
    return url.replace(
      scheme: url.scheme == 'https' ? 'wss' : 'ws',
      path: _join(url.path, path),
      queryParameters: {...query, ...credentials},
    );
  }
}

String _join(String base, String path) =>
    base.endsWith('/') ? '${base.substring(0, base.length - 1)}$path' : '$base$path';
