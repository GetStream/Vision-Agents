import 'generated/api.dart' as api;

/// How much work a search is worth.
enum SearchDepth {
  /// From the index, in a few hundred milliseconds.
  instant,
  fast,
  standard,

  /// Reads what it finds, and can take tens of seconds.
  deep,
}

/// How to find out today's answers.
///
/// Everything is optional because everything has an answer already: a named router config
/// decides what this does not say, and the router decides what the config does not. A
/// provider that cannot express a term refuses the request rather than dropping it.
final class SearchOptions {
  const SearchOptions({
    this.target,
    this.providers = const [],
    this.depth,
    this.results,
    this.includeDomains = const [],
    this.excludeDomains = const [],
    this.category,
    this.maxAgeHours,
    this.location,
  });

  /// A provider/model or a capability shortcut such as `search-fast`.
  final String? target;

  /// Where to try, in order, which wins over [target] and [depth].
  final List<String> providers;
  final SearchDepth? depth;

  /// How many hits to return.
  final int? results;

  /// Only answer from these domains.
  final List<String> includeDomains;
  final List<String> excludeDomains;

  /// The kind of source to prefer, for the providers that classify their index.
  final String? category;

  /// How stale a cached page may be. Zero forces a live crawl.
  final int? maxAgeHours;

  /// Country or region to answer from.
  final String? location;

  api.SearchOptions? get _wire {
    final wire = api.SearchOptions(
      target: target,
      providers: providers.isEmpty ? null : providers,
      depth: depth?.name,
      results: results,
      includeDomains: includeDomains.isEmpty ? null : includeDomains,
      excludeDomains: excludeDomains.isEmpty ? null : excludeDomains,
      category: category,
      maxAgeHours: maxAgeHours,
      location: location,
    );
    return wire.toJson().isEmpty ? null : wire;
  }
}

/// What a search found.
final class SearchAnswer {
  const SearchAnswer({
    required this.provider,
    required this.model,
    this.answer = '',
    this.results = const [],
  });

  final String provider;
  final String model;

  /// The provider's own summary, where it offers one: a sentence to say rather than a page.
  final String answer;

  /// The sources behind it, most relevant first.
  final List<SearchHit> results;
}

/// One source a search found.
final class SearchHit {
  const SearchHit({required this.url, this.title = '', this.text = '', this.score = 0});

  final String url;
  final String title;

  /// The relevant extract, which is what a model reads.
  final String text;
  final double score;
}

/// Looking something up, configured once.
///
/// Search is the one routed modality a device may reach: a question and its answer are one
/// round trip, and the answer is for whoever asked. Transcription, a voice and a model on
/// their own run over a socket the router refuses to a device.
final class SearchRouter {
  SearchRouter(this._operations, {this.config = '', this.tags = const {}});

  final api.Operations _operations;

  /// A stored router config, by name or id. Without one, every call says what it wants.
  final String config;

  /// Cost labels carried onto everything routed here, on top of the config's own.
  final Map<String, String> tags;

  /// Answers one question out of what is true now.
  Future<SearchAnswer> search(String query, [SearchOptions options = const SearchOptions()]) async {
    final answer = await _operations.search(
      body: api.SearchRequest(
        query: query,
        configId: config.isEmpty ? null : config,
        options: options._wire,
        tags: tags.isEmpty ? null : tags,
      ),
    );
    return SearchAnswer(
      provider: answer.provider,
      model: answer.model,
      answer: answer.answer ?? '',
      results: [
        for (final hit in answer.results)
          SearchHit(
            url: hit.url,
            title: hit.title ?? '',
            text: hit.text ?? '',
            score: hit.score ?? 0,
          ),
      ],
    );
  }
}
