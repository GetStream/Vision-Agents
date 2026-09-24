package io.getstream.visionagents.core

import io.getstream.visionagents.core.generated.SearchAnswer as AnswerSchema
import io.getstream.visionagents.core.generated.SearchDepth
import io.getstream.visionagents.core.generated.SearchOptions as OptionsSchema
import io.getstream.visionagents.core.generated.SearchRequest
import io.getstream.visionagents.core.generated.SearchResult

/**
 * Looking something up, configured once.
 *
 * Search is the one routed modality a device may reach: a question and its answer are one
 * round trip, and the answer is for whoever asked rather than for the app.
 *
 *     val router = agents.router(config = "healthcare")
 *     val answer = router.search("what changed in the pricing page")
 *
 * Transcription, a voice and a model are not here. Those run over the per-modality socket,
 * which the router refuses a device, so a pipeline of your own belongs to a backend.
 */
public class Router(
    public val backend: Backend,
    /** A stored router config, by name or id. Empty means every call says what it wants. */
    public val config: String = "",
    /** Cost labels carried onto everything routed here, on top of the config's own. */
    public val tags: Map<String, String> = emptyMap(),
) {
    /** Answers one question out of what is true now. */
    public suspend fun search(query: String, options: SearchOptions = SearchOptions()): SearchAnswer {
        val request = SearchRequest(
            query = query,
            configId = config.ifEmpty { null },
            options = options.schema,
            tags = tags.ifEmpty { null },
        )
        val answer = backend.post(
            listOf("v1", "search"),
            SearchRequest.serializer(),
            request,
            AnswerSchema.serializer(),
        )
        return SearchAnswer.of(answer)
    }
}

/**
 * How to find out today's answers.
 *
 * Null means the named router config decides, and the router decides what the config does not.
 * A provider that cannot express a term refuses the request rather than dropping it.
 */
public data class SearchOptions(
    /** A provider/model or a capability shortcut such as `search-fast`. */
    val target: String? = null,
    /**
     * How much work a search is worth: [Depth.Instant] answers from the index in a few hundred
     * milliseconds, [Depth.Deep] reads what it finds and can take tens of seconds.
     */
    val depth: Depth? = null,
    /** How many hits to return. */
    val results: Int? = null,
    /** Only answer from these domains. */
    val includeDomains: List<String> = emptyList(),
    val excludeDomains: List<String> = emptyList(),
    /** The kind of source to prefer, for the providers that classify their index. */
    val category: String? = null,
    /** How stale a cached page may be. Zero forces a live crawl. */
    val maxAgeHours: Int? = null,
    /** Country or region to answer from. */
    val location: String? = null,
) {
    public enum class Depth { Instant, Fast, Standard, Deep }

    internal val schema: OptionsSchema
        get() = OptionsSchema(
            target = target,
            depth = depth?.let { SearchDepth.valueOf(it.name.lowercase()) },
            results = results,
            includeDomains = includeDomains.ifEmpty { null },
            excludeDomains = excludeDomains.ifEmpty { null },
            category = category,
            maxAgeHours = maxAgeHours,
            location = location,
        )
}

/** What a search found. */
public data class SearchAnswer(
    val provider: String,
    val model: String,
    /** The provider's own summary, where it offers one: a sentence to say rather than a page to read. */
    val answer: String,
    /** The sources behind it, most relevant first. */
    val results: List<SearchHit>,
) {
    internal companion object {
        fun of(schema: AnswerSchema): SearchAnswer = SearchAnswer(
            provider = schema.provider,
            model = schema.model,
            answer = schema.answer.orEmpty(),
            results = schema.results.map(SearchHit::of),
        )
    }
}

/** One source a search found. */
public data class SearchHit(
    val title: String,
    val url: String,
    /** The relevant extract, which is what a model reads. */
    val text: String,
    val score: Double,
) {
    internal companion object {
        fun of(schema: SearchResult): SearchHit = SearchHit(
            title = schema.title.orEmpty(),
            url = schema.url,
            text = schema.text.orEmpty(),
            score = schema.score?.toDouble() ?: 0.0,
        )
    }
}
