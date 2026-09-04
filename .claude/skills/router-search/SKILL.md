---
name: router-search
description: What the search router can be asked for, what each provider calls it, and what it refuses to fake. Read before adding a search option or a search provider.
---

# Routing search

The per-modality half of [router-interface](../router-interface/SKILL.md). The vocabulary is
[`options.Search`](../../../acceleration/internal/options/options.go); the providers are under
[`internal/search`](../../../acceleration/internal/search) and are declared in
[`router.yaml`](../../../acceleration/internal/routing/router.yaml).

Search has no socket and no recording: a question and its answer are one round trip, so it is a
plain `POST /v1/search`. What separates the providers is whether a model writes the answer
before it comes back. The low-latency tier returns pages for the conversation's own model to
read; the high-quality tier pays a second model, and the seconds it takes, to have the sentence
written for it.

`depth` is not a term, it is the route. `instant` and `fast` pick the low-latency tier and
`standard` and `deep` the high-quality one, because asking for a deep answer is asking for the
tier that reads pages before answering rather than the one that hands them over.

## The top five, and what each calls the same thing

| Option | Exa | Tavily | Perplexity | Brave | Linkup |
| --- | --- | --- | --- | --- | --- |
| `depth` | `type`: `fast`, `auto`, `neural` | `search_depth`: `basic`, `advanced` | `sonar` vs `sonar-pro` | one tier | `depth`: `standard`, `deep` |
| `results` | `numResults` | `max_results` | `max_results` | `count` | — |
| `include_domains` | `includeDomains` | `include_domains` | `search_domain_filter` | `site:` in the query | `includeDomains` |
| `exclude_domains` | `excludeDomains` | `exclude_domains` | `-domain` in the filter | `site:` in the query | `excludeDomains` |
| `category` | `category` | `topic`: general, news, finance | — | `result_filter` | — |
| `max_age_hours` | `startPublishedDate` | `days` | `search_recency_filter` | `freshness` | `fromDate` |
| `location` | `userLocation` | `country` | `web_search_options.user_location` | `country` | — |
| `contents` | `contents: {text, highlights, summary}` | `include_raw_content`, `include_answer` | citations only | extra snippets | `outputType` |
| `output_schema` | `/answer` with `outputSchema` | — | `response_format` | — | `structuredOutputSchema` |

What the table is saying:

- **Recency is expressed five ways.** A date floor, a day count, a named window, a freshness
  code. `max_age_hours` is the one thing they can all be computed from, and zero means force a
  live crawl rather than trust a cached page.
- **A domain filter is not universal.** Brave has no parameter for it, only the `site:` operator
  in the query, and that is a different thing: it changes the query the user asked.
- **Only some providers return the extract.** `contents` asks for the text a model actually
  reads. Perplexity returns citations and the answer it wrote; there is no per-hit text to ask
  for.

## What the router refuses to fake

The same `Terms()` mechanism as the other modalities, and search is where it is most visible
today. `exa/fast` and `exa/auto` declare `[domains, category, recency, location, contents]`
because [`exa.go`](../../../acceleration/internal/search/exa/exa.go) sends all five. The Tavily
and Perplexity clients send only the query, the depth and the result count, so they declare
nothing — which means a request that filters by domain is not routed to them.

That is the honest state, not the finished state: Tavily and Perplexity both have the parameters
and our clients do not send them yet. Closing that gap means sending them in the client first
and declaring them second. Declaring a term to widen the candidate set is the one thing this
design exists to prevent, and a search that quietly ignored the domain filter is exactly the
failure it prevents: the caller cannot tell from the results that the filter was dropped.

A term nothing can serve is a 400 naming it. For a voice agent that is the right answer — an
agent that says "I could not restrict that to your own documentation" is more useful than one
that reads out a stranger's blog post.

## Adding an option

1. A field on `options.Search`, with `Merge`, plus a `Term` and a line in `Terms()` unless it
   picks the route the way `depth` does.
2. The same field on `SearchOptions` in
   [`openapi.yaml`](../../../acceleration/api/openapi.yaml), then regenerate all three clients.
3. Send it in each provider that takes it, translating into that vendor's shape — a cache age
   into a date floor, a country into whatever it calls a country — and declare it in
   `supports:`.
4. A test that a provider which does not send it is not chosen when it is asked for.
