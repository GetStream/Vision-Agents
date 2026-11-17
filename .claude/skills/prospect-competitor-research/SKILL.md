---
name: prospect-competitor-research
description: "Research a prospect's top 5 competitors and write a dated, sourced report: press releases, latest blog posts, changelogs, new features, where each competitor is ahead, and who is shipping AI features and how those features surface in the product. Use when someone asks for competitive research, a competitor analysis, a market scan, or call prep for a prospect or customer, given a company name and website, even if they never say competitors."
---

# Prospect competitor research

Write a report a seller can read before a call: who the prospect competes with, what those competitors shipped lately, where they are ahead, and how they are putting AI into their apps. Every claim has a date and a link.

## Collect the brief

Ask only for what is missing. Name and website are required.

| Input | Default |
| --- | --- |
| Prospect name | required |
| Website | required |
| Competitors | Research the top 5 |
| Focus | The prospect's core product |
| Window | News and posts from the last 6 months, AI features from the last 12 |

## Read the prospect first

Read their homepage, product pages, pricing, about page, and newsroom. Note what they sell, to whom, on which platforms (web, iOS, Android), and the words they use for the product. Then search news for the last 6 months: funding, launches, acquisitions, leadership changes. Note any AI features they already ship, so the gap analysis has a baseline.

## Pick the five competitors

A competitor is a product the same buyer would put on the same shortlist for the same job. Find candidates from G2 and Capterra categories, "alternatives to" and "vs" pages, the prospect's own comparison pages, and analyst lists. Pick five: usually the two or three incumbents plus the challengers growing fastest. Write one line per competitor saying why it made the list. If the user named competitors, use theirs and do not swap any out.

## Research each competitor

Research the five in parallel: launch one subagent per competitor with the brief below and the window, and ask each to return its notes in the shape of the per-competitor section of the report. Read the notes critically; drop anything without a source.

For each competitor, read:

- **Changelog and release notes**: the product's changelog, docs "what's new", and App Store and Play Store release notes.
- **Blog**: the five most recent posts, with dates and one line each.
- **Press**: their newsroom and press wire releases (PR Newswire, Business Wire) in the window.
- **Product and pricing**: new tiers, add-ons, and what moved behind a paywall.
- **Signals**: funding, acquisitions, partnerships, and job posts that name AI, LLM, or agent roles.

For every AI feature, record:

- What it does, in one line, and when it shipped.
- Where it lives in the app: chat assistant, copilot sidebar, inline suggestions, search, voice or video agent, autonomous agent, or back office only.
- Who it is for: end users, agents or staff, admins, developers.
- How it is sold: included, paid add-on, usage-based, beta, waitlist.
- The model or provider when they say so (OpenAI, Anthropic, Google, in-house). Do not guess.

## Evidence rules

- Prefer primary sources: their changelog, docs, blog, and newsroom over third-party write-ups.
- Every row and claim gets a date and a link. No date found means say "undated".
- Mark inference as inference. "Nothing found in the window" is a finding; do not fill gaps.
- Use the date a feature shipped, not the date an article mentioned it.

## Write the report

Put it in a new directory named for the prospect, outside any repo, as `competitor-research.md`, and say the path. Use this shape:

```markdown
# {Prospect}: competitor research ({date})

## Summary
Five bullets: the biggest moves, where the prospect is behind, the AI trend across the set.

## Prospect snapshot
What they sell, to whom, platforms, recent news, AI they already ship.

## Competitors
| Competitor | Why it's on the list | Size or stage | Latest notable move |

## Where competitors are ahead
One row per gap: the capability, who has it, since when, and the source. Group by theme
(AI, platform coverage, pricing, integrations, enterprise).

## AI feature tracker
| Company | Feature | Shipped | Where in the app | Who it's for | How it's sold | Model | Source |
Include the prospect as the first rows. Then a paragraph on the patterns: which surfaces
are becoming table stakes, who is moving fastest, and what nobody has shipped yet.

## Per competitor
### {Competitor}
- Positioning in one line
- New features in the window (dated, linked)
- AI features (as in the tracker, plus how they are introduced in the UI)
- Latest blog posts (5, dated)
- Press releases (dated)
- Signals: funding, hiring, partnerships

## Timeline
Every dated event across the five competitors, newest first.

## Openings for Stream
Where Stream Chat, Video, Moderation, or Vision Agents would close a gap the competitors
have opened, tied to rows above. Three to five bullets.

## Sources
Every link, grouped by company.
```

## Before you send it

- Five competitors, each with a reason, or the five the user named.
- Every feature, post, and press release has a date and a link, and falls inside the window or says it does not.
- Every AI feature says where it lives in the app and how it is sold.
- "Where competitors are ahead" and "Openings for Stream" point at rows in the report, not at general claims.
- The prospect's own AI features are in the tracker, so the gaps are real gaps.

Then send the path and the summary bullets.
