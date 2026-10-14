/**
 * semantic-release for @stream-io/vision-agents, run by .github/workflows/release-sdk-js.yml.
 *
 * The repository holds every SDK, the router and the Python packages, so a release reads
 * only commits scoped `sdk-js`: a `feat(router)!:` that happens to touch this directory
 * must not move this package. Tags are `sdk-js-v<version>`, clear of the Python `v*` and
 * the router's `accelerate-v*`.
 */

const scope = "sdk-js";

export default {
  // Every release lands on npm's `beta` dist-tag; `latest` moves only when somebody
  // promotes a version by hand.
  branches: [{ name: "accelerate", channel: "beta" }],
  tagFormat: "sdk-js-v${version}",
  plugins: [
    [
      "@semantic-release/commit-analyzer",
      {
        preset: "conventionalcommits",
        // The highest release among the matching rules wins, and `false` only wins when it
        // is matched first, so the catch-all leads. A commit no custom rule matches would
        // fall through to the default rules and release on any scope.
        releaseRules: [
          { release: false },
          { scope, type: "feat", release: "minor" },
          { scope, type: "fix", release: "patch" },
          { scope, type: "perf", release: "patch" },
          // A breaking change moves the minor while the package is 0.x. Going to 1.0.0 is
          // a decision, made by removing this rule.
          { scope, breaking: true, release: "minor" },
        ],
      },
    ],
    [
      "@semantic-release/release-notes-generator",
      { preset: "conventionalcommits", presetConfig: { scope, scopeOnly: true } },
    ],
    "@semantic-release/npm",
    [
      "@semantic-release/github",
      {
        // A draft, published by the workflow with `--latest=false`, so the repository's
        // latest release stays the Python one.
        draftRelease: true,
        releaseNameTemplate: "@stream-io/vision-agents v<%= nextRelease.version %>",
        // Every commit since the last tag is in context, router ones included, so a comment
        // or label per resolved pull request would land on work that is not in this package.
        successComment: false,
        releasedLabels: false,
        failTitle: "The sdk-js release is failing",
      },
    ],
  ],
};
