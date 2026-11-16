---
name: release-sdks
description: How to release every SDK at once, or one of them - the Release all SDKs workflow, what a published GitHub Release sets off, which SDKs are released another way, the one-time registry setup each needs, and what to check when a release fails. Read before releasing an SDK, adding a release workflow, or publishing a GitHub Release.
---

# Releasing the SDKs

Each SDK has its own version line, tag family and release workflow. `release-sdks.yml`
("Release all SDKs") starts those workflows together, waits and reports each one. It publishes
nothing itself: npm, NuGet, RubyGems and crates.io trust a workflow by its file name, so every
release stays a run of `release-sdk-<sdk>.yml`. Do not turn those into reusable workflows called
from one file, and do not rename them.

| SDK | Package | Registry | Workflow | Version comes from | Tag |
| --- | --- | --- | --- | --- | --- |
| JS | `@stream-io/stream-agents` | npm | `release-sdk-js.yml` | its `sdk-js` commits (semantic-release) | `sdk-js-v*` |
| .NET | `getstream-calypso` | NuGet | `release-sdk-dotnet.yml` | last tag + bump | `sdk-dotnet-v*` |
| Ruby | `getstream-vision-agents` | RubyGems | `release-sdk-ruby.yml` | last tag + bump | `sdk-ruby-v*` |
| Rust | `vision-agents` | crates.io | `release-sdk-rust.yml` | last tag + bump | `sdk-rust-v*` |
| PHP | `getstream/vision-agents` | Packagist, through a split repository | `release-sdk-php.yml` | last tag + bump | `sdk-php-v*` |
| Dart | `stream_agents` | pub.dev | `release-sdk-dart.yml` | its `pubspec.yaml` | `sdk-dart-v*` |
| Kotlin | `io.getstream:stream-agents-android-core` | stream-io-repo.com | `publish-sdk-kotlin.yml` | `gradle.properties` | `sdk-kotlin-v*` |
| Go | `github.com/GetStream/Vision-Agents/sdks/go` | the Go proxy | by hand, on `main` | you | `sdks/go/v*` |
| Swift | `stream-agents` (SwiftPM) | this repository | none | the GitHub Release | `v*` |
| Python | `vision-agents` and the plugins | PyPI | `release.yml` | the GitHub Release | `v*` |

## Release them all

Two ways in, both releasing the tip of `accelerate`:

**By hand.** Actions, "Release all SDKs", "Use workflow from" `accelerate`. Or:

```bash
gh workflow run release-sdks.yml --ref accelerate -f sdks=all -f bump=patch -f dry_run=true
```

- `sdks`: `all`, or a list such as `ruby,rust`. The names are `js`, `dotnet`, `ruby`, `rust`,
  `php`, `dart`.
- `bump`: the part of each SDK's own last version to raise. `minor` takes .NET 0.2.0 to 0.3.0
  and Ruby 0.1.0 to 0.2.0; the SDKs do not share a version.
- `dry_run`: on by default. A dry run works out every version, tests and packages, and
  publishes and tags nothing. Always do one first and read each run's summary: it lists the
  version and the commits going out.

**By publishing a GitHub Release.** A release `vX.Y.Z` of a commit on `accelerate` releases
every SDK for real, raising the part the release raised: `v0.7.0` a minor, `v0.7.1` a patch,
`v1.0.0` a major. The same event publishes the Python packages (`release.yml`), and the tag is
the Swift package's version. A prerelease, a tag that is not `vX.Y.Z`, or a release cut from
another branch starts no SDK. Run the dry run by hand before publishing the release.

The run's summary has one row per SDK with a link to its run. A failed SDK fails the run and
leaves the others as they were: fix it and run "Release all SDKs" again with only that SDK in
`sdks`, or run its own workflow.

## What the bump does not move

- **JS** decides its own version from commits scoped `sdk-js` since the last `sdk-js-v*` tag,
  and releases nothing when there are none (see the `sdk-js` skill).
- **Dart** releases the version committed in `sdks/dart/stream_agents/pubspec.yaml`, if it has
  no `sdk-dart-v*` tag yet. To include Dart, first merge a pull request that raises `version`
  and adds the `CHANGELOG.md` entry. pub.dev takes a package only from a run on a tag, so the
  run from `accelerate` tags the commit and starts a second run of `release-sdk-dart.yml` on
  that tag, which publishes it. If that run fails, run the workflow again from the tag.
- **Kotlin** is not started from here. Merging a pull request that raises `version` in
  `sdks/kotlin/gradle.properties` publishes it (see "Releasing" in `sdks/kotlin/README.md`).
- **Go** is not started from here. It is a copy of `sdks/go` on `main` and a tag on the merge
  commit (see "Releasing" in the `sdk-go` skill).

For one SDK at an exact version (`0.3.0-beta.1`), run that SDK's own workflow with `version`.

## Rules

- Never create a GitHub Release for one SDK. Every published release starts `release.yml` and
  `release-sdks.yml`. JS's release is made as a draft by `GITHUB_TOKEN`, which starts nothing.
- A tag is pushed after the publish succeeds, so a failed release leaves no tag and the next
  run works out the same version. Dart is the exception, as above. Never move or delete a
  released tag.
- The version in `version.rb`, `Cargo.toml` and `package.json` is not maintained. The release
  writes the one it publishes into the package and commits nothing back.
- A job that holds a registry's OIDC token restores no dependency cache.
- A new SDK gets `release-sdk-<sdk>.yml` with `bump`, `version` and `dry_run` inputs, the
  `.github/actions/sdk-version` step for its version, the `refs/heads/accelerate` guard, and a
  line in `release-sdks.yml`'s list. `release-sdk-ruby.yml` is the one to copy.

## One-time setup

The Actions tab offers "Run workflow" only for a file that is also on `main`, the default
branch. Copy `release-sdks.yml` and a new release workflow there to get the button; a run from
`main` does nothing, since every job is guarded to `accelerate`. Until then, start it with
`gh workflow run ... --ref accelerate`.

Each registry has to trust its workflow before the first real release. A dry run needs none of
this.

- **npm, NuGet**: done.
- **RubyGems**: on rubygems.org add a trusted publisher for the gem `getstream-vision-agents`
  (a pending one, while the gem does not exist): repository `GetStream/Vision-Agents`,
  workflow `release-sdk-ruby.yml`.
- **crates.io**: publish the first version by hand with a token (`cargo publish`, then
  `git tag sdk-rust-v<version>`), then add a trusted publisher in the crate's settings:
  `GetStream/Vision-Agents`, workflow `release-sdk-rust.yml`.
- **Packagist**: create the read-only split repository, submit it to Packagist as
  `getstream/vision-agents`, and set the repository variable `SDK_PHP_SPLIT_REPOSITORY`
  (`owner/name`) and the secret `SDK_PHP_SPLIT_TOKEN` (a token that may push to it).
- **pub.dev**: publish the first version of `stream_agents` by hand (see "Releasing" in the
  `sdk-dart` skill) and tag that commit `sdk-dart-v<version>`, then on the package's admin page
  enable publishing from GitHub Actions: repository `GetStream/Vision-Agents`, tag pattern
  `sdk-dart-v{{version}}`, `workflow_dispatch` events allowed.

## When a release fails

- **"did not start"** in the summary: the SDK's workflow is disabled, or GitHub has not seen
  it on `accelerate` yet. The "Start each SDK's release" step's log has the refusal.
- **`sdk-<sdk>-v<version> already exists`**: an exact `version` that is already released.
- **A publish step is refused** (401, 403, "no trusted publisher"): the registry's trust is
  missing, or names another file than the workflow that ran.
- **Published but not tagged**: the tag step failed after the publish. Tag that run's commit
  `sdk-<sdk>-v<version>` by hand, or the next bump tries the same version and the registry
  refuses it.
- **Dart is "already released"**: the pubspec version has a tag. Raise it in a pull request.
- **A release from a GitHub Release did nothing**: the run's summary says why (a prerelease,
  not `vX.Y.Z`, or the tag is not on `accelerate`).
