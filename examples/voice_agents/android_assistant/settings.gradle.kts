pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // Stream publishes its Android SDKs here; Maven Central stops at the older releases.
        maven("https://stream-io-repo.com") {
            content { includeGroup("io.getstream") }
        }
        // The sign-in card and AIConversation's contentPadding are not in a release of Stream
        // Chat yet, only in a snapshot.
        maven("https://snapshots.stream-io-repo.com") {
            content { includeGroup("io.getstream") }
        }
    }
}

rootProject.name = "android-assistant"

// The SDK as it is in this checkout, not a published one, as swift_assistant builds against
// sdks/swift/core.
includeBuild("../../../sdks/kotlin") {
    dependencySubstitution {
        substitute(module("io.getstream:vision-agents-core")).using(project(":core"))
    }
}

include(":app")
