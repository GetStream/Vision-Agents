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
    }
}

rootProject.name = "stream-agents-kotlin"

include(":core")
