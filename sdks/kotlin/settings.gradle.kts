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
    }
}

rootProject.name = "vision-agents-kotlin"

include(":core")

// The Android modules need an Android SDK to configure at all. `-PcoreOnly` is how the plain
// JVM image builds and tests core; it is asked for rather than guessed from the environment,
// so a CI job missing its SDK fails instead of quietly skipping half the build.
if (!providers.gradleProperty("coreOnly").isPresent) {
    include(":ui", ":rtc")
}
