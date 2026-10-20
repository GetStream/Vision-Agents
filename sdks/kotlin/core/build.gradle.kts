plugins {
    // AGP 9 compiles Kotlin itself, so the Kotlin Android plugin is not applied.
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.serialization)
}

android {
    namespace = "io.getstream.visionagents.android.core"
    compileSdk = 37
    defaultConfig {
        minSdk = 24
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    testOptions {
        unitTests.all { test ->
            test.useJUnitPlatform()
            // Pointing the live tests at a router is a different test run, not an up-to-date one.
            test.inputs.property("live", providers.environmentVariable("VISION_AGENTS_URL").orElse(""))
            // A socket bug otherwise hangs the build instead of failing it.
            test.systemProperty("junit.jupiter.execution.timeout.default", "60s")
            test.testLogging {
                events("passed", "failed", "skipped")
                exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
            }
        }
    }
}

base {
    archivesName = "vision-agents-core"
}

kotlin {
    explicitApi()
}

dependencies {
    api(libs.coroutines.core)
    api(libs.serialization.json)
    api(libs.okhttp)
    api(libs.stream.chat.client)
    api(libs.stream.video.core)
    implementation(libs.ktor.client.core)
    implementation(libs.ktor.client.okhttp)
    implementation(libs.ktor.client.websockets)

    testImplementation(libs.kotlin.test)
    testImplementation(libs.coroutines.test)
    testImplementation(libs.ktor.server.cio)
    testImplementation(libs.ktor.server.websockets)
}
