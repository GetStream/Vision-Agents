plugins {
    // AGP 9 compiles Kotlin itself, so the Kotlin Android plugin is not applied.
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.serialization)
    `maven-publish`
    signing
}

android {
    namespace = "io.getstream.agents.android.core"
    compileSdk = 37
    defaultConfig {
        minSdk = 24
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    publishing {
        singleVariant("release") { withSourcesJar() }
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
    archivesName = "stream-agents-android-core"
}

// .github/workflows/publish-sdk-kotlin.yml publishes this to the Stream repository: Gradle signs
// and stages a Maven tree under build/staged-repo, and a job that never holds the signing key
// uploads it. The version is gradle.properties' unless the workflow passes -Pversion.
publishing {
    publications {
        register<MavenPublication>("release") {
            artifactId = base.archivesName.get()
            afterEvaluate { from(components["release"]) }
            pom {
                name = artifactId
                description = "Android client for Stream Agents."
                url = "https://github.com/GetStream/Vision-Agents/tree/accelerate/sdks/kotlin"
                licenses {
                    license {
                        name = "Apache-2.0"
                        url = "https://github.com/GetStream/Vision-Agents/blob/accelerate/LICENSE"
                    }
                }
                developers {
                    developer {
                        name = "Stream"
                        url = "https://getstream.io"
                    }
                }
                scm {
                    connection = "scm:git:github.com/GetStream/Vision-Agents.git"
                    url = "https://github.com/GetStream/Vision-Agents/tree/accelerate/sdks/kotlin"
                }
            }
        }
    }
    repositories {
        maven {
            name = "streamRepoStaging"
            url = uri(rootProject.layout.buildDirectory.dir("staged-repo"))
        }
    }
}

signing {
    val keyId = providers.environmentVariable("SIGNING_KEY_ID").orNull
    val key = providers.environmentVariable("SIGNING_KEY").orNull
    val password = providers.environmentVariable("SIGNING_PASSWORD").orNull
    if (keyId != null && key != null && password != null) {
        useInMemoryPgpKeys(keyId, key, password)
        sign(publishing.publications)
    }
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
