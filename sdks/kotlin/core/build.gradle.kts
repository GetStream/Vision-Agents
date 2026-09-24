import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
    `java-library`
}

base {
    archivesName = "vision-agents-core"
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    explicitApi()
    compilerOptions {
        jvmTarget = JvmTarget.JVM_17
    }
}

dependencies {
    api(libs.coroutines.core)
    api(libs.serialization.json)
    api(libs.okhttp)
    implementation(libs.ktor.client.core)
    implementation(libs.ktor.client.okhttp)
    implementation(libs.ktor.client.websockets)

    testImplementation(libs.kotlin.test)
    testImplementation(libs.coroutines.test)
    testImplementation(libs.ktor.server.cio)
    testImplementation(libs.ktor.server.websockets)
}

tasks.test {
    useJUnitPlatform()
    // Pointing the live tests at a router is a different test run, not an up-to-date one.
    inputs.property("live", providers.environmentVariable("VISION_AGENTS_URL").orElse(""))
    // A socket bug otherwise hangs the build instead of failing it.
    systemProperty("junit.jupiter.execution.timeout.default", "60s")
    testLogging {
        events("passed", "failed", "skipped")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
