plugins {
    // AGP 9 compiles Kotlin itself, so the Kotlin Android plugin is not applied.
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "io.getstream.visionagents.rtc"
    compileSdk = 37
    defaultConfig {
        minSdk = 24
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

base {
    archivesName = "vision-agents-rtc"
}

kotlin {
    explicitApi()
}

dependencies {
    api(project(":core"))
    api(libs.stream.video.core)
    api(libs.stream.chat.client)
    api(platform(libs.compose.bom))
    implementation(libs.stream.video.compose)
    implementation(libs.compose.foundation)
    implementation(libs.compose.material3)
    implementation(libs.lifecycle.compose)
}
