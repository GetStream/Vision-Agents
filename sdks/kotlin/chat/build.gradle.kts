plugins {
    // AGP 9 compiles Kotlin itself, so the Kotlin Android plugin is not applied.
    alias(libs.plugins.android.library)
}

android {
    namespace = "io.getstream.visionagents.chat"
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
    archivesName = "vision-agents-chat"
}

kotlin {
    explicitApi()
}

dependencies {
    api(project(":core"))
    api(libs.stream.chat.client)
}
