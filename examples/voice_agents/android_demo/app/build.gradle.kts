import java.util.Properties

plugins {
    // AGP 9 compiles Kotlin itself, so the Kotlin Android plugin is not applied.
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

// The Stream key and the addresses stay out of the source: local.properties, which git
// ignores, or the environment, as CI has neither.
val local = Properties().apply {
    rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use(::load)
}

fun setting(name: String, environment: String, default: String): String =
    local.getProperty(name) ?: providers.environmentVariable(environment).orNull ?: default

android {
    namespace = "io.getstream.visionagents.android.demo"
    compileSdk = 37
    defaultConfig {
        applicationId = "io.getstream.visionagents.android.demo"
        minSdk = 24
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        // 10.0.2.2 is the emulator's name for the Mac's localhost. On a device, use the Mac's
        // address on the network, for example http://192.168.1.20:8080.
        buildConfigField("String", "ROUTER_URL", "\"${setting("demo.routerUrl", "DEMO_ROUTER_URL", "http://10.0.2.2:8080")}\"")
        buildConfigField("String", "BACKEND_URL", "\"${setting("demo.backendUrl", "DEMO_BACKEND_URL", "http://10.0.2.2:8099")}\"")
        buildConfigField("String", "STREAM_API_KEY", "\"${setting("demo.streamApiKey", "STREAM_API_KEY", "")}\"")
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        buildConfig = true
        compose = true
    }
}

dependencies {
    implementation(libs.vision.agents.core)
    implementation(libs.stream.chat.ai.compose)
    implementation(libs.okhttp)

    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.activity.compose)
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.lifecycle.runtime.compose)
    debugImplementation(libs.compose.ui.tooling)
}
