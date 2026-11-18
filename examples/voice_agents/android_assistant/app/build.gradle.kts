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

/** Puts the file that describes the app's components among the app's assets. */
abstract class ComponentDefinitions : DefaultTask() {
    @get:InputFile
    @get:PathSensitive(PathSensitivity.NONE)
    abstract val source: RegularFileProperty

    @get:OutputDirectory
    abstract val assets: DirectoryProperty

    @TaskAction
    fun copy() {
        source.get().asFile.copyTo(assets.get().file(source.get().asFile.name).asFile, overwrite = true)
    }
}

// The agent's own description of the components it can place in a reply, which its instructions
// are written from too: one file for the agent and every app that runs it.
val componentDefinitions = tasks.register<ComponentDefinitions>("componentDefinitions") {
    source = rootProject.layout.projectDirectory.file("../swift_assistant/app/SwiftAssistant/UIComponents.json")
}

androidComponents {
    onVariants { variant ->
        variant.sources.assets?.addGeneratedSourceDirectory(componentDefinitions, ComponentDefinitions::assets)
    }
}

android {
    namespace = "io.getstream.agents.androidassistant"
    compileSdk = 37
    defaultConfig {
        applicationId = "io.getstream.visionagents.androidassistant"
        minSdk = 24
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        // localhost is the phone's own, which `adb reverse tcp:8080 tcp:8080` and
        // `adb reverse tcp:8098 tcp:8098` hand on to the Mac's, on a device and an emulator alike.
        buildConfigField("String", "ROUTER_URL", "\"${setting("assistant.routerUrl", "ASSISTANT_ROUTER_URL", "http://localhost:8080")}\"")
        buildConfigField("String", "BACKEND_URL", "\"${setting("assistant.backendUrl", "ASSISTANT_BACKEND_URL", "http://localhost:8098")}\"")
        buildConfigField("String", "STREAM_API_KEY", "\"${setting("assistant.streamApiKey", "STREAM_API_KEY", "")}\"")
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        buildConfig = true
        compose = true
    }
    testOptions {
        unitTests.all { it.useJUnitPlatform() }
    }
}

dependencies {
    implementation(libs.stream.agents.android.core)
    implementation(libs.stream.chat.ai.compose)
    implementation(libs.stream.chat.ai.openui)
    implementation(libs.stream.chat.ai.a2ui)
    implementation(libs.okhttp)
    // The connector sign-in opens in an Auth Tab, or a Custom Tab where the browser has none.
    implementation(libs.browser)
    // The trip map, on OpenStreetMap's data, which needs no key.
    implementation(libs.maplibre)

    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.activity.compose)
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.lifecycle.runtime.compose)
    debugImplementation(libs.compose.ui.tooling)

    testImplementation(libs.kotlin.test)
}
