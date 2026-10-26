package io.getstream.visionagents.androidassistant

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Edge to edge, so the conversation scrolls on under the composer and the system bars.
        enableEdgeToEdge()
        setContent {
            AssistantTheme { AssistantApp() }
        }
    }
}
