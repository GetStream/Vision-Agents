@file:Suppress("MagicNumber") // Each brand's colour and corners.

package io.getstream.agents.androidassistant

import android.content.Context
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import io.getstream.chat.android.ai.openui.rendering.OpenUITheme

/**
 * The brand the agent's UI wears: its accent, and how round its corners are.
 *
 * Settings chooses one, and the conversation shows it at once, in the chat and on a call: the
 * agent's UI is drawn in the brand's [openUITheme]. The agent writes the same UI whichever is
 * chosen.
 */
enum class Brand(val title: String, val subtitle: String, val accent: Color, val cornerRadius: Dp) {
    Stream("Stream Default", "Electric blue accent · 16dp rounded cards", Color(0xFF005FFF), 16.dp),
    Ember("Ember Hospitality", "Warm terracotta accent · 10dp compact cards", Color(0xFFD95D24), 10.dp),
    Nordic("Nordic Mobility", "Crisp teal accent · 6dp architectural edges", Color(0xFF0F766E), 6.dp);

    /** The OpenUI theme in this brand, over the Material one. Smaller parts round less, in proportion. */
    @Composable
    fun openUITheme(): OpenUITheme {
        val theme = OpenUITheme.fromMaterial()
        return theme.copy(
            colors = theme.colors.copy(accent = accent, info = accent),
            radius = theme.radius.copy(small = cornerRadius / 2, medium = cornerRadius * 0.75f, large = cornerRadius),
        )
    }

    companion object {
        private const val PREFERENCES = "juno"
        private const val KEY = "brand"

        /** The brand chosen on this phone, Stream's until one is. */
        fun stored(context: Context): Brand {
            val name = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE).getString(KEY, null)
            return entries.firstOrNull { it.name == name } ?: Stream
        }

        fun store(context: Context, brand: Brand) {
            context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE).edit().putString(KEY, brand.name).apply()
        }
    }
}
