@file:Suppress("MagicNumber") // A palette of colour literals.

package io.getstream.visionagents.android.demo

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/**
 * The look Stream's AI components have out of the box, light and dark: they read their colours
 * from the Material theme, and this is the palette Stream Chat's AI sample gives them.
 */
@Composable
fun DemoTheme(dark: Boolean = isSystemInDarkTheme(), content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = if (dark) Dark else Light, content = content)
}

private val Blue600 = Color(0xFF0169CC)
private val Blue400 = Color(0xFF339CFF)
private val White = Color(0xFFFFFFFF)
private val Grey100 = Color(0xFFF3F3F3)
private val Grey200 = Color(0xFFECECEC)
private val Grey500 = Color(0xFF5D5D5D)
private val Grey600 = Color(0xFF424242)
private val Grey700 = Color(0xFF2F2F2F)
private val Grey800 = Color(0xFF212121)
private val Grey900 = Color(0xFF0D0D0D)
private val Red500 = Color(0xFFE02E2A)
private val BlueGrey40 = Color(0xFF6B7280)
private val BlueGrey80 = Color(0xFF9CA3AF)

private val Light = lightColorScheme(
    primary = Blue600,
    onPrimary = White,
    secondary = BlueGrey40,
    onSecondary = White,
    background = White,
    onBackground = Grey900,
    surface = White,
    onSurface = Grey900,
    surfaceVariant = Grey200,
    onSurfaceVariant = Grey500,
    surfaceContainer = Grey100,
    surfaceContainerHigh = Grey200,
    outline = Grey900.copy(alpha = 0.1f),
    outlineVariant = Grey900.copy(alpha = 0.1f),
    error = Red500,
    onError = White,
)

private val Dark = darkColorScheme(
    primary = Blue400,
    onPrimary = White,
    secondary = BlueGrey80,
    onSecondary = White,
    background = Grey800,
    onBackground = White,
    surface = Grey800,
    onSurface = White,
    surfaceVariant = Grey600,
    onSurfaceVariant = Grey100,
    surfaceContainer = Grey700,
    surfaceContainerHigh = Grey600,
    outline = White.copy(alpha = 0.15f),
    outlineVariant = White.copy(alpha = 0.15f),
    error = Red500,
    onError = White,
)
