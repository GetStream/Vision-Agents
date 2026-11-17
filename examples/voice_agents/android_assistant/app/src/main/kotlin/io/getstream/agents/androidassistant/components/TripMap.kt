@file:Suppress("MagicNumber") // The map's size, its pins and how they drop.

package io.getstream.agents.androidassistant.components

import android.annotation.SuppressLint
import android.graphics.RectF
import android.view.MotionEvent
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.view.doOnLayout
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import io.getstream.chat.android.ai.compose.generativeui.AIUIComponent
import io.getstream.chat.android.ai.compose.generativeui.AIUIComponentDefinition
import io.getstream.chat.android.ai.openui.rendering.OpenUITheme
import kotlinx.coroutines.delay
import org.maplibre.android.MapLibre
import org.maplibre.android.camera.CameraUpdateFactory
import org.maplibre.android.geometry.LatLngBounds
import org.maplibre.android.maps.MapLibreMap
import org.maplibre.android.maps.MapLibreMapOptions
import org.maplibre.android.maps.MapView
import org.maplibre.android.maps.Style
import org.maplibre.android.style.expressions.Expression
import org.maplibre.android.style.layers.CircleLayer
import org.maplibre.android.style.layers.Property
import org.maplibre.android.style.layers.PropertyFactory
import org.maplibre.android.style.layers.SymbolLayer
import org.maplibre.android.style.sources.GeoJsonSource
import org.maplibre.geojson.Feature
import org.maplibre.geojson.FeatureCollection
import org.maplibre.geojson.Point

/** A plan's map, which asks about a place when its pin is tapped. */
internal fun tripMap(definition: AIUIComponentDefinition) = AIUIComponent(definition) { arguments, context ->
    val places = arguments.objects("places").mapNotNull(TripPlace::from)
    TripMap(places, dropsPins = context.isStreaming) { place ->
        context.emit("select_place", label = "Tell me more about ${place.name}", payload = mapOf("name" to place.name))
    }
}

internal data class TripPlace(val name: String, val latitude: Double, val longitude: Double) {
    companion object {
        /** The place the agent wrote, or null while it is incomplete or off the globe. */
        fun from(json: Map<*, *>): TripPlace? {
            val name = (json["name"] as? String)?.takeIf { it.isNotEmpty() } ?: return null
            val latitude = json.number("latitude")?.takeIf { it in -90.0..90.0 } ?: return null
            val longitude = json.number("longitude")?.takeIf { it in -180.0..180.0 } ?: return null
            return TripPlace(name, latitude, longitude)
        }
    }
}

// OpenFreeMap's styles, light and dark, on OpenStreetMap's data, which take no key.
private const val LIGHT_STYLE = "https://tiles.openfreemap.org/styles/liberty"
private const val DARK_STYLE = "https://tiles.openfreemap.org/styles/dark"
private const val SOURCE = "trip-places"
private const val PINS = "trip-pins"
private const val NAMES = "trip-names"

/**
 * Told whether a finger is on a trip map, so a screen-wide swipe, such as the drawer's, leaves the
 * map's pans and pinches to the map.
 */
internal val LocalMapTouch = staticCompositionLocalOf<(Boolean) -> Unit> { {} }

/**
 * A plan's places on a map, in the brand's accent, light or dark with the theme. Tapping a pin asks
 * about the place. Drawn as the agent writes it, the map drops the places' pins one after another.
 */
@Composable
private fun TripMap(places: List<TripPlace>, dropsPins: Boolean, onSelect: (TripPlace) -> Unit) {
    val theme = OpenUITheme.current
    val dark = theme.colors.surface.luminance() < 0.5f
    val pins = PinColors(
        fill = theme.colors.accent.toArgb(),
        ring = theme.colors.surface.toArgb(),
        label = theme.colors.textPrimary.toArgb(),
    )
    val context = LocalContext.current
    val touchSlop = with(LocalDensity.current) { 12.dp.toPx() }
    val margin = with(LocalDensity.current) { 8.dp.roundToPx() }
    val shownPlaces by rememberUpdatedState(places)
    val select by rememberUpdatedState(onSelect)
    val touching by rememberUpdatedState(LocalMapTouch.current)
    val shownPins by rememberUpdatedState(pins)
    var pinned by remember { mutableIntStateOf(if (dropsPins) 0 else places.size) }
    var map by remember { mutableStateOf<MapLibreMap?>(null) }
    var style by remember { mutableStateOf<Style?>(null) }

    val view = remember {
        MapLibre.getInstance(context)
        val options = MapLibreMapOptions.createFromAttributes(context)
            // A TextureView, which scrolls and clips with the conversation as a SurfaceView does not.
            .textureMode(true)
            // OpenStreetMap's and OpenFreeMap's credits stay behind the (i); MapLibre's logo would
            // crowd a map this small.
            .logoEnabled(false)
            .attributionEnabled(true)
            .attributionMargins(intArrayOf(margin, 0, 0, margin))
        MapView(context, options).apply {
            onCreate(null)
            keepGestures { touching(it) }
            showOnFirstFrame()
            getMapAsync { loaded ->
                map = loaded
                loaded.uiSettings.isRotateGesturesEnabled = false
                loaded.uiSettings.isTiltGesturesEnabled = false
                loaded.addOnMapClickListener { point ->
                    val at = loaded.projection.toScreenLocation(point)
                    val area = RectF(at.x - touchSlop, at.y - touchSlop, at.x + touchSlop, at.y + touchSlop)
                    val name = loaded.queryRenderedFeatures(area, PINS, NAMES).firstNotNullOfOrNull { it.getStringProperty("name") }
                    val place = shownPlaces.firstOrNull { it.name == name } ?: return@addOnMapClickListener false
                    select(place)
                    true
                }
            }
        }
    }
    view.FollowLifecycle()
    // A map that leaves under a finger, as a reply that finishes and is drawn again, sees no lift.
    DisposableEffect(view) { onDispose { touching(false) } }

    // The first pin waits for the map to come in.
    LaunchedEffect(places.size) {
        while (pinned < places.size) {
            delay(if (pinned == 0) 600 else 250)
            pinned++
        }
    }
    // The camera is at the places before the first style is set, so a new map never asks for the
    // tiles of the style's own camera, over the whole world.
    LaunchedEffect(map, places) {
        val shown = map ?: return@LaunchedEffect
        val bounds = region(places) ?: return@LaunchedEffect
        view.doOnLayout { shown.moveCamera(CameraUpdateFactory.newLatLngBounds(bounds, 0)) }
    }
    // The style follows the theme on the same map, its pins added again on each.
    LaunchedEffect(map, dark) {
        val shown = map ?: return@LaunchedEffect
        view.doOnLayout {
            // The style before is unusable from here.
            style = null
            shown.setStyle(Style.Builder().fromUri(if (dark) DARK_STYLE else LIGHT_STYLE)) { style = it.withPins(shownPins) }
        }
    }
    LaunchedEffect(style, places, pinned) {
        style?.getSourceAs<GeoJsonSource>(SOURCE)?.setGeoJson(features(places.take(pinned)))
    }
    LaunchedEffect(style, pins) {
        style?.colorPins(pins)
    }
    // A map whose style never loads, as without a network, still shows rather than leave a gap.
    LaunchedEffect(view) {
        delay(3_000)
        view.alpha = 1f
    }

    AndroidView(
        factory = { view },
        modifier = Modifier
            .fillMaxWidth()
            .height(220.dp)
            .clip(RoundedCornerShape(theme.radius.large)),
    )
}

/**
 * Keeps a drag or a pinch that starts on the map on the map, rather than scrolling the
 * conversation, and tells [onTouch] while a finger is on it.
 */
@SuppressLint("ClickableViewAccessibility") // The map handles the touch itself; this only claims it.
private fun MapView.keepGestures(onTouch: (Boolean) -> Unit) {
    setOnTouchListener { touched, event ->
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_POINTER_DOWN -> {
                touched.parent?.requestDisallowInterceptTouchEvent(true)
                onTouch(true)
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> onTouch(false)
        }
        false
    }
}

/**
 * Keeps the map clear, over the card behind it, until it has drawn a frame of its style, rather
 * than showing the empty view's black, or until the map fails to load.
 */
private fun MapView.showOnFirstFrame() {
    alpha = 0f
    getMapAsync { loaded ->
        addOnDidFinishRenderingFrameListener(
            object : MapView.OnDidFinishRenderingFrameListener {
                override fun onDidFinishRenderingFrame(fully: Boolean, frameEncodingTime: Double, frameRenderingTime: Double) {
                    if (loaded.style == null) return
                    removeOnDidFinishRenderingFrameListener(this)
                    animate().alpha(1f).setDuration(150)
                }
            },
        )
    }
    addOnDidFailLoadingMapListener { alpha = 1f }
}

/** Starts, resumes, pauses and stops the map with the screen, and destroys it when it leaves. */
@Composable
private fun MapView.FollowLifecycle() {
    val lifecycle = LocalLifecycleOwner.current.lifecycle
    DisposableEffect(this, lifecycle) {
        var state = Lifecycle.State.CREATED
        fun moveTo(target: Lifecycle.State) {
            if (state < Lifecycle.State.STARTED && target >= Lifecycle.State.STARTED) onStart()
            if (state < Lifecycle.State.RESUMED && target >= Lifecycle.State.RESUMED) onResume()
            if (state >= Lifecycle.State.RESUMED && target < Lifecycle.State.RESUMED) onPause()
            if (state >= Lifecycle.State.STARTED && target < Lifecycle.State.STARTED) onStop()
            state = target
        }
        val observer = LifecycleEventObserver { _, event -> moveTo(event.targetState.coerceAtLeast(Lifecycle.State.CREATED)) }
        lifecycle.addObserver(observer)
        onDispose {
            lifecycle.removeObserver(observer)
            moveTo(Lifecycle.State.CREATED)
            onDestroy()
        }
    }
}

/**
 * A pin's colours: the brand's accent inside a ring of the card's surface, and its name in the
 * theme's text colour on a halo of that surface, so both read on the light map and the dark one.
 */
private data class PinColors(val fill: Int, val ring: Int, val label: Int)

/** Adds the places' pins and names over the map, empty until the places are set. */
private fun Style.withPins(colors: PinColors): Style = apply {
    addSource(GeoJsonSource(SOURCE, FeatureCollection.fromFeatures(emptyList())))
    addLayer(
        CircleLayer(PINS, SOURCE).withProperties(
            PropertyFactory.circleRadius(8f),
            PropertyFactory.circleStrokeWidth(2f),
        ),
    )
    addLayer(
        SymbolLayer(NAMES, SOURCE).withProperties(
            PropertyFactory.textField(Expression.get("name")),
            // A font the style's glyphs have.
            PropertyFactory.textFont(arrayOf("Noto Sans Regular")),
            PropertyFactory.textSize(12f),
            PropertyFactory.textAnchor(Property.TEXT_ANCHOR_TOP),
            PropertyFactory.textOffset(arrayOf(0f, 1f)),
            PropertyFactory.textHaloWidth(1.5f),
        ),
    )
    colorPins(colors)
}

private fun Style.colorPins(colors: PinColors) {
    getLayer(PINS)?.setProperties(PropertyFactory.circleColor(colors.fill), PropertyFactory.circleStrokeColor(colors.ring))
    getLayer(NAMES)?.setProperties(PropertyFactory.textColor(colors.label), PropertyFactory.textHaloColor(colors.ring))
}

private fun features(places: List<TripPlace>): FeatureCollection = FeatureCollection.fromFeatures(
    places.map { place ->
        Feature.fromGeometry(Point.fromLngLat(place.longitude, place.latitude)).apply { addStringProperty("name", place.name) }
    },
)

/**
 * The region that shows every place, with some room around them, or null without places. Places
 * nearer a pole than the map reaches are shown at its edge.
 */
internal fun region(places: List<TripPlace>): LatLngBounds? {
    if (places.isEmpty()) return null
    val south = places.minOf { it.latitude }.coerceIn(-85.0, 85.0)
    val north = places.maxOf { it.latitude }.coerceIn(-85.0, 85.0)
    val west = places.minOf { it.longitude }
    val east = places.maxOf { it.longitude }
    val latitudeSpan = maxOf(0.02, (north - south) * 1.8) / 2
    val longitudeSpan = maxOf(0.02, (east - west) * 1.8) / 2
    val latitude = (south + north) / 2
    val longitude = (west + east) / 2
    return LatLngBounds.from(
        (latitude + latitudeSpan).coerceAtMost(85.0),
        (longitude + longitudeSpan).coerceAtMost(180.0),
        (latitude - latitudeSpan).coerceAtLeast(-85.0),
        (longitude - longitudeSpan).coerceAtLeast(-180.0),
    )
}
