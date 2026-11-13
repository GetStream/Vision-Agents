import 'dart:async';
import 'dart:convert';

// ignore: deprecated_member_use
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart' show OverflowBoxFit;
import 'package:flutter/services.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:http/http.dart' as http;
import 'package:intl/intl.dart';
import 'package:latlong2/latlong.dart';
import 'package:stream_chat_flutter_ai/stream_chat_flutter_ai.dart';

import 'device_tools.dart';

/// The phone's own views the agent can place in the UI of a reply, such as a trip plan.
///
/// What each one is, what the agent fills in and what it reports back are in
/// `assets/UIComponents.json`, a copy of swift_assistant's, which the agent's instructions are
/// written from too. Each one loads what it shows on the phone: the place cards their photos
/// from Wikipedia, and the calendar the person's own days, which the agent never sees.
abstract final class TripComponents {
  /// The components, once [load] has read their definitions.
  static List<AIUIComponent<Object?>> all = const [];

  /// Reads the components' definitions from the app's assets.
  static Future<void> load() async {
    final json =
        jsonDecode(await rootBundle.loadString('assets/UIComponents.json')) as List<Object?>;
    all = [
      for (final entry in json)
        ?_component(AIUIComponentDefinition.fromJson(entry! as Map<String, Object?>)),
    ];
  }

  static AIUIComponent<Object?>? _component(AIUIComponentDefinition definition) =>
      switch (definition.name) {
        'place_card' => PlaceCardComponent(definition),
        'trip_map' => TripMapComponent(definition),
        'calendar_check' => CalendarCheckComponent(definition),
        _ => null,
      };
}

typedef PlaceArguments = ({String title, String? note});

/// A place's card, which asks about the place when tapped.
class PlaceCardComponent extends AIUIComponent<PlaceArguments> {
  PlaceCardComponent(this.definition);

  @override
  final AIUIComponentDefinition definition;

  @override
  PlaceArguments readArguments(AIUIArguments arguments) =>
      (title: arguments.string('title'), note: arguments.optional('note', arguments.string));

  @override
  Widget build(BuildContext context, PlaceArguments place, AIUIComponentContext ui) => PlaceCard(
    title: place.title,
    note: place.note,
    onAsk: () => ui.emit(
      'ask_about',
      label: 'Tell me more about ${place.title}',
      payload: {'title': place.title},
    ),
  );
}

/// One place of a plan, with its photo and summary from Wikipedia. Tapping it asks about it.
class PlaceCard extends StatefulWidget {
  const PlaceCard({super.key, required this.title, this.note, required this.onAsk});

  final String title;
  final String? note;
  final VoidCallback onAsk;

  @override
  State<PlaceCard> createState() => _PlaceCardState();
}

class _PlaceCardState extends State<PlaceCard> {
  static const _photoSide = 88.0;

  WikipediaSummary? _summary;
  var _isLoading = true;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void didUpdateWidget(PlaceCard oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.title != widget.title) {
      // The last place's photo and summary are not this one's.
      _summary = null;
      _isLoading = true;
      _load();
    }
  }

  Future<void> _load() async {
    final title = widget.title;
    final summary = await Wikipedia.summary(title);
    if (mounted && title == widget.title) {
      setState(() {
        _summary = summary;
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final colors = context.streamColorScheme;
    final text = context.streamTextTheme;
    final summary = _summary;
    return Semantics(
      button: true,
      hint: 'Asks Juno about it',
      child: Material(
        color: colors.backgroundSurface,
        borderRadius: BorderRadius.circular(16),
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: widget.onAsk,
          child: Padding(
            padding: const EdgeInsets.all(10),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // The photo's place is kept while it loads, and given up when there is none.
                if (_isLoading || summary?.thumbnail != null) ...[
                  ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: SizedBox.square(
                      dimension: _photoSide,
                      child: AnimatedSwitcher(
                        duration: const Duration(milliseconds: 300),
                        child: summary?.thumbnail == null
                            ? ColoredBox(
                                key: const ValueKey(false),
                                color: colors.backgroundSurfaceStrong,
                              )
                            : Image.network(
                                summary!.thumbnail!,
                                key: ValueKey(summary.thumbnail),
                                fit: BoxFit.cover,
                                // Decoded at the size it shows at, not the photo's own.
                                cacheWidth: (_photoSide * MediaQuery.devicePixelRatioOf(context))
                                    .round(),
                                errorBuilder: (context, _, _) =>
                                    ColoredBox(color: colors.backgroundSurfaceStrong),
                              ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                ],
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    spacing: 4,
                    children: [
                      Text(
                        summary?.title ?? widget.title,
                        style: text.bodyEmphasis.copyWith(color: colors.textPrimary),
                      ),
                      if (widget.note case final note?)
                        Text(
                          note,
                          style: text.captionDefault.copyWith(color: colors.accentPrimary),
                        ),
                      if (summary?.extract case final extract?)
                        Text(
                          extract,
                          maxLines: 3,
                          overflow: TextOverflow.ellipsis,
                          style: text.captionDefault.copyWith(color: colors.textSecondary),
                        ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/// A place's summary on Wikipedia.
typedef WikipediaSummary = ({String title, String? extract, String? thumbnail});

/// Places' summaries on Wikipedia, kept while the app runs.
abstract final class Wikipedia {
  static final _summaries = <String, Future<WikipediaSummary?>>{};

  /// The summary of the article titled [title], or null when it can't be loaded. Cards asking
  /// for the same place at once share one request.
  static Future<WikipediaSummary?> summary(String title) =>
      _summaries[title] ??= _fetch(title).then((summary) {
        // Asked for again next time, rather than kept as missing.
        if (summary == null) _summaries.remove(title);
        return summary;
      });

  static Future<WikipediaSummary?> _fetch(String title) async {
    final url = Uri.https(
      'en.wikipedia.org',
      '/api/rest_v1/page/summary/${title.replaceAll(' ', '_')}',
    );
    try {
      final response = await http
          .get(url, headers: {'User-Agent': 'FlutterAssistant/1.0 (Stream Vision Agents example)'})
          .timeout(const Duration(seconds: 10));
      if (response.statusCode != 200) return null;
      final json = jsonDecode(response.body) as Map<String, Object?>;
      final summary = (
        title: json['title'] as String? ?? title,
        extract: json['extract'] as String?,
        thumbnail: (json['thumbnail'] as Map<String, Object?>?)?['source'] as String?,
      );
      return summary;
    } on Object {
      return null;
    }
  }
}

/// A place on a plan's map.
typedef TripPlace = ({String name, double latitude, double longitude});

/// A plan's map, which asks about a place when its pin is tapped.
class TripMapComponent extends AIUIComponent<List<TripPlace>> {
  TripMapComponent(this.definition);

  @override
  final AIUIComponentDefinition definition;

  @override
  List<TripPlace> readArguments(AIUIArguments arguments) => [
    for (final place in arguments.objects('places'))
      (
        name: place.string('name'),
        latitude: place.number('latitude'),
        longitude: place.number('longitude'),
      ),
  ];

  @override
  Widget build(BuildContext context, List<TripPlace> places, AIUIComponentContext ui) => TripMap(
    places: places,
    dropsPins: ui.isStreaming,
    onSelect: (place) => ui.emit(
      'select_place',
      label: 'Tell me more about ${place.name}',
      payload: {'name': place.name},
    ),
  );
}

/// A plan's places on an OpenStreetMap map. Tapping a pin asks about the place. Drawn as the
/// agent writes it, the map drops the places' pins one after another.
class TripMap extends StatefulWidget {
  const TripMap({super.key, required this.places, required this.dropsPins, required this.onSelect});

  final List<TripPlace> places;
  final bool dropsPins;
  final ValueChanged<TripPlace> onSelect;

  @override
  State<TripMap> createState() => _TripMapState();
}

class _TripMapState extends State<TripMap> {
  final _map = MapController();
  late int _pinned = widget.dropsPins ? 0 : widget.places.length;
  Timer? _dropping;

  @override
  void initState() {
    super.initState();
    _drop();
  }

  void _drop() {
    _dropping?.cancel();
    if (_pinned >= widget.places.length) return;
    // The first pin waits for the map to come in.
    _dropping = Timer(Duration(milliseconds: _pinned == 0 ? 600 : 250), () {
      if (!mounted) return;
      setState(() => _pinned += 1);
      _drop();
    });
  }

  @override
  void didUpdateWidget(TripMap oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!_samePlaces(oldWidget.places, widget.places)) {
      _dropping?.cancel();
      _pinned = widget.places.length;
      if (widget.places.isNotEmpty) _map.fitCamera(_fit(widget.places));
    }
  }

  static bool _samePlaces(List<TripPlace> a, List<TripPlace> b) =>
      a.length == b.length &&
      [for (var i = 0; i < a.length; i++) a[i] == b[i]].every((same) => same);

  static CameraFit _fit(List<TripPlace> places) => CameraFit.coordinates(
    coordinates: [for (final place in places) LatLng(place.latitude, place.longitude)],
    padding: const EdgeInsets.all(36),
    maxZoom: 14,
  );

  @override
  void dispose() {
    _dropping?.cancel();
    _map.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final colors = context.streamColorScheme;
    final places = widget.places;
    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: SizedBox(
        height: 220,
        // A2UI's rows and columns measure their children by dry layout, which the map's
        // LayoutBuilder cannot answer. This takes the room it is given without asking it.
        child: OverflowBox(
          fit: OverflowBoxFit.max,
          child: FlutterMap(
            mapController: _map,
            options: MapOptions(
              initialCenter: const LatLng(0, 0),
              initialZoom: 1,
              initialCameraFit: places.isEmpty ? null : _fit(places),
              interactionOptions: const InteractionOptions(
                flags: InteractiveFlag.all & ~InteractiveFlag.rotate,
              ),
            ),
            children: [
              TileLayer(
                urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                userAgentPackageName: 'io.getstream.visionagents.flutterassistant',
              ),
              MarkerLayer(
                markers: [
                  for (final place in places.take(_pinned))
                    Marker(
                      point: LatLng(place.latitude, place.longitude),
                      width: 40,
                      height: 40,
                      alignment: Alignment.topCenter,
                      child: Semantics(
                        button: true,
                        label: place.name,
                        child: GestureDetector(
                          onTap: () => widget.onSelect(place),
                          child: TweenAnimationBuilder<double>(
                            tween: Tween(begin: 0, end: 1),
                            duration: const Duration(milliseconds: 350),
                            curve: Curves.elasticOut,
                            builder: (context, t, child) => Transform.scale(scale: t, child: child),
                            child: Icon(Icons.location_on, size: 40, color: colors.accentPrimary),
                          ),
                        ),
                      ),
                    ),
                ],
              ),
              SimpleAttributionWidget(
                source: Text(
                  'OpenStreetMap contributors',
                  style: context.streamTextTheme.metadataDefault.copyWith(
                    color: colors.textPrimary,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

typedef CalendarRange = ({DateTime start, DateTime end});

/// The person's calendar for a plan's days.
class CalendarCheckComponent extends AIUIComponent<CalendarRange> {
  CalendarCheckComponent(this.definition);

  @override
  final AIUIComponentDefinition definition;

  @override
  CalendarRange readArguments(AIUIArguments arguments) =>
      (start: arguments.date('start'), end: arguments.date('end'));

  @override
  Widget build(BuildContext context, CalendarRange days, AIUIComponentContext ui) =>
      CalendarCheck(start: days.start, end: days.end);
}

/// One day of the person's calendar: what is on it, or nothing.
typedef CalendarDay = ({DateTime date, List<String> events});

/// The person's own calendar for the days of a plan, read on the phone. The agent never sees it.
class CalendarCheck extends StatefulWidget {
  const CalendarCheck({super.key, required this.start, required this.end});

  final DateTime start;
  final DateTime end;

  @override
  State<CalendarCheck> createState() => _CalendarCheckState();
}

class _CalendarCheckState extends State<CalendarCheck> {
  List<CalendarDay> _days = const [];
  String? _failure;

  @override
  void initState() {
    super.initState();
    _load();
  }

  @override
  void didUpdateWidget(CalendarCheck oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.start != widget.start || oldWidget.end != widget.end) _load();
  }

  Future<void> _load() async {
    final (start, end) = (widget.start, widget.end);
    final days = await DeviceTools.calendarDays(start, end);
    if (!mounted || start != widget.start || end != widget.end) return;
    setState(() {
      _days = days ?? const [];
      _failure = days == null
          ? 'Let Juno see your calendar in Settings to check these days.'
          : null;
    });
  }

  @override
  Widget build(BuildContext context) {
    final colors = context.streamColorScheme;
    final text = context.streamTextTheme;
    final day = DateFormat('EEE d MMM');
    return DecoratedBox(
      decoration: BoxDecoration(
        color: colors.backgroundSurface,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          spacing: 8,
          children: [
            Row(
              spacing: 6,
              children: [
                Icon(Icons.calendar_month, size: 18, color: colors.textPrimary),
                Text('Your calendar', style: text.bodyEmphasis.copyWith(color: colors.textPrimary)),
              ],
            ),
            if (_failure case final failure?)
              Text(failure, style: text.captionDefault.copyWith(color: colors.textSecondary)),
            for (final entry in _days)
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                spacing: 12,
                children: [
                  SizedBox(
                    width: 96,
                    child: Text(
                      day.format(entry.date),
                      style: text.captionEmphasis.copyWith(color: colors.textPrimary),
                    ),
                  ),
                  Expanded(
                    child: entry.events.isEmpty
                        ? Text(
                            'Free',
                            style: text.captionDefault.copyWith(color: colors.accentSuccess),
                          )
                        : Text(
                            entry.events.join(' · '),
                            style: text.captionDefault.copyWith(color: colors.textPrimary),
                          ),
                  ),
                ],
              ),
            Text(
              "Only you see this. Juno doesn't.",
              style: text.metadataDefault.copyWith(color: colors.textTertiary),
            ),
          ],
        ),
      ),
    );
  }
}
