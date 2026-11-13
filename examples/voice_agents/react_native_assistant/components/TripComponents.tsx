import { useEffect, useState } from 'react';
import { Image, Pressable, StyleSheet, Text, View } from 'react-native';

import {
  type AIUIComponent,
  type AIUIComponentDefinition,
  defineAIUIComponent,
  parseAIUIComponentDefinition,
} from 'stream-chat-expo/ai-components';

import uiComponents from '../agent/UIComponents.json';

// The trip planner's components, as the Swift, Android and web assistants describe them, from the
// same `UIComponents.json` the agent's instructions are written from.
const definitionOf = (name: string): AIUIComponentDefinition => {
  const definition = parseAIUIComponentDefinition(
    uiComponents.find((component) => component.name === name),
  );
  if (!definition) throw new Error(`UIComponents.json has no ${name}`);
  return definition;
};

type Summary = { extract?: string; thumbnail?: { source: string } };

// A place's photo and first lines from Wikipedia, loaded on the phone, so the agent writes no
// image links.
const useWikipediaSummary = (title: string) => {
  const [summary, setSummary] = useState<Summary>();
  useEffect(() => {
    const controller = new AbortController();
    fetch(
      `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title.replace(/ /g, '_'))}`,
      { signal: controller.signal },
    )
      .then((response) => (response.ok ? response.json() : undefined))
      .then((json: Summary | undefined) => setSummary(json))
      .catch(() => undefined);
    return () => controller.abort();
  }, [title]);
  return summary;
};

const PlaceCard = defineAIUIComponent<{ title: string; note?: string }>(
  definitionOf('place_card'),
  ({ args, context }) => {
    const summary = useWikipediaSummary(args.title);
    return (
      <Pressable
        accessibilityHint='Asks about the place'
        accessibilityRole='button'
        onPress={() =>
          context.emit('ask_about', `Tell me more about ${args.title}`, { title: args.title })
        }
        style={({ pressed }) => [styles.card, pressed && styles.pressed]}
      >
        {summary?.thumbnail ? (
          <Image source={{ uri: summary.thumbnail.source }} style={styles.photo} />
        ) : (
          <View style={[styles.photo, styles.photoPlaceholder]} />
        )}
        <View style={styles.cardText}>
          <Text numberOfLines={1} style={styles.title}>
            {args.title}
          </Text>
          {args.note ? <Text style={styles.note}>{args.note}</Text> : null}
          {summary?.extract ? (
            <Text numberOfLines={2} style={styles.extract}>
              {summary.extract}
            </Text>
          ) : null}
        </View>
      </Pressable>
    );
  },
);

type Place = { latitude: number; longitude: number; name: string };

const TripMap = defineAIUIComponent<{ places: Place[] }>(
  definitionOf('trip_map'),
  ({ args, context }) => (
    <View style={styles.map}>
      {args.places.map((place, index) => (
        <Pressable
          accessibilityRole='button'
          key={`${place.name}-${index}`}
          onPress={() =>
            context.emit('select_place', `Tell me more about ${place.name}`, { name: place.name })
          }
          style={({ pressed }) => [styles.pin, pressed && styles.pressed]}
        >
          <Text style={styles.pinIndex}>{index + 1}</Text>
          <Text numberOfLines={1} style={styles.pinName}>
            {place.name}
          </Text>
          <Text style={styles.coordinates}>
            {place.latitude.toFixed(3)}, {place.longitude.toFixed(3)}
          </Text>
        </Pressable>
      ))}
    </View>
  ),
);

const dayOf = (date: Date) =>
  date.toLocaleDateString(undefined, { day: 'numeric', month: 'short', weekday: 'short' });

const CalendarCheck = defineAIUIComponent<{ end: Date; start: Date }>(
  definitionOf('calendar_check'),
  ({ args }) => (
    <View style={styles.calendar}>
      <Text style={styles.title}>
        {dayOf(args.start)} – {dayOf(args.end)}
      </Text>
      <Text style={styles.note}>Your calendar stays on this phone.</Text>
    </View>
  ),
);

/** The app's components the agent can place in a reply's UI. */
export const tripComponents: AIUIComponent[] = [PlaceCard, TripMap, CalendarCheck];

const styles = StyleSheet.create({
  calendar: {
    backgroundColor: '#ffffff',
    borderColor: '#ececec',
    borderRadius: 12,
    borderWidth: 1,
    gap: 2,
    padding: 12,
  },
  card: {
    alignItems: 'center',
    backgroundColor: '#ffffff',
    borderColor: '#ececec',
    borderRadius: 12,
    borderWidth: 1,
    flexDirection: 'row',
    gap: 12,
    padding: 8,
  },
  cardText: { flex: 1, gap: 2 },
  coordinates: { color: '#7a7a7a', fontSize: 12 },
  extract: { color: '#4b4b4b', fontSize: 13 },
  map: { backgroundColor: '#eef4ff', borderRadius: 12, gap: 4, padding: 8 },
  note: { color: '#7a7a7a', fontSize: 13 },
  photo: { borderRadius: 8, height: 64, width: 64 },
  photoPlaceholder: { backgroundColor: '#f2f2f2' },
  pin: { alignItems: 'center', flexDirection: 'row', gap: 8, paddingVertical: 6 },
  pinIndex: {
    backgroundColor: '#005fff',
    borderRadius: 10,
    color: '#ffffff',
    fontSize: 12,
    fontWeight: '700',
    height: 20,
    lineHeight: 20,
    overflow: 'hidden',
    textAlign: 'center',
    width: 20,
  },
  pinName: { flex: 1, fontSize: 15 },
  pressed: { opacity: 0.7 },
  title: { fontSize: 15, fontWeight: '600' },
});
