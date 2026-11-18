import { createContext, useContext, useEffect, useState } from 'react';
import { Image, Pressable, StyleSheet, Text, View } from 'react-native';
import Svg, { Path } from 'react-native-svg';

import {
  type AIUIComponent,
  type AIUIComponentDefinition,
  defineAIUIComponent,
  parseAIUIComponentDefinition,
} from 'stream-chat-expo/ai-components';

import uiComponents from '../agent/UIComponents.json';
import { useBrand } from './Brand';

// The trip planner's components, as the Swift, Android and web assistants describe them, from the
// same `UIComponents.json` the agent's instructions are written from. Each wears the brand the
// settings chose.
const definitionOf = (name: string): AIUIComponentDefinition => {
  const definition = parseAIUIComponentDefinition(
    uiComponents.find((component) => component.name === name),
  );
  if (!definition) throw new Error(`UIComponents.json has no ${name}`);
  return definition;
};

/**
 * Whether the agent is answering, when a tap would only be refused: a reply's UI sends what the
 * person does as their next message, which waits for the answer before it.
 */
export const AnsweringContext = createContext(false);

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
    const { brand } = useBrand();
    return (
      <Pressable
        accessibilityHint='Asks about the place'
        accessibilityRole='button'
        onPress={() =>
          context.emit('ask_about', `Tell me more about ${args.title}`, { title: args.title })
        }
        style={({ pressed }) => [
          styles.card,
          { borderRadius: brand.cornerRadius },
          pressed && styles.pressed,
        ]}
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
  ({ args, context }) => {
    const { brand } = useBrand();
    return (
      <View style={[styles.map, { borderRadius: brand.cornerRadius }]}>
        {args.places.map((place, index) => (
          <Pressable
            accessibilityRole='button'
            key={`${place.name}-${index}`}
            onPress={() =>
              context.emit('select_place', `Tell me more about ${place.name}`, { name: place.name })
            }
            style={({ pressed }) => [styles.pin, pressed && styles.pressed]}
          >
            <Text style={[styles.pinIndex, { backgroundColor: brand.accent }]}>{index + 1}</Text>
            <Text numberOfLines={1} style={styles.pinName}>
              {place.name}
            </Text>
            <Text style={styles.coordinates}>
              {place.latitude.toFixed(3)}, {place.longitude.toFixed(3)}
            </Text>
          </Pressable>
        ))}
      </View>
    );
  },
);

const dayOf = (date: Date) =>
  date.toLocaleDateString(undefined, { day: 'numeric', month: 'short', weekday: 'short' });

const CalendarCheck = defineAIUIComponent<{ end: Date; start: Date }>(
  definitionOf('calendar_check'),
  ({ args }) => {
    const { brand } = useBrand();
    return (
      <View style={[styles.calendar, { borderRadius: brand.cornerRadius }]}>
        <Text style={styles.title}>
          {dayOf(args.start)} – {dayOf(args.end)}
        </Text>
        <Text style={styles.note}>Your calendar stays on this phone.</Text>
      </View>
    );
  },
);

type Restaurant = { cuisine: string; name: string; priceTier?: string; rating: number };

const RestaurantCard = defineAIUIComponent<Restaurant>(
  definitionOf('restaurant_card'),
  ({ args, context }) => {
    const { brand } = useBrand();
    const answering = useContext(AnsweringContext);
    return (
      <Pressable
        accessibilityHint='Asks for its menu and highlights'
        accessibilityRole='button'
        accessibilityState={{ disabled: answering }}
        disabled={answering}
        onPress={() =>
          context.emit('inspect_venue', `Show me the menu and highlights for ${args.name}`, {
            name: args.name,
          })
        }
        style={({ pressed }) => [
          styles.calendar,
          { borderRadius: brand.cornerRadius },
          pressed && styles.pressed,
        ]}
      >
        <View style={styles.restaurantHead}>
          <Text numberOfLines={1} style={[styles.title, styles.restaurantName]}>
            {args.name}
          </Text>
          <Text style={[styles.rating, { color: brand.accent }]}>
            ★ {args.rating.toFixed(1)}
            {args.priceTier ? ` · ${args.priceTier}` : ''}
          </Text>
        </View>
        <Text style={styles.note}>{args.cuisine}</Text>
      </Pressable>
    );
  },
);

// Half a circle of radius 56 from the left end to the right, and how long it is.
const GAUGE_ARC = 'M 14 72 A 56 56 0 0 1 126 72';
const GAUGE_LENGTH = Math.PI * 56;

const AvailabilityGauge = defineAIUIComponent<{ occupancyPct: number; waitLabel: string }>(
  definitionOf('availability_gauge'),
  ({ args }) => {
    const { brand } = useBrand();
    const occupancy = Math.max(0, Math.min(100, args.occupancyPct));
    return (
      <View
        accessibilityLabel={`Occupancy ${Math.round(occupancy)} percent, ${args.waitLabel}`}
        accessible
        style={[styles.calendar, styles.gauge, { borderRadius: brand.cornerRadius }]}
      >
        <Svg height={84} viewBox='0 0 140 84' width={140}>
          <Path d={GAUGE_ARC} fill='none' stroke='#ececec' strokeLinecap='round' strokeWidth={10} />
          <Path
            d={GAUGE_ARC}
            fill='none'
            stroke={brand.accent}
            strokeDasharray={`${(occupancy / 100) * GAUGE_LENGTH} ${GAUGE_LENGTH}`}
            strokeLinecap='round'
            strokeWidth={10}
          />
        </Svg>
        <Text style={styles.gaugeValue}>{Math.round(occupancy)}%</Text>
        <Text style={styles.note}>{args.waitLabel}</Text>
      </View>
    );
  },
);

type DiningTable = { available: boolean; id: string; seats: number; zone: string };

const SeatSelector = defineAIUIComponent<{ tables: DiningTable[] }>(
  definitionOf('seat_selector'),
  ({ args, context }) => {
    const { brand } = useBrand();
    const answering = useContext(AnsweringContext);
    const [chosen, setChosen] = useState<string>();
    // The person's table while it stays free, the first free one otherwise: an update can take
    // the table they chose.
    const available = args.tables.filter((table) => table.available);
    const selected = (available.find((table) => table.id === chosen) ?? available[0])?.id;
    return (
      <View style={[styles.calendar, { borderRadius: brand.cornerRadius }]}>
        <Text style={styles.title}>Select a table zone</Text>
        <View style={styles.tables}>
          {args.tables.map((table) => {
            const isSelected = table.id === selected;
            return (
              <Pressable
                accessibilityRole='button'
                accessibilityState={{ disabled: answering || !table.available, selected: isSelected }}
                disabled={answering || !table.available}
                key={table.id}
                onPress={() => {
                  setChosen(table.id);
                  context.emit(
                    'select_table',
                    `Reserve table ${table.id} (${table.zone} for ${table.seats})`,
                    { id: table.id, seats: table.seats, zone: table.zone },
                  );
                }}
                style={({ pressed }) => [
                  styles.table,
                  { borderRadius: brand.cornerRadius / 2 },
                  isSelected && { backgroundColor: brand.accent, borderColor: brand.accent },
                  !table.available && styles.tableTaken,
                  pressed && styles.pressed,
                ]}
              >
                <Text style={[styles.tableText, isSelected && styles.tableTextSelected]}>
                  {table.id} · {table.zone} · {table.seats}
                </Text>
              </Pressable>
            );
          })}
        </View>
      </View>
    );
  },
);

/** The app's components the agent can place in a reply's UI. */
export const tripComponents: AIUIComponent[] = [
  PlaceCard,
  TripMap,
  CalendarCheck,
  RestaurantCard,
  AvailabilityGauge,
  SeatSelector,
];

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
  gauge: { alignItems: 'center' },
  gaugeValue: { fontSize: 22, fontWeight: '700', marginTop: -36 },
  map: { backgroundColor: '#eef4ff', borderRadius: 12, gap: 4, padding: 8 },
  note: { color: '#7a7a7a', fontSize: 13 },
  photo: { borderRadius: 8, height: 64, width: 64 },
  photoPlaceholder: { backgroundColor: '#f2f2f2' },
  pin: { alignItems: 'center', flexDirection: 'row', gap: 8, paddingVertical: 6 },
  pinIndex: {
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
  rating: { fontSize: 14, fontWeight: '600' },
  restaurantHead: { alignItems: 'center', flexDirection: 'row', gap: 8 },
  restaurantName: { flex: 1 },
  table: {
    borderColor: '#d5dbe1',
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  tableTaken: { opacity: 0.4 },
  tableText: { fontSize: 14 },
  tableTextSelected: { color: '#ffffff', fontWeight: '600' },
  tables: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginTop: 8 },
  title: { fontSize: 15, fontWeight: '600' },
});
