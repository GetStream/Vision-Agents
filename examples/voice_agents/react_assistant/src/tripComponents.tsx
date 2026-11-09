import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { useEffect, useRef, useState } from "react";
import {
  defineAIUIComponent,
  parseAIUIComponentDefinition,
  type AIUIComponent,
  type AIUIComponentDefinition,
  type AIUIComponentProps,
} from "stream-chat-react/ai-components";

// The components are described once, for the iOS and Android apps too, beside the iOS app.
import described from "../../swift_assistant/app/SwiftAssistant/UIComponents.json";
import { calendarDays } from "./reminders";

// Each component's view, by the name the agent places it by.
const views: Record<string, (definition: AIUIComponentDefinition) => AIUIComponent> = {
  place_card: (definition) => defineAIUIComponent(definition, PlaceCard),
  trip_map: (definition) => defineAIUIComponent(definition, TripMap),
  calendar_check: (definition) => defineAIUIComponent(definition, CalendarCheck),
};

/**
 * The views the agent can place in a reply's UI, besides the ones its format has: a card for a
 * place with its Wikipedia photo, a map of a plan, and the person's own calendar for its days.
 * The same components, by the same names, as the iOS app's.
 */
export const tripComponents: AIUIComponent[] = (described as unknown[]).flatMap((json) => {
  const definition = parseAIUIComponentDefinition(json);
  const view = definition && views[definition.name];
  return definition && view ? [view(definition)] : [];
});

type PlaceArguments = { title: string; note?: string };

/** One place of a plan, with its photo and summary from Wikipedia. Tapping it asks about it. */
function PlaceCard({ args, context }: AIUIComponentProps<PlaceArguments>) {
  const summary = useWikipediaSummary(args.title);
  return (
    <button
      className="place-card"
      onClick={() => context.emit("ask_about", `Tell me more about ${args.title}`, { title: args.title })}
      title="Asks Juno about it"
      type="button"
    >
      {summary?.thumbnail ? (
        <img alt="" className="place-card-photo" src={summary.thumbnail.source} />
      ) : (
        <span className="place-card-photo" />
      )}
      <span className="place-card-text">
        <strong>{summary?.title ?? args.title}</strong>
        {args.note && <span className="place-card-note">{args.note}</span>}
        {summary?.extract && <span className="place-card-extract">{summary.extract}</span>}
      </span>
    </button>
  );
}

type WikipediaSummary = { title: string; extract?: string; thumbnail?: { source: string } };

const summaries = new Map<string, Promise<WikipediaSummary | undefined>>();

/** A place's Wikipedia summary, read once per title while the page is open. */
export function wikipediaSummary(title: string): Promise<WikipediaSummary | undefined> {
  let summary = summaries.get(title);
  if (!summary) {
    const path = encodeURIComponent(title.replaceAll(" ", "_"));
    summary = fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${path}`)
      .then((response) => (response.ok ? (response.json() as Promise<WikipediaSummary>) : undefined))
      .catch(() => undefined);
    summaries.set(title, summary);
  }
  return summary;
}

function useWikipediaSummary(title: string) {
  const [summary, setSummary] = useState<WikipediaSummary | undefined>(undefined);
  useEffect(() => {
    let live = true;
    void wikipediaSummary(title).then((loaded) => {
      if (live) setSummary(loaded);
    });
    return () => {
      live = false;
    };
  }, [title]);
  return summary;
}

type TripPlace = { name: string; latitude: number; longitude: number };
type TripMapArguments = { places: TripPlace[] };

/**
 * A map of a plan with a pin for each place, dropped one after another while the agent writes
 * the plan. Tapping a pin asks about its place.
 */
function TripMap({ args, context }: AIUIComponentProps<TripMapArguments>) {
  const element = useRef<HTMLDivElement>(null);
  const [map, setMap] = useState<L.Map | undefined>(undefined);
  const [pinned, setPinned] = useState(context.isStreaming ? 0 : args.places.length);
  const places = args.places;
  const placesKey = JSON.stringify(places);
  const emit = useRef(context.emit);
  emit.current = context.emit;

  useEffect(() => {
    if (!element.current) return;
    const made = L.map(element.current, { scrollWheelZoom: false });
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
      maxZoom: 19,
    }).addTo(made);
    setMap(made);
    return () => {
      made.remove();
    };
  }, []);

  useEffect(() => {
    if (!map || places.length === 0) return;
    map.fitBounds(L.latLngBounds(places.map((place) => [place.latitude, place.longitude])), {
      maxZoom: 14,
      padding: [24, 24],
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, placesKey]);

  // The first pin waits for the map to come in, the rest follow it.
  useEffect(() => {
    if (pinned >= places.length) return;
    const timer = setTimeout(() => setPinned((count) => count + 1), pinned === 0 ? 600 : 250);
    return () => clearTimeout(timer);
  }, [pinned, places.length]);

  useEffect(() => {
    if (!map) return;
    const pins = L.layerGroup(
      places.slice(0, pinned).map((place) =>
        L.circleMarker([place.latitude, place.longitude], { className: "trip-map-pin", radius: 8 })
          .bindTooltip(place.name)
          .on("click", () =>
            emit.current("select_place", `Tell me more about ${place.name}`, { name: place.name }),
          ),
      ),
    ).addTo(map);
    return () => {
      pins.remove();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, pinned, placesKey]);

  return <div aria-label="Map of the plan" className="trip-map" ref={element} role="img" />;
}

type CalendarArguments = { start: Date; end: Date };

/**
 * The person's own calendar for the days of a plan, read in the page. The agent never sees it:
 * it is so the person can see what the trip would clash with.
 */
function CalendarCheck({ args }: AIUIComponentProps<CalendarArguments>) {
  const days = calendarDays(args.start, args.end);
  return (
    <section className="calendar-check">
      <h4>Your calendar</h4>
      <ul>
        {days.map(({ date, events }) => (
          <li key={date.toISOString()}>
            <span className="calendar-check-day">
              {date.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" })}
            </span>
            {events.length === 0 ? (
              <span className="calendar-check-free">Free</span>
            ) : (
              <span>{events.join(" · ")}</span>
            )}
          </li>
        ))}
      </ul>
      <p className="calendar-check-private">Only you see this. Juno doesn't.</p>
    </section>
  );
}
