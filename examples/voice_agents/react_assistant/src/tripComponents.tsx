import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { createContext, useContext, useEffect, useRef, useState } from "react";
import {
  defineAIUIComponent,
  parseAIUIComponentDefinition,
  type AIUIComponent,
  type AIUIComponentDefinition,
  type AIUIComponentProps,
} from "stream-chat-react/ai-components";

// The components are described once, for the iOS and Android apps too, beside the iOS app.
import described from "../../swift_assistant/app/SwiftAssistant/UIComponents.json";
import { Assistant } from "./assistant";
import bookingDescribed from "./bookingComponents.json";
import type { ChatScreen, ChatScreenSnapshot } from "./chatScreen";
import { calendarDays } from "./reminders";
import { isReservationOf, RESERVE_TABLE, reserveTable, type DiningTable, type Reservation } from "./reserveTable";
import { useSelected } from "./store";

/**
 * The conversation the components are shown in, which holds the session a card's tool is given
 * to. Undefined where they are shown outside one.
 */
export const ChatScreenContext = createContext<ChatScreen | undefined>(undefined);

// Each component's view, by the name the agent places it by.
const views: Record<string, (definition: AIUIComponentDefinition) => AIUIComponent> = {
  place_card: (definition) => defineAIUIComponent(definition, PlaceCard),
  trip_map: (definition) => defineAIUIComponent(definition, TripMap),
  calendar_check: (definition) => defineAIUIComponent(definition, CalendarCheck),
  restaurant_card: (definition) => defineAIUIComponent(definition, RestaurantCard),
  availability_gauge: (definition) => defineAIUIComponent(definition, AvailabilityGauge),
  seat_selector: (definition) => defineAIUIComponent(definition, SeatSelector),
};

/**
 * The views the agent can place in a reply's UI, besides the ones its format has: a card for a
 * place with its Wikipedia photo, a map of a plan, the person's own calendar, and hospitality
 * booking elements (restaurant summary, live occupancy gauge, and table zone selector).
 */
export const tripComponents: AIUIComponent[] = [...(described as unknown[]), ...(bookingDescribed as unknown[])].flatMap(
  (json) => {
    const definition = parseAIUIComponentDefinition(json);
    const view = definition && views[definition.name];
    return definition && view ? [view(definition)] : [];
  },
);

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

type RestaurantCardArguments = {
  name: string;
  cuisine: string;
  rating: number;
  priceTier?: string;
};

function RestaurantCard({ args, context }: AIUIComponentProps<RestaurantCardArguments>) {
  return (
    <button
      className="restaurant-card"
      onClick={() =>
        context.emit("inspect_venue", `Show me the menu and highlights for ${args.name}`, {
          name: args.name,
        })
      }
      type="button"
    >
      <div className="restaurant-card-head">
        <strong>{args.name}</strong>
        <span className="restaurant-card-rating">
          ★ {args.rating.toFixed(1)}
          {args.priceTier ? ` · ${args.priceTier}` : ""}
        </span>
      </div>
      <span className="restaurant-card-cuisine">{args.cuisine}</span>
    </button>
  );
}

type AvailabilityGaugeArguments = {
  occupancyPct: number;
  waitLabel: string;
};

function AvailabilityGauge({ args }: AIUIComponentProps<AvailabilityGaugeArguments>) {
  const clamped = Math.max(0, Math.min(100, args.occupancyPct));
  const arcLength = 176;
  const filled = (clamped / 100) * arcLength;
  return (
    <div aria-label={`Occupancy ${Math.round(clamped)} percent, ${args.waitLabel}`} className="availability-gauge">
      <svg viewBox="0 0 140 84" className="availability-gauge-svg" role="img">
        <path
          d="M 14 72 A 56 56 0 0 1 126 72"
          fill="none"
          stroke="var(--border)"
          strokeLinecap="round"
          strokeWidth="10"
        />
        <path
          d="M 14 72 A 56 56 0 0 1 126 72"
          fill="none"
          stroke="var(--accent)"
          strokeDasharray={`${filled} ${arcLength}`}
          strokeLinecap="round"
          strokeWidth="10"
        />
        <text x="70" y="58" textAnchor="middle" className="availability-gauge-value">
          {Math.round(clamped)}%
        </text>
      </svg>
      <span className="availability-gauge-label">{args.waitLabel}</span>
    </div>
  );
}

type SeatSelectorArguments = { tables: DiningTable[] };

/**
 * A venue's tables, by zone and seats. Tapping a free one chooses it and asks to reserve it; a
 * taken one can't be tapped.
 *
 * The card brings its own tool, `reserve_table`: the agent has no way to book a table until a
 * card offering tables is on the screen, so the card gives the conversation's session the tool
 * the first time it shows (`Session.addTools`), and takes it back once a table is reserved
 * (`Session.removeTool`). Its tables then show the booking, even if the agent shows them again.
 */
function SeatSelector({ args, context }: AIUIComponentProps<SeatSelectorArguments>) {
  const screen = useContext(ChatScreenContext);
  const shown: CardState = screen ?? noScreen;
  const reservation = useSelected(shown, (state) =>
    isReservationOf(state.reservation, args.tables) ? state.reservation : undefined,
  );
  const [selected, setSelected] = useState<string | undefined>(
    () => args.tables.find((t) => t.available)?.id,
  );
  const tables = useRef(args.tables);
  tables.current = args.tables;
  // A conversation carried on shows its messages before its session is back, so a card in them
  // brings its tool once the session is open.
  const isOpen = useSelected(shown, (state) => state.session !== undefined);
  const isBooked = reservation !== undefined;

  // Brought once the tables are all in, which is when the tool can tell a free one.
  useEffect(() => {
    if (!screen || !isOpen || isBooked || context.isStreaming) return;
    void screen.bring(
      RESERVE_TABLE,
      reserveTable(tables.current, Assistant.user.name, (made) => screen.reserved(made)),
    );
  }, [screen, isOpen, isBooked, context.isStreaming]);

  if (reservation) return <ReservationConfirmation reservation={reservation} />;
  return (
    <div className="seat-selector">
      <span className="seat-selector-title">Select a table zone</span>
      <div className="seat-selector-grid">
        {args.tables.map((table) => (
          <button
            className={`seat-pill${selected === table.id ? " active" : ""}`}
            disabled={!table.available}
            key={table.id}
            onClick={() => {
              setSelected(table.id);
              context.emit(
                "select_table",
                `Reserve table ${table.id} (${table.zone} for ${table.seats})`,
                { id: table.id, zone: table.zone, seats: table.seats },
              );
            }}
            type="button"
          >
            <strong>{table.id}</strong> · {table.zone} ({table.seats}p)
          </button>
        ))}
      </div>
    </div>
  );
}

/** What of the conversation a seat selector follows: its session and the booking made in it. */
type CardState = {
  get: () => Pick<ChatScreenSnapshot, "session" | "reservation">;
  subscribe: (listener: () => void) => () => void;
};

/** No session or reservation, for a seat selector shown outside a conversation. */
const noScreen: CardState = {
  get: () => ({ session: undefined, reservation: undefined }),
  subscribe: () => () => undefined,
};

/** What a seat selector shows once one of its tables is booked. */
function ReservationConfirmation({ reservation }: { reservation: Reservation }) {
  const { table, when, code } = reservation;
  return (
    <div className="seat-selector reservation-confirmation" role="status">
      <span aria-hidden className="reservation-confirmation-mark">
        ✓
      </span>
      <span className="reservation-confirmation-text">
        <strong>Table {table.id} reserved</strong>
        <span className="reservation-confirmation-detail">
          {table.zone} · {table.seats} people · {when}
        </span>
        <span className="reservation-confirmation-code">Confirmation {code}</span>
      </span>
    </div>
  );
}
