import EventKit
import MapKit
import StreamAgents
import StreamChatAI
import SwiftUI

/// The phone's own views the agent can place in the UI of a reply, such as a trip plan or a
/// table for dinner.
///
/// What each one is, what the agent fills in and what it reports back are in
/// UIComponents.json, which the agent's instructions are written from too (`openui/`), so the
/// agent knows them as the app does. Each one loads what it shows on the phone: the place cards
/// their photos from Wikipedia, and the calendar the person's own days, which the agent never sees.
/// Each wears the brand Settings chose (`Brand`).
@MainActor
enum TripComponents {
    static let all: [any AIUIComponent] = definitions.compactMap { component($0) }

    private static let definitions: [AIUIComponentDefinition] = {
        guard let url = Bundle.main.url(forResource: "UIComponents", withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return [] }
        return (try? JSONDecoder().decode([AIUIComponentDefinition].self, from: data)) ?? []
    }()

    private static func component(_ definition: AIUIComponentDefinition) -> (any AIUIComponent)? {
        switch definition.name {
        case "place_card": PlaceCardComponent(definition: definition)
        case "trip_map": TripMapComponent(definition: definition)
        case "calendar_check": CalendarCheckComponent(definition: definition)
        case "restaurant_card": RestaurantCardComponent(definition: definition)
        case "availability_gauge": AvailabilityGaugeComponent(definition: definition)
        case "seat_selector": SeatSelectorComponent(definition: definition)
        default: nil
        }
    }
}

/// A place's card, which asks about the place when tapped.
struct PlaceCardComponent: AIUIComponent {
    let definition: AIUIComponentDefinition

    func body(arguments: PlaceArguments, context: AIUIComponentContext) -> some View {
        PlaceCard(place: arguments) {
            context.emit(
                "ask_about", label: "Tell me more about \(arguments.title)",
                payload: ["title": .string(arguments.title)])
        }
    }
}

/// A plan's map, which asks about a place when its pin is tapped.
struct TripMapComponent: AIUIComponent {
    let definition: AIUIComponentDefinition

    func body(arguments: TripMapArguments, context: AIUIComponentContext) -> some View {
        TripMap(places: arguments.places, dropsPins: context.isStreaming) { place in
            context.emit(
                "select_place", label: "Tell me more about \(place.name)",
                payload: ["name": .string(place.name)])
        }
    }
}

/// The person's calendar for a plan's days.
struct CalendarCheckComponent: AIUIComponent {
    let definition: AIUIComponentDefinition

    func body(arguments: CalendarArguments, context: AIUIComponentContext) -> some View {
        CalendarCheck(start: arguments.start, end: arguments.end)
    }
}

struct PlaceArguments: Decodable {
    let title: String
    let note: String?
}

/// One place of a plan, with its photo and summary from Wikipedia. Tapping it asks about it.
struct PlaceCard: View {
    let place: PlaceArguments
    let onAsk: () -> Void

    @Environment(\.brand) private var brand
    @State private var summary: Wikipedia.Summary?

    var body: some View {
        Button(action: onAsk) {
            HStack(alignment: .top, spacing: 12) {
                AsyncImage(url: summary?.thumbnail?.source, transaction: Transaction(animation: .easeOut(duration: 0.3))) { phase in
                    if let image = phase.image {
                        image.resizable().scaledToFill()
                    } else {
                        Rectangle().fill(.quaternary)
                    }
                }
                .frame(width: 88, height: 88)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                VStack(alignment: .leading, spacing: 4) {
                    Text(summary?.title ?? place.title)
                        .font(.headline)
                    if let note = place.note {
                        Text(note)
                            .font(.subheadline)
                            .foregroundStyle(.tint)
                    }
                    if let extract = summary?.extract {
                        Text(extract)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .lineLimit(3)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(10)
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: brand.cornerRadius))
            .contentShape(RoundedRectangle(cornerRadius: brand.cornerRadius))
        }
        .buttonStyle(.plain)
        .tint(brand.accent)
        .accessibilityHint("Asks Juno about it")
        .task(id: place.title) {
            let loaded = await Wikipedia.summary(of: place.title)
            withAnimation(.easeOut(duration: 0.3)) {
                summary = loaded
            }
        }
    }
}

/// A place's summary on Wikipedia, kept while the app runs.
@MainActor
enum Wikipedia {
    struct Summary: Decodable, Sendable {
        struct Image: Decodable, Sendable {
            let source: URL
        }

        let title: String
        let extract: String?
        let thumbnail: Image?
    }

    private static var summaries: [String: Summary] = [:]

    static func summary(of title: String) async -> Summary? {
        if let known = summaries[title] { return known }
        let path = title.replacingOccurrences(of: " ", with: "_")
        guard let encoded = path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed),
              let url = URL(string: "https://en.wikipedia.org/api/rest_v1/page/summary/" + encoded)
        else { return nil }
        var request = URLRequest(url: url)
        request.setValue("SwiftAssistant/1.0 (Stream Agents example)", forHTTPHeaderField: "User-Agent")
        guard let (data, _) = try? await URLSession.shared.data(for: request),
              let summary = try? JSONDecoder().decode(Summary.self, from: data)
        else { return nil }
        summaries[title] = summary
        return summary
    }
}

struct TripPlace: Decodable, Hashable {
    let name: String
    let latitude: Double
    let longitude: Double
}

struct TripMapArguments: Decodable {
    let places: [TripPlace]
}

/// A plan's places on a map. Tapping a pin asks about the place. Drawn as the agent writes it, the
/// map drops the places' pins one after another.
struct TripMap: View {
    let places: [TripPlace]
    let onSelect: (TripPlace) -> Void

    @Environment(\.brand) private var brand
    @State private var position: MapCameraPosition
    @State private var pinned: Int
    @State private var selection: String?

    init(places: [TripPlace], dropsPins: Bool, onSelect: @escaping (TripPlace) -> Void) {
        self.places = places
        self.onSelect = onSelect
        _position = State(initialValue: .region(Self.region(of: places)))
        _pinned = State(initialValue: dropsPins ? 0 : places.count)
    }

    var body: some View {
        Map(position: $position, selection: $selection) {
            ForEach(Array(places.prefix(pinned)), id: \.name) { place in
                Marker(
                    place.name, systemImage: "mappin",
                    coordinate: CLLocationCoordinate2D(latitude: place.latitude, longitude: place.longitude)
                )
                .tint(brand.accent)
                .tag(place.name)
            }
        }
        .frame(height: 220)
        .clipShape(RoundedRectangle(cornerRadius: brand.cornerRadius))
        .task {
            // The first pin waits for the map to come in.
            for count in min(pinned, places.count)..<places.count {
                try? await Task.sleep(for: .milliseconds(count == 0 ? 600 : 250))
                withAnimation(.spring(response: 0.35, dampingFraction: 0.7)) {
                    pinned = count + 1
                }
            }
        }
        .onChange(of: places) {
            position = .region(Self.region(of: places))
            pinned = places.count
        }
        .onChange(of: selection) { _, name in
            guard let name, let place = places.first(where: { $0.name == name }) else { return }
            selection = nil
            onSelect(place)
        }
    }
}

extension TripMap {
    /// The region that shows every place, with some room around them.
    static func region(of places: [TripPlace]) -> MKCoordinateRegion {
        let latitudes = places.map(\.latitude)
        let longitudes = places.map(\.longitude)
        guard let south = latitudes.min(), let north = latitudes.max(),
              let west = longitudes.min(), let east = longitudes.max()
        else { return MKCoordinateRegion(.world) }
        return MKCoordinateRegion(
            center: CLLocationCoordinate2D(latitude: (south + north) / 2, longitude: (west + east) / 2),
            span: MKCoordinateSpan(latitudeDelta: max(0.02, (north - south) * 1.8), longitudeDelta: max(0.02, (east - west) * 1.8))
        )
    }
}

struct CalendarArguments: Decodable {
    let start: Date
    let end: Date
}

/// The person's own calendar for the days of a plan, read on the phone. The agent never sees it.
struct CalendarCheck: View {
    let start: Date
    let end: Date

    @Environment(\.brand) private var brand
    @State private var days: [CalendarDay] = []
    @State private var failure: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Your calendar", systemImage: "calendar")
                .font(.subheadline.weight(.semibold))
            if let failure {
                Text(failure)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            ForEach(days) { day in
                HStack(alignment: .firstTextBaseline, spacing: 12) {
                    Text(day.date.formatted(.dateTime.weekday(.abbreviated).day().month(.abbreviated)))
                        .font(.footnote.weight(.medium))
                        .frame(width: 96, alignment: .leading)
                    if day.events.isEmpty {
                        Text("Free")
                            .font(.footnote)
                            .foregroundStyle(.green)
                    } else {
                        Text(day.events.joined(separator: " · "))
                            .font(.footnote)
                    }
                }
            }
            Text("Only you see this. Juno doesn't.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: brand.cornerRadius))
        .task(id: [start, end]) {
            let loaded = await CalendarDay.days(from: start, to: end)
            withAnimation(.easeOut(duration: 0.3)) {
                switch loaded {
                case .some(let loaded):
                    days = loaded
                    failure = nil
                case .none:
                    failure = "Let Juno see your calendar in Settings to check these days."
                }
            }
        }
    }
}

/// One day of the person's calendar: what is on it, or nothing.
struct CalendarDay: Identifiable, Sendable {
    let date: Date
    let events: [String]

    var id: Date { date }

    /// The days from `start` to `end`, at most two weeks of them, or nil when the calendar is not
    /// shared with the app.
    nonisolated static func days(from start: Date, to end: Date) async -> [CalendarDay]? {
        let store = EKEventStore()
        guard (try? await store.requestFullAccessToEvents()) == true else { return nil }
        let calendar = Calendar.current
        var days: [CalendarDay] = []
        var day = calendar.startOfDay(for: start)
        let last = calendar.startOfDay(for: max(start, end))
        while day <= last, days.count < 14 {
            let next = calendar.date(byAdding: .day, value: 1, to: day) ?? day.addingTimeInterval(86_400)
            let events = store.events(matching: store.predicateForEvents(withStart: day, end: next, calendars: nil))
            days.append(CalendarDay(date: day, events: events.map { event in
                let title = event.title ?? "Busy"
                return event.isAllDay ? title : event.startDate.formatted(date: .omitted, time: .shortened) + " " + title
            }))
            day = next
        }
        return days
    }
}

/// A restaurant's card, which asks for its menu when tapped.
struct RestaurantCardComponent: AIUIComponent {
    let definition: AIUIComponentDefinition

    func body(arguments: RestaurantArguments, context: AIUIComponentContext) -> some View {
        RestaurantCard(venue: arguments) {
            context.emit(
                "inspect_venue", label: "Show me the menu and highlights for \(arguments.name)",
                payload: ["name": .string(arguments.name)])
        }
    }
}

/// How busy a venue is right now, and the wait.
struct AvailabilityGaugeComponent: AIUIComponent {
    let definition: AIUIComponentDefinition

    func body(arguments: AvailabilityArguments, context: AIUIComponentContext) -> some View {
        AvailabilityGauge(occupancy: arguments.occupancyPct, wait: arguments.waitLabel)
    }
}

/// A venue's tables by zone, which asks to reserve the one tapped.
struct SeatSelectorComponent: AIUIComponent {
    let definition: AIUIComponentDefinition

    func body(arguments: SeatSelectorArguments, context: AIUIComponentContext) -> some View {
        SeatSelector(tables: arguments.tables) { table in
            context.emit(
                "select_table", label: "Reserve table \(table.id) (\(table.zone) for \(table.seatCount))",
                payload: ["id": .string(table.id), "zone": .string(table.zone), "seats": .number(Double(table.seatCount))])
        }
    }
}

struct RestaurantArguments: Decodable {
    let name: String
    let cuisine: String
    let rating: Double
    let priceTier: String?
}

/// A restaurant, its cuisine and its rating. Tapping it asks for its menu and highlights.
struct RestaurantCard: View {
    let venue: RestaurantArguments
    let onInspect: () -> Void

    @Environment(\.brand) private var brand

    private var badge: String {
        let rating = "★ " + venue.rating.formatted(.number.precision(.fractionLength(1)))
        return venue.priceTier.map { rating + " · " + $0 } ?? rating
    }

    var body: some View {
        Button(action: onInspect) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Text(venue.name)
                        .font(.headline)
                    Spacer(minLength: 0)
                    Text(badge)
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(brand.accent)
                }
                Text(venue.cuisine)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: brand.cornerRadius))
            .contentShape(RoundedRectangle(cornerRadius: brand.cornerRadius))
        }
        .buttonStyle(.plain)
        .accessibilityHint("Asks Juno for its menu and highlights")
    }
}

struct AvailabilityArguments: Decodable {
    let occupancyPct: Double
    let waitLabel: String
}

/// How full a venue is, as a half dial in the brand's accent, with the wait under it.
struct AvailabilityGauge: View {
    let occupancy: Double
    let wait: String

    @Environment(\.brand) private var brand

    private var percent: Double { min(max(occupancy, 0), 100) }

    var body: some View {
        VStack(spacing: 6) {
            ZStack(alignment: .bottom) {
                HalfDial(fraction: 1)
                    .stroke(.quaternary, style: StrokeStyle(lineWidth: 10, lineCap: .round))
                HalfDial(fraction: percent / 100)
                    .stroke(brand.accent, style: StrokeStyle(lineWidth: 10, lineCap: .round))
                    .animation(.easeOut(duration: 0.4), value: percent)
                Text("\(Int(percent.rounded()))%")
                    .font(.title3.weight(.semibold))
                    .contentTransition(.numericText())
            }
            .frame(width: 140, height: 78)
            Text(wait)
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding(12)
        .frame(maxWidth: .infinity)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: brand.cornerRadius))
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(Int(percent.rounded())) percent full, \(wait)")
    }
}

/// The top half of a circle standing on the bottom of its frame, drawn from the left for
/// `fraction` of the way round.
private struct HalfDial: Shape {
    var fraction: Double

    var animatableData: Double {
        get { fraction }
        set { fraction = newValue }
    }

    func path(in rect: CGRect) -> Path {
        let radius = min(rect.width / 2 - 8, rect.height - 12)
        var path = Path()
        path.addArc(
            center: CGPoint(x: rect.midX, y: rect.maxY - 6), radius: radius,
            startAngle: .degrees(180), endAngle: .degrees(180 + 180 * fraction), clockwise: false)
        return path
    }
}

struct SeatSelectorArguments: Decodable {
    let tables: [DiningTable]
}

struct DiningTable: Decodable, Hashable {
    let id: String
    let zone: String
    let seats: Double
    let available: Bool

    var seatCount: Int { Int(seats.rounded()) }
}

/// A venue's tables, by zone and seats. Tapping a free one chooses it and asks to reserve it;
/// a taken one can't be tapped.
///
/// The card brings its own tool, `reserve_table`: the agent has no way to book a table until a
/// card offering tables is on the screen, so the card gives the conversation's session the tool
/// the first time it shows (`AgentSession.add(tools:)`), and takes it back once a table is
/// reserved (`AgentSession.remove(toolNamed:)`).
struct SeatSelector: View {
    let tables: [DiningTable]
    let onSelect: (DiningTable) -> Void

    @Environment(\.brand) private var brand
    /// The conversation the card is shown in, which holds the session its tool is given to. Nil
    /// where the card is shown outside one.
    @Environment(AgentChat.self) private var chat: AgentChat?
    @State private var selected: String?

    init(tables: [DiningTable], onSelect: @escaping (DiningTable) -> Void) {
        self.tables = tables
        self.onSelect = onSelect
        _selected = State(initialValue: tables.first(where: \.available)?.id)
    }

    /// The booking of one of these tables, which ends what the card is for. The agent can
    /// show the same tables again after it, and they show the booking too.
    private var reservation: Reservation? {
        chat?.reservation.flatMap { $0.tables == Set(tables.map(\.id)) ? $0 : nil }
    }

    var body: some View {
        Group {
            if let reservation {
                ReservationConfirmation(reservation: reservation)
            } else {
                picker
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: brand.cornerRadius))
        .task {
            guard let chat, reservation == nil else { return }
            await chat.bring(Self.reserveTable(from: tables) { reservation in await chat.reserved(reservation) })
        }
    }

    private var picker: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Select a table zone")
                .font(.subheadline.weight(.semibold))
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 8, alignment: .leading)], alignment: .leading, spacing: 8) {
                ForEach(tables, id: \.id) { table in
                    let isSelected = selected == table.id
                    Button {
                        selected = table.id
                        onSelect(table)
                    } label: {
                        Text("\(table.id) · \(table.zone) (\(table.seatCount)p)")
                            .font(.footnote.weight(isSelected ? .semibold : .regular))
                            .strikethrough(!table.available)
                            .lineLimit(1)
                            .minimumScaleFactor(0.8)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 7)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(
                                isSelected ? AnyShapeStyle(brand.accent.opacity(0.16)) : AnyShapeStyle(.quaternary.opacity(0.5)),
                                in: RoundedRectangle(cornerRadius: brand.cornerRadius / 2))
                            .overlay {
                                RoundedRectangle(cornerRadius: brand.cornerRadius / 2)
                                    .strokeBorder(isSelected ? brand.accent : .clear)
                            }
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(table.available ? .primary : .tertiary)
                    .disabled(!table.available)
                    .accessibilityAddTraits(isSelected ? .isSelected : [])
                }
            }
        }
    }

    /// What the tool the card brings is called.
    static let reserveTableTool = "reserve_table"

    /// Reserves one of the free tables a card offers, once the person says it may. `onReserved`
    /// hears the reservation.
    static func reserveTable(
        from tables: [DiningTable], onReserved: @escaping @Sendable (Reservation) async -> Void
    ) -> AgentTool {
        AgentTool(
            name: reserveTableTool,
            description: "Reserves a table from the seat selector on the person's screen. Call it when "
                + "they chose a table, such as by tapping one, with that table's id.",
            parameters: .strings(
                [
                    "table_id": "the id of the table, as the seat selector shows it, such as T4",
                    "when": "the day and time of the booking as they would read it, such as tomorrow at 8pm",
                    "summary": "the table, the party and the time as they would read it, such as Terrace, for 4, tomorrow at 8pm",
                ],
                required: ["table_id", "when", "summary"]),
            displayTitle: "Reserving your table",
            approval: .init(
                title: "Reserve this table?",
                message: "Juno books it under your name.",
                reasonArgument: "summary",
                allowTitle: "Reserve",
                declineTitle: "Not now")
        ) { arguments in
            let id = arguments["table_id"]?.stringValue ?? ""
            let when = arguments["when"]?.stringValue ?? ""
            guard let table = tables.first(where: { $0.id == id && $0.available }) else {
                let free = tables.filter(\.available).map(\.id).joined(separator: ", ")
                throw ToolFailure("\(id) is not a free table on the seat selector. The free ones are \(free).")
            }
            guard !when.isEmpty else { throw ToolFailure("Say when the table is for.") }
            let reservation = Reservation(
                table: table, when: when, code: String(UUID().uuidString.prefix(6)),
                tables: Set(tables.map(\.id)))
            await onReserved(reservation)
            return "Reserved table \(table.id) (\(table.zone), for \(table.seatCount)) for \(when), under "
                + "\(Assistant.user.name ?? "their name"). The confirmation code is \(reservation.code). The seat "
                + "selector on their screen now shows this booking, so confirm it in a sentence and do not show "
                + "or patch the seat selector again."
        }
    }
}

/// A booking the seat selector's tool made, and the tables of the card it was made from.
struct Reservation: Sendable, Equatable {
    let table: DiningTable
    let when: String
    let code: String
    let tables: Set<String>
}

/// What a seat selector shows once one of its tables is booked.
struct ReservationConfirmation: View {
    let reservation: Reservation

    @Environment(\.brand) private var brand

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "checkmark.circle.fill")
                .font(.title2)
                .foregroundStyle(brand.accent)
            VStack(alignment: .leading, spacing: 4) {
                Text("Table \(reservation.table.id) reserved")
                    .font(.subheadline.weight(.semibold))
                Text("\(reservation.table.zone) · \(reservation.table.seatCount) people · \(reservation.when)")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                Text("Confirmation \(reservation.code)")
                    .font(.footnote.monospaced())
            }
        }
        .accessibilityElement(children: .combine)
    }
}
