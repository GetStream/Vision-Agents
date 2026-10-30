import EventKit
import MapKit
import StreamChatAI
import SwiftUI

/// The phone's own views the agent can place in the UI of a reply, such as a trip plan.
///
/// What each one is, what the agent fills in and what it reports back are in
/// UIComponents.json, which the agent's instructions are written from too (`openui/`), so the
/// agent knows them as the app does. Each one loads what it shows on the phone: the place cards
/// their photos from Wikipedia, and the calendar the person's own days, which the agent never sees.
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
            .background(.background.secondary, in: RoundedRectangle(cornerRadius: 16))
            .contentShape(RoundedRectangle(cornerRadius: 16))
        }
        .buttonStyle(.plain)
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
        request.setValue("SwiftAssistant/1.0 (Stream Vision Agents example)", forHTTPHeaderField: "User-Agent")
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
                .tag(place.name)
            }
        }
        .frame(height: 220)
        .clipShape(RoundedRectangle(cornerRadius: 16))
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
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 16))
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
