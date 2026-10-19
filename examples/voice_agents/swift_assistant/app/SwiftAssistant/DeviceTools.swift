import CoreLocation
import EventKit
import Foundation
import VisionAgentsCore

/// What the agent can ask this phone: the time and place, the calendar and reminders, the
/// newest photo, and to send an email from the person's Gmail.
///
/// The agent runs in the backend and knows none of this. Each tool runs here, when the model
/// calls it, and only its answer leaves the phone. The two that share something personal ask
/// first: the call waits in the session's `approvals` until the person answers, on the reply's
/// step in a chat or on a card over a call.
enum DeviceTools {
    static let all = [localContext, location, events, reminder, sendEmail, latestPhoto]

    /// The date, time and timezone, which a model in a data centre has no way to know.
    ///
    /// It takes no arguments, and still declares an empty object of them: a tool sent with no
    /// parameters at all is refused by xAI, which fails the whole reply.
    private static let localContext = AgentTool(
        name: "get_local_context",
        description: "The person's current date, local time, timezone and language. Call it "
            + "before answering anything that depends on today or the time.",
        parameters: .strings([:])
    ) { _ in
        let now = Date()
        let zone = TimeZone.current
        let day = now.formatted(.dateTime.weekday(.wide).day().month(.wide).year())
        let time = now.formatted(.dateTime.hour().minute())
        return "It is \(day), \(time) in \(zone.identifier) (\(zone.abbreviation() ?? "")). "
            + "Today as a date is \(now.formatted(isoDay)). "
            + "Their language is \(Locale.current.identifier)."
    }

    /// Where the person is, to the street. It asks first, since it is where they are.
    private static let location = AgentTool(
        name: "get_location",
        description: "Where the person is: their street, neighbourhood and city. Only call it "
            + "when the answer depends on where they are, such as the weather or places nearby.",
        parameters: .strings(
            ["reason": "why you need it, in a few words they would recognise"],
            required: ["reason"]),
        displayTitle: "Checking where you are",
        approval: .init(
            title: "Share your location?",
            message: "Juno gets your street and city, once.",
            reasonArgument: "reason",
            allowTitle: "Share",
            declineTitle: "Not now")
    ) { _ in
        let here = try await Locator.current()
        guard let place = try await CLGeocoder().reverseGeocodeLocation(here).first else {
            return "They are at latitude \(here.coordinate.latitude), longitude "
                + "\(here.coordinate.longitude), which has no address."
        }
        let parts = [place.thoroughfare, place.subLocality, place.locality, place.country]
        return "They are on " + parts.compactMap { $0 }.joined(separator: ", ") + "."
    }

    /// What is on the calendar for one day.
    private static let events = AgentTool(
        name: "list_events",
        description: "The events on the person's calendar for one day.",
        parameters: .strings(
            ["day": "the day as YYYY-MM-DD, in their timezone"], required: ["day"]),
        displayTitle: "Checking your calendar"
    ) { arguments in
        guard let start = try? isoDay.parse(arguments["day"]?.stringValue ?? "") else {
            throw ToolFailure("The day has to be a date like 2026-10-16.")
        }
        let store = EKEventStore()
        guard try await store.requestFullAccessToEvents() else {
            throw ToolFailure("They have not let Juno see their calendar. It is in Settings.")
        }
        let end = Calendar.current.date(byAdding: .day, value: 1, to: start) ?? start
        let found = store.events(
            matching: store.predicateForEvents(withStart: start, end: end, calendars: nil))
        guard !found.isEmpty else { return "Nothing is on their calendar that day." }
        return found.map { event in
            let when = event.isAllDay
                ? "All day"
                : "\(event.startDate.formatted(date: .omitted, time: .shortened))–"
                    + event.endDate.formatted(date: .omitted, time: .shortened)
            let place = event.location.map { " at \($0)" } ?? ""
            return "\(when): \(event.title ?? "Busy")\(place)"
        }.joined(separator: "\n")
    }

    /// Adds a reminder, once the person says it may.
    private static let reminder = AgentTool(
        name: "add_reminder",
        description: "Adds a reminder to the person's Reminders, with an alert when it is due.",
        parameters: .object([
            "type": .string("object"),
            "properties": .object([
                "title": .object([
                    "type": .string("string"),
                    "description": .string("what to remind them of, such as Call mum"),
                ]),
                "due": .object([
                    "type": .string("string"),
                    "description": .string(
                        "when, as an ISO 8601 date and time with their UTC offset, such as "
                            + "2026-10-16T18:00:00+02:00. Leave it out for no due time."),
                ]),
                "summary": .object([
                    "type": .string("string"),
                    "description": .string(
                        "the reminder and its time as they would read it, such as Call mum, "
                            + "today at 6pm"),
                ]),
            ]),
            "required": .array([.string("title"), .string("summary")]),
        ]),
        displayTitle: "Adding a reminder",
        approval: .init(
            title: "Add this reminder?",
            message: "It goes in your Reminders, with an alert when it is due.",
            reasonArgument: "summary",
            allowTitle: "Add reminder",
            declineTitle: "Not now")
    ) { arguments in
        let title = arguments["title"]?.stringValue ?? ""
        guard !title.isEmpty else { throw ToolFailure("A reminder needs a title.") }
        let store = EKEventStore()
        guard try await store.requestFullAccessToReminders() else {
            throw ToolFailure("They have not let Juno add reminders. It is in Settings.")
        }
        let reminder = EKReminder(eventStore: store)
        reminder.title = title
        reminder.calendar = store.defaultCalendarForNewReminders()
        if let due = arguments["due"]?.stringValue,
            let date = try? Date.ISO8601FormatStyle().parse(due)
        {
            reminder.dueDateComponents = Calendar.current.dateComponents(
                [.year, .month, .day, .hour, .minute], from: date)
            reminder.addAlarm(EKAlarm(absoluteDate: date))
        }
        try store.save(reminder, commit: true)
        return "Added the reminder \"\(title)\"."
    }

    /// A day as YYYY-MM-DD, in the person's timezone.
    /// Sends an email from the person's Gmail, once they say it may.
    ///
    /// Google's Gmail MCP server writes drafts but cannot send, so sending is a tool here: the
    /// approval is asked on the phone, and the backend sends through the router's proxy for the
    /// person's Gmail connection, which holds the credential. The question shows who it goes to,
    /// read from the call itself rather than from anything the model wrote about it.
    private static let sendEmail = AgentTool(
        name: "send_email",
        description: "Sends an email from the person's Gmail. Only call it once they have asked "
            + "for it to be sent, after you have shown them the subject and the body.",
        parameters: .strings(
            [
                "to": "who it is to: one or more email addresses, separated by commas",
                "subject": "the subject line",
                "body": "the email itself, in plain text",
            ],
            required: ["to", "subject", "body"]),
        displayTitle: "Sending your email",
        approval: .init(
            title: "Send this email?",
            message: "Juno sends it from your Gmail, as you.",
            reasonArgument: "to",
            allowTitle: "Send",
            declineTitle: "Not now")
    ) { arguments in
        let to = arguments["to"]?.stringValue ?? ""
        let subject = arguments["subject"]?.stringValue ?? ""
        let body = arguments["body"]?.stringValue ?? ""
        guard !to.isEmpty, !body.isEmpty else {
            throw ToolFailure("An email needs someone to go to and a body.")
        }
        _ = try await GmailAccount.send(to: to, subject: subject, body: body)
        return "Sent \"\(subject)\" to \(to) from their Gmail."
    }

    /// The newest picture in the person's photo library, which the agent is shown. It asks
    /// first, since it is one of their photos.
    ///
    /// The router hands a tool's pictures to the agent's vision skill, which runs on a call,
    /// so this works in voice mode; in a chat the agent is told it could not look.
    private static let latestPhoto = AgentTool(
        name: "look_at_latest_photo",
        description: "Look at the newest photo in the person's photo library. Call it when they "
            + "ask about a photo they just took or their latest picture.",
        parameters: .strings(
            ["reason": "why you want to see it, in a few words they would recognise"],
            required: ["reason"]),
        displayTitle: "Looking at your latest photo",
        approval: .init(
            title: "Show Juno your latest photo?",
            message: "Juno sees the newest picture in your library, once.",
            reasonArgument: "reason",
            allowTitle: "Show",
            declineTitle: "Not now")
    ) { _ in
        guard let latest = try await Photos.latest() else {
            return ToolOutput(text: "Their photo library has no pictures.")
        }
        let when = latest.taken.map { ", taken " + $0.formatted(date: .abbreviated, time: .shortened) }
        return ToolOutput(text: "Their newest photo\(when ?? "").", images: [latest.image])
    }

    private static let isoDay = Date.ISO8601FormatStyle(timeZone: .current).year().month().day()
}

/// Why a tool could not answer, in words the model passes on.
struct ToolFailure: LocalizedError {
    let errorDescription: String?

    init(_ message: String) {
        errorDescription = message
    }
}

/// One reading of where the phone is.
@MainActor
private enum Locator {
    private static let manager = CLLocationManager()

    static func current() async throws -> CLLocation {
        if manager.authorizationStatus == .notDetermined {
            manager.requestWhenInUseAuthorization()
        }
        let deadline = ContinuousClock.now + .seconds(15)
        for try await update in CLLocationUpdate.liveUpdates() {
            if let location = update.location { return location }
            if [.denied, .restricted].contains(manager.authorizationStatus) {
                throw ToolFailure("They have not let Juno see where they are. It is in Settings.")
            }
            if ContinuousClock.now > deadline { break }
        }
        throw ToolFailure("The phone could not tell where it is just now.")
    }
}
