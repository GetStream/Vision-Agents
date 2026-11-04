import EventKit
import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  private let calendar = CalendarChannel()

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
    if let registrar = engineBridge.pluginRegistry.registrar(forPlugin: "CalendarChannel") {
      calendar.register(with: registrar.messenger())
    }
  }
}

/// The calendar and the reminders, for the agent's list_events and add_reminder tools.
///
/// EventKit, as swift_assistant's tools use it. What the phone could not do comes back as a
/// `tool_failure` error whose message the model reads.
final class CalendarChannel {
  private let store = EKEventStore()
  private var channel: FlutterMethodChannel?

  func register(with messenger: FlutterBinaryMessenger) {
    let channel = FlutterMethodChannel(
      name: "io.getstream.visionagents.flutterassistant/calendar", binaryMessenger: messenger)
    channel.setMethodCallHandler { [weak self] call, result in
      self?.handle(call, result: result)
    }
    self.channel = channel
  }

  private func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    let arguments = call.arguments as? [String: Any] ?? [:]
    Task { @MainActor in
      do {
        switch call.method {
        case "listEvents":
          result(try await events(on: arguments["day"] as? String ?? ""))
        case "addReminder":
          let due = (arguments["due"] as? NSNumber).map {
            Date(timeIntervalSince1970: $0.doubleValue / 1000)
          }
          try await addReminder(title: arguments["title"] as? String ?? "", due: due)
          result(nil)
        default:
          result(FlutterMethodNotImplemented)
        }
      } catch let failure as ToolFailure {
        result(FlutterError(code: "tool_failure", message: failure.message, details: nil))
      } catch {
        result(FlutterError(code: "tool_failure", message: error.localizedDescription, details: nil))
      }
    }
  }

  /// What is on the calendar for one day, each as a line: its times or All day, its title,
  /// and where.
  private func events(on day: String) async throws -> [String] {
    let isoDay = DateFormatter()
    isoDay.locale = Locale(identifier: "en_US_POSIX")
    isoDay.dateFormat = "yyyy-MM-dd"
    guard let start = isoDay.date(from: day) else {
      throw ToolFailure("The day has to be a date like 2026-10-16.")
    }
    let allowed: Bool
    if #available(iOS 17, *) {
      allowed = try await store.requestFullAccessToEvents()
    } else {
      allowed = try await store.requestAccess(to: .event)
    }
    guard allowed else {
      throw ToolFailure("They have not let Juno see their calendar. It is in Settings.")
    }
    let end = Calendar.current.date(byAdding: .day, value: 1, to: start) ?? start
    let found = store.events(
      matching: store.predicateForEvents(withStart: start, end: end, calendars: nil))
    let time = DateFormatter()
    time.dateStyle = .none
    time.timeStyle = .short
    return found.map { event in
      let when =
        event.isAllDay
        ? "All day"
        : "\(time.string(from: event.startDate))–\(time.string(from: event.endDate))"
      let place = event.location.map { " at \($0)" } ?? ""
      return "\(when): \(event.title ?? "Busy")\(place)"
    }
  }

  /// Adds a reminder to the person's Reminders, with an alert when it is due.
  private func addReminder(title: String, due: Date?) async throws {
    let allowed: Bool
    if #available(iOS 17, *) {
      allowed = try await store.requestFullAccessToReminders()
    } else {
      allowed = try await store.requestAccess(to: .reminder)
    }
    guard allowed else {
      throw ToolFailure("They have not let Juno add reminders. It is in Settings.")
    }
    let reminder = EKReminder(eventStore: store)
    reminder.title = title
    reminder.calendar = store.defaultCalendarForNewReminders()
    if let due {
      reminder.dueDateComponents = Calendar.current.dateComponents(
        [.year, .month, .day, .hour, .minute], from: due)
      reminder.addAlarm(EKAlarm(absoluteDate: due))
    }
    try store.save(reminder, commit: true)
  }
}

/// Why a tool could not answer, in words the model passes on.
struct ToolFailure: Error {
  let message: String

  init(_ message: String) {
    self.message = message
  }
}
