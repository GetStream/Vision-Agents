/**
 * Juno's calendar in this browser: the reminders the agent added, and an alert when each is due.
 *
 * A browser has no calendar or reminders an app can read or add to, as a phone does, so the
 * page keeps its own, in localStorage. It answers `list_events` and takes what `add_reminder`
 * adds, and a reminder alerts with a notification at its time while a Juno tab is open.
 */
export interface Reminder {
  readonly id: string;
  readonly title: string;
  /** When it is due, in milliseconds since the epoch. */
  readonly due: number;
}

const KEY = "juno.reminders";
const timers = new Map<string, ReturnType<typeof setTimeout>>();

/** Every reminder kept here, soonest first. */
export function reminders(): Reminder[] {
  try {
    const kept: unknown = JSON.parse(localStorage.getItem(KEY) ?? "[]");
    return Array.isArray(kept)
      ? (kept as Reminder[]).filter((each) => typeof each?.title === "string" && typeof each.due === "number").sort((a, b) => a.due - b.due)
      : [];
  } catch {
    return [];
  }
}

/** The reminders due on the local day that starts at `start`. */
export function remindersOn(start: Date): Reminder[] {
  const end = new Date(start.getFullYear(), start.getMonth(), start.getDate() + 1).getTime();
  return reminders().filter((each) => each.due >= start.getTime() && each.due < end);
}

/** Keeps a reminder and sets its alert. */
export function addReminder(title: string, due: Date): Reminder {
  const reminder: Reminder = { id: crypto.randomUUID(), title, due: due.getTime() };
  localStorage.setItem(KEY, JSON.stringify([...reminders(), reminder]));
  schedule(reminder);
  return reminder;
}

/** Sets the alert of every reminder still to come, which is what opening the page does. */
export function scheduleReminders(): void {
  for (const reminder of reminders()) {
    schedule(reminder);
  }
}

function schedule(reminder: Reminder): void {
  const wait = reminder.due - Date.now();
  // A timer longer than about 24 days fires at once, so one that far off waits for a later visit.
  if (wait <= 0 || wait > 2 ** 31 - 1 || timers.has(reminder.id)) {
    return;
  }
  timers.set(
    reminder.id,
    setTimeout(() => {
      timers.delete(reminder.id);
      if ("Notification" in window && Notification.permission === "granted") {
        new Notification(reminder.title, { body: "Juno's reminder", tag: reminder.id });
      }
    }, wait),
  );
}
