import { Tools } from "@stream-io/calypso";
import { z } from "zod";

import { isoDay, parseDay } from "./dates";
import { sendEmail } from "./gmail";
import { addReminder, remindersOn } from "./reminders";

/**
 * What the agent can ask this page: the time and place, Juno's calendar and reminders, and to
 * send an email from the person's Gmail. The same tools, by the same names, as the iOS and
 * Android apps.
 *
 * The agent runs in the backend and knows none of this. Each tool runs here, when the model
 * calls it, and only its answer leaves the page. The ones that share or do something personal
 * ask first: the call waits in the session's `approvals` until the person answers, on the
 * reply's step in a chat or on a card over a call. A permission the browser needs is asked for
 * the first time a tool needs it.
 */
export function deviceTools(): Tools {
  return new Tools()
    .register({
      name: "get_local_context",
      description:
        "The person's current date, local time, timezone and language. Call it before answering " +
        "anything that depends on today or the time.",
      // It takes no arguments, and still declares an empty object of them: a tool sent with no
      // parameters at all is refused by xAI, which fails the whole reply.
      parameters: z.object({}),
      run: () => localContext(new Date()),
    })
    .register({
      name: "get_location",
      description:
        "Where the person is: their street, neighbourhood and city. Only call it when the answer " +
        "depends on where they are, such as the weather or places nearby.",
      parameters: z.object({ reason: z.string().describe("why you need it, in a few words they would recognise") }),
      displayTitle: "Checking where you are",
      approval: {
        title: "Share your location?",
        message: "Juno gets your street and city, once.",
        reasonArgument: "reason",
        allowTitle: "Share",
        declineTitle: "Not now",
      },
      run: () => location(),
    })
    .register({
      name: "list_events",
      description: "The events on the person's calendar for one day.",
      parameters: z.object({ day: z.string().describe("the day as YYYY-MM-DD, in their timezone") }),
      displayTitle: "Checking your calendar",
      run: ({ day }) => events(day),
    })
    .register({
      name: "add_reminder",
      description: "Adds a reminder to the person's calendar, with an alert when it is due.",
      parameters: z.object({
        title: z.string().describe("what to remind them of, such as Call mum"),
        due: z
          .string()
          .describe("when, as an ISO 8601 date and time with their UTC offset, such as 2026-10-16T18:00:00+02:00."),
        summary: z.string().describe("the reminder and its time as they would read it, such as Call mum, today at 6pm"),
      }),
      displayTitle: "Adding a reminder",
      approval: {
        title: "Add this reminder?",
        message: "It goes in Juno's calendar, with an alert when it is due.",
        reasonArgument: "summary",
        allowTitle: "Add reminder",
        declineTitle: "Not now",
      },
      run: ({ title, due }) => reminder(title, due),
    })
    .register({
      name: "send_email",
      description:
        "Sends an email from the person's Gmail. Only call it once they have asked for it to be " +
        "sent, after you have shown them the subject and the body.",
      parameters: z.object({
        to: z.string().describe("who it is to: one or more email addresses, separated by commas"),
        subject: z.string().describe("the subject line"),
        body: z.string().describe("the email itself, in plain text"),
      }),
      displayTitle: "Sending your email",
      approval: {
        title: "Send this email?",
        message: "Juno sends it from your Gmail, as you.",
        reasonArgument: "to",
        allowTitle: "Send",
        declineTitle: "Not now",
      },
      run: async ({ to, subject, body }) => {
        if (!to || !body) {
          throw new Error("An email needs someone to go to and a body.");
        }
        await sendEmail(to, subject, body);
        return `Sent "${subject}" to ${to} from their Gmail.`;
      },
    });
}

/** The date, time and timezone, which a model in a data centre has no way to know. */
export function localContext(now: Date): string {
  const zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
  const day = now.toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
  const time = now.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  const abbreviation =
    new Intl.DateTimeFormat("en-US", { timeZoneName: "short" }).formatToParts(now).find((part) => part.type === "timeZoneName")
      ?.value ?? "";
  return (
    `It is ${day}, ${time} in ${zone} (${abbreviation}). ` +
    `Today as a date is ${isoDay(now)}. ` +
    `Their language is ${navigator.language}.`
  );
}

/** Where the person is, to the street. */
async function location(): Promise<string> {
  const here = await new Promise<GeolocationPosition>((resolve, reject) => {
    if (!("geolocation" in navigator)) {
      reject(new Error("This browser cannot tell where it is."));
      return;
    }
    navigator.geolocation.getCurrentPosition(resolve, (error) => {
      reject(
        new Error(
          error.code === error.PERMISSION_DENIED
            ? "They have not let Juno see where they are. It is in the browser's site settings."
            : "The browser could not tell where it is just now.",
        ),
      );
    }, { timeout: 15_000, maximumAge: 60_000 });
  });
  const { latitude, longitude } = here.coords;
  const place = await address(latitude, longitude).catch(() => undefined);
  if (!place) {
    return `They are at latitude ${latitude}, longitude ${longitude}, which has no address.`;
  }
  return `They are on ${place}.`;
}

/**
 * The street, neighbourhood, city and country at a point, from OpenStreetMap's Nominatim: the
 * browser has no geocoder of its own, as a phone does.
 */
async function address(latitude: number, longitude: number): Promise<string | undefined> {
  const url = new URL("https://nominatim.openstreetmap.org/reverse");
  url.search = new URLSearchParams({ format: "jsonv2", lat: String(latitude), lon: String(longitude), zoom: "18" }).toString();
  const response = await fetch(url, { headers: { "Accept-Language": navigator.language } });
  if (!response.ok) return undefined;
  const found = (await response.json()) as { address?: Record<string, string> };
  const parts = found.address ?? {};
  const street = parts["road"] ?? parts["pedestrian"];
  const area = parts["suburb"] ?? parts["neighbourhood"] ?? parts["quarter"];
  const city = parts["city"] ?? parts["town"] ?? parts["village"];
  const named = [street, area, city, parts["country"]].filter(Boolean);
  return named.length > 0 ? named.join(", ") : undefined;
}

/** What is on Juno's calendar for one day. */
function events(day: string): string {
  const start = parseDay(day);
  if (!start) {
    throw new Error("The day has to be a date like 2026-10-16.");
  }
  const found = remindersOn(start);
  if (found.length === 0) {
    return "Nothing is on their calendar that day.";
  }
  return found
    .map((each) => `${new Date(each.due).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}: ${each.title}`)
    .join("\n");
}

/** Adds a reminder, once the person said it may. */
function reminder(title: string, due: string): string {
  if (!title) {
    throw new Error("A reminder needs a title.");
  }
  const when = new Date(due);
  if (!/T\d{2}:\d{2}/.test(due) || Number.isNaN(when.getTime())) {
    throw new Error("A reminder needs a due time, such as 2026-10-16T18:00:00+02:00.");
  }
  addReminder(title, when);
  if (!("Notification" in window) || Notification.permission !== "granted") {
    return `Added the reminder "${title}". They have not let Juno show notifications, so it will not alert them; it is in the browser's site settings.`;
  }
  return `Added the reminder "${title}".`;
}
