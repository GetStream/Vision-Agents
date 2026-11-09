import type { StreamUser } from "./backend.js";
import type { Client, Schemas } from "./client.js";
import { ConfigurationError } from "./errors.js";

/** Where a guest is remembered between page loads. */
export const GUEST_STORAGE_KEY = "stream-vision-agents-guest";

/**
 * What a page asking for a guest with no token is told when the router, or Stream's proxy in
 * front of it, admits no anonymous callers.
 */
export const ANONYMOUS_GUEST_REFUSED =
  "this router admits no anonymous callers, so a page cannot mint its own guest: mint it on " +
  "your backend with guestUser and pass it to setUser";

/** How long the cookie a guest is remembered in lasts, in days. */
const COOKIE_DAYS = 30;

const refreshing = new WeakMap<GuestStore, Map<string, Promise<Guest>>>();
const browserStores = new WeakMap<object, GuestStore>();

export interface GuestUserOptions {
  /** What to call them. Shown in chat and video, so "Guest" is the honest default. */
  name?: string;
  /** Anything of the caller's own to keep against them. */
  custom?: Record<string, unknown>;
  /**
   * Mint a new guest even if one is remembered here.
   *
   * For a "not me" button: the person at the keyboard is a different person, and reusing
   * the remembered guest would hand them the previous one's conversations.
   */
  fresh?: boolean;
}

/**
 * Somewhere to keep a guest between page loads.
 *
 * Narrow on purpose, so a caller can pass their own: a React Native app has no cookies and
 * no localStorage, and a server has neither and wants neither.
 */
export interface GuestStore {
  read(): string | undefined;
  write(value: string): void;
  clear(): void;
}

/** A guest as it was minted, which is the id and the token that proves it. */
export type Guest = Schemas["GuestUser"];

/**
 * Gets or creates a guest so somebody can talk to an agent before they sign up.
 *
 * Remembered where there is somewhere to remember it — a cookie, then localStorage — so
 * reloading the page is the same guest and not a second one with an empty history. On a
 * server nothing is remembered, because there is no "this person" there to remember: a
 * process handling two visitors would hand them each other's conversations.
 *
 * A page with only the app's key asks anonymously, which the router must admit; one that does
 * not throws a `ConfigurationError`, and the guest is minted by the app's backend instead.
 *
 * A remembered guest whose `expires_at` has passed comes back with their id and the
 * `refresh_secret` they were last handed, which the router answers with the same guest, a fresh
 * token and the next secret. The secret works once, so whatever comes back replaces what was
 * remembered; one already spent is answered with a new guest, since a guest id is no proof of
 * anything. Name and custom go along, kept by a returning guest and given to a new one. An
 * expired guest with no secret is minted afresh. One with no `expires_at` is handed back as it
 * is: its token is not verified here, which would mean shipping the key.
 */
export async function guestUser(
  client: Client,
  options: GuestUserOptions = {},
  store: GuestStore | undefined = browserStore(),
): Promise<Guest> {
  const held = options.fresh ? undefined : read(store);
  if (held && !expired(held)) {
    return held;
  }
  // An expired guest with no secret, remembered from before there were any, cannot prove who
  // they are; their token no longer works, so they start again as a new guest.
  const comingBack = held?.refresh_secret ? held : undefined;

  const key = JSON.stringify([client.backend.url, client.backend.apiKey, client.backend.customerId, held?.id]);
  let pending = store && !options.fresh ? refreshing.get(store) : undefined;
  if (store && !options.fresh && !pending) {
    pending = new Map();
    refreshing.set(store, pending);
  }
  const heldRefresh = pending?.get(key);
  if (heldRefresh) {
    return heldRefresh;
  }
  const minting = (async () => {
    const minted = await client.post("/v1/agents/guests", {
      body: {
        ...(comingBack ? { id: comingBack.id, refresh_secret: comingBack.refresh_secret } : {}),
        ...(options.name ? { name: options.name } : {}),
        ...(options.custom ? { custom: options.custom } : {}),
      },
    });
    store?.write(JSON.stringify(minted));
    return minted;
  })();
  pending?.set(key, minting);
  try {
    return await minting;
  } finally {
    pending?.delete(key);
  }
}

/**
 * Forgets the remembered guest without minting another.
 *
 * What a sign-out does: the account they signed into is who they are now, and the guest they
 * were before must not be handed to whoever uses the browser next.
 */
export function forgetGuest(store: GuestStore | undefined = browserStore()): void {
  store?.clear();
}

/**
 * Moves a guest's conversations onto the account they turned out to be.
 *
 * Server side only, and the one thing here that most needs to be: only the app's own backend
 * knows that a given guest is a given account, because it is the thing that just
 * authenticated them. A page able to ask this could claim anybody's conversations by guessing
 * a guest id, so the router refuses it from one — this throws first, with a reason, rather
 * than letting a 403 be the explanation.
 */
export function claimGuestUser(
  client: Client,
  guest: Guest | string,
  real: StreamUser | string,
): Promise<Schemas["ClaimGuestResult"]> {
  if (!client.backend.serverSide) {
    throw new ConfigurationError(
      "claiming a guest is server side only: it is the backend that just authenticated the " +
        "account that knows which guest it was",
    );
  }

  const guestId = typeof guest === "string" ? guest : guest.id;
  const userId = typeof real === "string" ? real : real.id;
  if (!guestId || !userId) {
    throw new ConfigurationError("claiming a guest needs the guest and the account");
  }

  return client.post("/v1/agents/guests/claim", {
    body: { guest_id: guestId, user_id: userId },
  });
}

function read(store: GuestStore | undefined): Guest | undefined {
  const held = store?.read();
  if (!held) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(held) as Guest;
    return parsed.id && parsed.token ? parsed : undefined;
  } catch {
    // Something else is under the key, or the value was truncated. Minting a fresh guest is
    // the recoverable answer; throwing would leave the page unable to ask anything at all.
    return undefined;
  }
}

/** Whether the router said when the guest's token stops working, and that time has passed. */
function expired(guest: Guest): boolean {
  const at = guest.expires_at ? Date.parse(guest.expires_at) : Number.NaN;
  return !Number.isNaN(at) && at <= Date.now();
}

/**
 * The browser's own storage, cookie first.
 *
 * A cookie outlives a tab and is shared across them, which is what "the same visitor" means
 * to a person. localStorage is the fallback for a context that has no cookies, and neither is
 * available on a server, where this is undefined and nothing is remembered.
 */
export function browserStore(): GuestStore | undefined {
  const document = (globalThis as { document?: { cookie: string } }).document;
  if (document) {
    let store = browserStores.get(document);
    if (!store) {
      store = cookieStore(document);
      browserStores.set(document, store);
    }
    return store;
  }
  const storage = (globalThis as { localStorage?: Storage }).localStorage;
  if (storage) {
    let store = browserStores.get(storage);
    if (!store) {
      store = {
        read: () => storage.getItem(GUEST_STORAGE_KEY) ?? undefined,
        write: (value) => storage.setItem(GUEST_STORAGE_KEY, value),
        clear: () => storage.removeItem(GUEST_STORAGE_KEY),
      };
      browserStores.set(storage, store);
    }
    return store;
  }
  return undefined;
}

function cookieStore(document: { cookie: string }): GuestStore {
  return {
    read() {
      for (const pair of document.cookie.split(";")) {
        const [name, ...rest] = pair.trim().split("=");
        if (name === GUEST_STORAGE_KEY) {
          return decodeURIComponent(rest.join("="));
        }
      }
      return undefined;
    },
    write(value) {
      const expires = new Date(Date.now() + COOKIE_DAYS * 86_400_000).toUTCString();
      // SameSite=Lax rather than Strict: the guest has to survive arriving from a link,
      // which is how most of them arrive. No Secure flag, because this has to work on a
      // localhost page too, and what the cookie holds is a token scoped to one guest.
      document.cookie =
        `${GUEST_STORAGE_KEY}=${encodeURIComponent(value)}; path=/; expires=${expires}; SameSite=Lax`;
    },
    clear() {
      document.cookie = `${GUEST_STORAGE_KEY}=; path=/; max-age=0; SameSite=Lax`;
    },
  };
}
