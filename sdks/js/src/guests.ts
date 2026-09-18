import type { StreamUser } from "./backend.js";
import type { Client, Schemas } from "./client.js";
import { ConfigurationError } from "./errors.js";

/** Where a guest is remembered between page loads. */
export const GUEST_STORAGE_KEY = "stream-vision-agents-guest";

/** How long the cookie a guest is remembered in lasts, in days. */
const COOKIE_DAYS = 30;

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
 * The remembered token is not checked here. It is a JWT the router verifies, and verifying
 * it in the page would mean shipping the key to do it; a token that has expired fails on the
 * next request with a 401, which is the same thing a stale session of any kind does.
 */
export async function guestUser(
  client: Client,
  options: GuestUserOptions = {},
  store: GuestStore | undefined = browserStore(),
): Promise<Guest> {
  if (!options.fresh) {
    const held = read(store);
    if (held) {
      return held;
    }
  }

  const minted = await client.post("/v1/agents/guests", {
    body: {
      ...(options.name ? { name: options.name } : {}),
      ...(options.custom ? { custom: options.custom } : {}),
    },
  });
  store?.write(JSON.stringify(minted));
  return minted;
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
    return cookieStore(document);
  }
  const storage = (globalThis as { localStorage?: Storage }).localStorage;
  if (storage) {
    return {
      read: () => storage.getItem(GUEST_STORAGE_KEY) ?? undefined,
      write: (value) => storage.setItem(GUEST_STORAGE_KEY, value),
      clear: () => storage.removeItem(GUEST_STORAGE_KEY),
    };
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
