/**
 * Runs `body` as a page whose browser says it is in `zone`, then puts the globals back.
 *
 * A page is told apart by its `document`, so an empty one stands in for it. The zone is what
 * `Intl.DateTimeFormat().resolvedOptions().timeZone` answers, stubbed so a test does not
 * depend on the zone of the machine running it; an Error is thrown from there instead.
 */
export function inPage<T>(zone: unknown, body: () => Promise<T>): Promise<T> {
  return reporting(zone, true, body);
}

/** Runs `body` off a page, on a machine whose `Intl` says it is in `zone`. */
export function offPage<T>(zone: unknown, body: () => Promise<T>): Promise<T> {
  return reporting(zone, false, body);
}

async function reporting<T>(zone: unknown, page: boolean, body: () => Promise<T>): Promise<T> {
  const global = globalThis as { document?: unknown };
  const hadDocument = "document" in global;
  const document = global.document;
  const format = Intl.DateTimeFormat;
  Intl.DateTimeFormat = function () {
    return {
      resolvedOptions: () => {
        if (zone instanceof Error) {
          throw zone;
        }
        return { timeZone: zone };
      },
    };
  } as unknown as typeof Intl.DateTimeFormat;
  if (page) {
    global.document = {};
  } else {
    delete global.document;
  }
  try {
    return await body();
  } finally {
    Intl.DateTimeFormat = format;
    if (hadDocument) {
      global.document = document;
    } else {
      delete global.document;
    }
  }
}
