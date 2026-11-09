/**
 * A field of a Stream Chat message that Stream does not define itself: top level in
 * stream-chat 9, under `custom` in 10. The top level is read first, and a `null` there is
 * read as absent.
 *
 * Read by shape rather than by version, so a message from either works, and so does one the
 * app handed over from a client this package never saw. Only what the message holds is read,
 * never what every object inherits.
 */
export function messageField(message: object, key: string): unknown {
  const top = own(message, key);
  if (top !== undefined && top !== null) {
    return top;
  }
  // In stream-chat 9, `custom` is just another field of the app's, and can be anything.
  const custom = own(message, "custom");
  return typeof custom === "object" && custom !== null && !Array.isArray(custom) ? own(custom, key) : undefined;
}

function own(holder: object, key: string): unknown {
  return Object.hasOwn(holder, key) ? (holder as Record<string, unknown>)[key] : undefined;
}
