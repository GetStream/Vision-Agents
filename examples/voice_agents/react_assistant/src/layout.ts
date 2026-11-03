import { useCallback, useState, useSyncExternalStore } from "react";

/** Wide enough for the conversations to sit beside the chat rather than slide over it. */
export const WIDE = "(min-width: 768px)";

/** The system's dark appearance, which the page's own colours follow too. */
export const DARK = "(prefers-color-scheme: dark)";

/** Whether `query` matches the window now, kept current as it is resized. */
export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (listener: () => void) => {
      const list = window.matchMedia(query);
      list.addEventListener("change", listener);
      return () => list.removeEventListener("change", listener);
    },
    [query],
  );
  return useSyncExternalStore(subscribe, () => window.matchMedia(query).matches);
}

/**
 * A choice the person made that the page keeps for next time, in localStorage. Storage can be
 * blocked, in a private window say, and the choice then lasts as long as the page.
 */
export function useStoredFlag(key: string, initial: boolean): [boolean, (value: boolean) => void] {
  const [value, setValue] = useState(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored === null ? initial : stored === "true";
    } catch {
      return initial;
    }
  });
  const store = useCallback(
    (next: boolean) => {
      setValue(next);
      try {
        localStorage.setItem(key, String(next));
      } catch {
        // Kept for this page only.
      }
    },
    [key],
  );
  return [value, store];
}
