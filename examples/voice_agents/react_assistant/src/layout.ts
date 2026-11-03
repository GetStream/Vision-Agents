import { useCallback, useLayoutEffect, useState, useSyncExternalStore } from "react";

/** Wide enough for the conversations to sit beside the chat rather than slide over it. */
export const WIDE = "(min-width: 768px)";

/** The system's dark appearance, which the page follows until the person picks one. */
const DARK = "(prefers-color-scheme: dark)";

export type Theme = "light" | "dark";

const THEME_KEY = "juno.theme";

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

/** The appearance the person picked, or undefined while the page follows the system. */
export function storedTheme(): Theme | undefined {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    return stored === "light" || stored === "dark" ? stored : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Puts the page in `theme`: styles.css reads it from the root element, over the system's. Done
 * before the first draw for a stored choice, so a dark page does not flash light.
 */
export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
}

/** The page's appearance: the system's until the person picks one, which is kept for next time. */
export function useTheme(): [Theme, (theme: Theme) => void] {
  const system: Theme = useMediaQuery(DARK) ? "dark" : "light";
  const [chosen, setChosen] = useState(storedTheme);
  const theme = chosen ?? system;

  useLayoutEffect(() => applyTheme(theme), [theme]);

  const choose = useCallback((next: Theme) => {
    setChosen(next);
    try {
      localStorage.setItem(THEME_KEY, next);
    } catch {
      // Kept for this page only.
    }
  }, []);
  return [theme, choose];
}
