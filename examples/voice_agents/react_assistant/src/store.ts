import { useSyncExternalStore } from "react";

/**
 * A value views follow: `get` and `subscribe`, bound, for `useSyncExternalStore`, as the SDK's own
 * `session.state` and `session.conversation` are.
 */
export class Store<T> {
  private value: T;
  private readonly listeners = new Set<() => void>();

  constructor(initial: T) {
    this.value = initial;
  }

  readonly get = (): T => this.value;

  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  /** Merges `patch` in and tells every listener. */
  update(patch: Partial<T>): void {
    this.value = { ...this.value, ...patch };
    for (const listener of [...this.listeners]) {
      listener();
    }
  }
}

/**
 * One part of what a `get`/`subscribe` pair holds, kept current. The view redraws only when that
 * part changes, so `select` returns a value that stays the same object while it is unchanged.
 */
export function useSelected<T, S>(
  store: { get: () => T; subscribe: (listener: () => void) => () => void },
  select: (value: T) => S,
): S {
  return useSyncExternalStore(store.subscribe, () => select(store.get()));
}

/** What any `get`/`subscribe` pair holds now, kept current. */
export function useStore<T>(store: { get: () => T; subscribe: (listener: () => void) => () => void }): T {
  return useSyncExternalStore(store.subscribe, store.get);
}
