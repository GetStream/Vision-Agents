/** Conversations under Today, Yesterday, This week and then their month, in the router's order. */
export interface Day<T> {
  readonly title: string;
  readonly items: readonly T[];
}

/**
 * The items grouped by the day each started, keeping their order. A day can appear twice: the
 * list is in order of activity, and the days are when each conversation started.
 */
export function daySections<T>(items: readonly T[], startedAt: (item: T) => Date, now = new Date()): Day<T>[] {
  const days: { title: string; items: T[] }[] = [];
  for (const item of items) {
    const title = dayTitle(startedAt(item), now);
    const last = days[days.length - 1];
    if (last?.title === title) {
      last.items.push(item);
    } else {
      days.push({ title, items: [item] });
    }
  }
  return days;
}

export function dayTitle(date: Date, now = new Date()): string {
  const days = Math.round((startOfDay(now) - startOfDay(date)) / 86_400_000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return "This week";
  return date.toLocaleDateString(undefined, { month: "long", year: "numeric" });
}

function startOfDay(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}
