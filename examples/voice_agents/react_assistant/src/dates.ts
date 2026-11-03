/** A day as YYYY-MM-DD, in the person's timezone. */
export function isoDay(date: Date): string {
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

/** The local day a YYYY-MM-DD names, or undefined for anything else. */
export function parseDay(day: string): Date | undefined {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(day);
  if (!match) return undefined;
  const date = new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
  return isoDay(date) === day ? date : undefined;
}
