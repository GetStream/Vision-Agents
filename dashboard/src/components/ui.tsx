import Link from "next/link";
import type { ReactNode } from "react";

export function PageHeading({
  title,
  description,
  action,
}: {
  title: string;
  description?: string;
  action?: ReactNode;
}) {
  return (
    <div className="mb-6 flex items-start justify-between gap-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {description ? (
          <p className="mt-1 text-sm text-muted">{description}</p>
        ) : null}
      </div>
      {action}
    </div>
  );
}

export function Panel({
  title,
  aside,
  children,
  className = "",
}: {
  title?: string;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-xl border border-line bg-surface ${className}`}
    >
      {title ? (
        <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-3">
          <h2 className="text-sm font-medium">{title}</h2>
          {aside}
        </header>
      ) : null}
      {children}
    </section>
  );
}

/**
 * Section is a panel that folds away.
 *
 * Configuring an agent is four unrelated jobs, and all four open at once is a page nobody
 * can find anything on. The open one is the caller's business rather than the section's,
 * so several can be open together and reloading the page does not close them.
 */
export function Section({
  title,
  description,
  open,
  onToggle,
  aside,
  children,
  className = "",
}: {
  title: string;
  description?: string;
  open: boolean;
  onToggle: () => void;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-xl border border-line bg-surface ${className}`}>
      <header className="flex items-center gap-3 px-4 py-3">
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={open}
          className="flex min-w-0 flex-1 items-center gap-3 text-left"
        >
          <span
            aria-hidden
            className={`text-xs text-muted transition-transform ${open ? "rotate-90" : ""}`}
          >
            ▶
          </span>
          <span className="min-w-0">
            <span className="block text-sm font-medium">{title}</span>
            {description ? (
              <span className="mt-0.5 block text-xs text-muted">{description}</span>
            ) : null}
          </span>
        </button>
        {aside}
      </header>
      {open ? <div className="border-t border-line">{children}</div> : null}
    </section>
  );
}

/**
 * Tabs switches between two views of the same thing.
 *
 * Which one is showing is the caller's business rather than the tabs': a page that keeps it
 * in the URL and a page that keeps it in state both want the same strip of labels.
 */
export function Tabs({
  tabs,
  active,
  onSelect,
}: {
  tabs: { id: string; label: string }[];
  active: string;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="mb-4 flex gap-1 border-b border-line">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          onClick={() => onSelect(tab.id)}
          className={`-mb-px border-b-2 px-3 py-2 text-sm transition ${
            tab.id === active
              ? "border-foreground font-medium text-foreground"
              : "border-transparent text-muted hover:text-foreground"
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

export function Tile({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-xl border border-line bg-surface px-4 py-3">
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
      {hint ? <div className="mt-0.5 text-xs text-muted">{hint}</div> : null}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="px-4 py-8 text-center text-sm text-muted">{children}</p>;
}

export function Notice({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-xl border border-line bg-surface px-4 py-3 text-sm text-muted ${className}`}
    >
      {children}
    </div>
  );
}

export function Failure({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <p className="rounded-xl border border-line bg-surface px-4 py-6 text-sm text-muted">
      The router did not answer: {message}
    </p>
  );
}

export function Button({
  children,
  onClick,
  type = "button",
  variant = "primary",
  disabled = false,
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  type?: "button" | "submit";
  variant?: "primary" | "quiet" | "danger";
  disabled?: boolean;
  /** Why the button is disabled, which the button itself cannot say. */
  title?: string;
}) {
  const styles = {
    primary: "bg-foreground text-background hover:opacity-90",
    quiet: "border border-line hover:bg-line/40",
    danger: "border border-line text-red-600 hover:bg-red-500/10",
  }[variant];

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`rounded-lg px-3 py-1.5 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${styles}`}
    >
      {children}
    </button>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-muted">{label}</span>
      {children}
      {hint ? <span className="mt-1 block text-xs text-muted">{hint}</span> : null}
    </label>
  );
}

export const inputStyle =
  "mt-1 w-full rounded-lg border border-line bg-background px-3 py-1.5 text-sm outline-none focus:border-foreground/40";

export function CallLink({ id, children }: { id: string; children: ReactNode }) {
  return (
    <Link href={`/calls/${id}`} className="hover:underline">
      {children}
    </Link>
  );
}

/** ms renders a latency the way somebody reading a call scans it. */
export function ms(value: number | null | undefined): string {
  if (value === null || value === undefined || value <= 0) {
    return "—";
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(2)}s`;
  }
  return `${Math.round(value)}ms`;
}

/** duration is how long a call ran, or how long it has been running. */
export function duration(from: string, to?: string | null): string {
  const seconds = Math.max(
    0,
    Math.round(
      ((to ? new Date(to).getTime() : Date.now()) - new Date(from).getTime()) /
        1000,
    ),
  );
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
}

export function clock(at: string | number): string {
  return new Date(at).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}
