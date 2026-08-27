import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";

import { Providers } from "@/components/Providers";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Vision Agents",
  description: "Watch the calls, and read back why they went the way they did.",
};

const sections = [
  { href: "/", label: "Overview" },
  { href: "/agents", label: "Agents" },
  { href: "/voices", label: "Voices" },
  { href: "/telephony", label: "Telephony" },
];

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="flex min-h-full flex-col font-sans">
        <Providers>
          <header className="border-b border-line">
            <nav className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-4">
              <Link href="/" className="text-sm font-semibold tracking-tight">
                Vision Agents
              </Link>
              <div className="flex gap-4 text-sm text-muted">
                {sections.map((section) => (
                  <Link
                    key={section.href}
                    href={section.href}
                    className="transition-colors hover:text-foreground"
                  >
                    {section.label}
                  </Link>
                ))}
              </div>
            </nav>
          </header>
          <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-8">
            {children}
          </main>
        </Providers>
      </body>
    </html>
  );
}
