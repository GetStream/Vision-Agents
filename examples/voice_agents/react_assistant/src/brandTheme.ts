import { useCallback, useEffect, useState } from "react";

export type BrandProfileId = "stream" | "ember" | "nordic";

export interface BrandProfile {
  readonly id: BrandProfileId;
  readonly title: string;
  readonly description: string;
  readonly accent: string;
  readonly radiusPx: number;
}

export const BRAND_PROFILES: readonly BrandProfile[] = [
  {
    id: "stream",
    title: "Stream Default",
    description: "Electric blue accent · 16px rounded cards",
    accent: "#005fff",
    radiusPx: 16,
  },
  {
    id: "ember",
    title: "Ember Hospitality",
    description: "Warm terracotta accent · 10px compact cards",
    accent: "#d95d24",
    radiusPx: 10,
  },
  {
    id: "nordic",
    title: "Nordic Mobility",
    description: "Crisp teal accent · 6px architectural edges",
    accent: "#0f766e",
    radiusPx: 6,
  },
];

const BRAND_STORAGE_KEY = "juno.brand-profile";

export function storedBrandProfile(): BrandProfileId {
  try {
    const raw = localStorage.getItem(BRAND_STORAGE_KEY);
    if (raw === "stream" || raw === "ember" || raw === "nordic") {
      return raw;
    }
  } catch {
    // Ignore restricted storage.
  }
  return "stream";
}

export function applyBrandProfile(id: BrandProfileId): void {
  const profile = BRAND_PROFILES.find((p) => p.id === id) ?? BRAND_PROFILES[0]!;
  const root = document.documentElement;
  root.dataset.brand = profile.id;
  root.style.setProperty("--accent", profile.accent);
  root.style.setProperty("--brand-radius", `${profile.radiusPx}px`);
  root.style.setProperty("--str-chat__ai-accent", profile.accent);
  root.style.setProperty("--str-chat__ai-link", profile.accent);
  root.style.setProperty("--str-chat__ai-button-primary-bg", profile.accent);
  root.style.setProperty("--str-chat__ai-voice-color", profile.accent);
}

export function useBrandProfile(): [BrandProfileId, (next: BrandProfileId) => void] {
  const [brand, setBrand] = useState<BrandProfileId>(storedBrandProfile);

  useEffect(() => {
    applyBrandProfile(brand);
  }, [brand]);

  const choose = useCallback((next: BrandProfileId) => {
    setBrand(next);
    applyBrandProfile(next);
    try {
      localStorage.setItem(BRAND_STORAGE_KEY, next);
    } catch {
      // Session-only fallback.
    }
  }, []);

  return [brand, choose];
}
