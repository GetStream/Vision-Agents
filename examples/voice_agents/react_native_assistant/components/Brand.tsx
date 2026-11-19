import { createContext, type PropsWithChildren, useContext, useMemo, useState } from 'react';

import { StreamTheme } from 'stream-chat-expo/ai-components';

/**
 * The brand the agent's UI wears: its accent, and how round its corners are. The settings in the
 * drawer choose one, and the conversation shows it at once: the A2UI renderer's theme and the
 * app's own components both read it. The agent writes the same UI whichever is chosen. The same
 * three as the iOS and web assistants.
 */
export type Brand = {
  id: 'stream' | 'ember' | 'nordic';
  title: string;
  subtitle: string;
  accent: string;
  /** The corners of a card. Smaller parts round less, in proportion. */
  cornerRadius: number;
};

export const BRANDS: readonly Brand[] = [
  {
    accent: '#005FFF',
    cornerRadius: 16,
    id: 'stream',
    subtitle: 'Electric blue accent · 16pt rounded cards',
    title: 'Stream Default',
  },
  {
    accent: '#D95D24',
    cornerRadius: 10,
    id: 'ember',
    subtitle: 'Warm terracotta accent · 10pt compact cards',
    title: 'Ember Hospitality',
  },
  {
    accent: '#0F766E',
    cornerRadius: 6,
    id: 'nordic',
    subtitle: 'Crisp teal accent · 6pt architectural edges',
    title: 'Nordic Mobility',
  },
];

type BrandContextValue = { brand: Brand; setBrand: (brand: Brand) => void };

const BrandContext = createContext<BrandContextValue>({
  brand: BRANDS[0],
  setBrand: () => undefined,
});

/** The brand the agent's UI wears, and how the settings change it. */
export const useBrand = () => useContext(BrandContext);

// The AI components' theme in a brand: the A2UI renderer's accent, and its corners.
const themeOf = ({ accent, cornerRadius }: Brand) => ({
  a2ui: {
    button: { borderRadius: cornerRadius * 0.75 },
    card: { borderRadius: cornerRadius },
    chip: { borderRadius: cornerRadius / 2 },
    colors: { accent },
    input: { borderRadius: cornerRadius / 2 },
  },
});

/**
 * The AI components' theme, in the brand chosen. Kept for as long as the app runs: like the
 * conversations, the choice starts afresh on the next launch.
 */
export const BrandProvider = ({ children }: PropsWithChildren) => {
  const [brand, setBrand] = useState(BRANDS[0]);
  const value = useMemo(() => ({ brand, setBrand }), [brand]);
  const style = useMemo(() => themeOf(brand), [brand]);
  return (
    <BrandContext.Provider value={value}>
      <StreamTheme style={style}>{children}</StreamTheme>
    </BrandContext.Provider>
  );
};
