import { useEffect, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import { type ReplyFormat, replyFormat, setReplyFormat } from '../agent/replyFormat';
import { BRANDS, useBrand } from '../components/Brand';

// The formats this app shows. OpenUI is the iOS and web apps'.
const FORMATS: { description: string; format: ReplyFormat; title: string }[] = [
  { description: 'Answers in Markdown', format: 'markdown', title: 'Text' },
  {
    description: 'Answers with cards, maps and forms, in A2UI',
    format: 'a2ui',
    title: 'Interactive',
  },
];

type ChoiceProps = {
  chosen: boolean;
  description: string;
  disabled?: boolean;
  onPress: () => void;
  title: string;
};

const Choice = ({ chosen, description, disabled = false, onPress, title }: ChoiceProps) => (
  <Pressable
    accessibilityRole='button'
    accessibilityState={{ disabled, selected: chosen }}
    disabled={disabled}
    onPress={onPress}
    style={({ pressed }) => [styles.choice, { opacity: disabled ? 0.4 : pressed ? 0.6 : 1 }]}
  >
    <View style={styles.choiceText}>
      <Text style={styles.choiceTitle}>{title}</Text>
      <Text style={styles.choiceDescription}>{description}</Text>
    </View>
    {chosen ? <Text style={styles.check}>✓</Text> : null}
  </Pressable>
);

/**
 * The app's settings, under the conversations: how the agent answers, which the backend keeps
 * for every app, and the brand its UI wears on this phone.
 */
export const Settings = () => {
  const { brand, setBrand } = useBrand();
  // The format the backend has, once read or saved: a choice shows only what was confirmed.
  const [format, setFormat] = useState<ReplyFormat>();
  const [failure, setFailure] = useState<string>();
  // While the first read or a save is on its way, the choices wait for it, so two answers can't
  // land out of order.
  const [busy, setBusy] = useState(true);

  useEffect(() => {
    let live = true;
    replyFormat()
      .then(
        (current) => live && setFormat(current),
        (cause: unknown) => live && setFailure(String((cause as Error)?.message ?? cause)),
      )
      .finally(() => live && setBusy(false));
    return () => {
      live = false;
    };
  }, []);

  const choose = async (chosen: ReplyFormat) => {
    setBusy(true);
    setFailure(undefined);
    try {
      await setReplyFormat(chosen);
      setFormat(chosen);
    } catch (cause) {
      setFailure(String((cause as Error)?.message ?? cause));
      // The backend may have taken the change before its answer was lost: show what it has.
      setFormat(undefined);
      await replyFormat().then(setFormat, () => undefined);
    } finally {
      setBusy(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.heading}>Answers</Text>
      {FORMATS.map((item) => (
        <Choice
          chosen={item.format === format}
          description={item.description}
          disabled={busy}
          key={item.format}
          onPress={() => choose(item.format)}
          title={item.title}
        />
      ))}
      {format === 'openui' ? (
        <Text style={styles.footnote}>
          Another app chose OpenUI, which this app shows as text. Pick one above to change it.
        </Text>
      ) : null}
      <Text style={styles.footnote}>From the next conversation on.</Text>
      {failure ? <Text style={styles.failure}>{failure}</Text> : null}
      <Text style={styles.heading}>Brand</Text>
      {BRANDS.map((item) => (
        <Choice
          chosen={item.id === brand.id}
          description={item.subtitle}
          key={item.id}
          onPress={() => setBrand(item)}
          title={item.title}
        />
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  check: { color: '#005FFF', fontSize: 17, fontWeight: '700' },
  choice: { alignItems: 'center', flexDirection: 'row', gap: 8, paddingVertical: 6 },
  choiceDescription: { color: '#7a7a7a', fontSize: 12 },
  choiceText: { flex: 1 },
  choiceTitle: { fontSize: 15 },
  container: {
    borderTopColor: 'grey',
    borderTopWidth: 1,
    marginHorizontal: 12,
    paddingBottom: 8,
  },
  failure: { color: '#B00020', fontSize: 12 },
  footnote: { color: '#7a7a7a', fontSize: 12 },
  heading: { fontSize: 15, fontWeight: 'bold', paddingBottom: 4, paddingTop: 12 },
});
