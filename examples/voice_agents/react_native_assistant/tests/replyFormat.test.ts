import { formatSaved, setReplyFormat } from '../agent/replyFormat';

jest.mock('../agent/config', () => ({ AgentConfig: { tokenUrl: 'http://backend' } }));

// A PUT the test answers when it wants, as a backend that applies the change before it answers.
const heldPut = () => {
  let answer: (response: { ok: boolean; status: number }) => void = () => undefined;
  global.fetch = jest.fn(
    () =>
      new Promise((resolve) => {
        answer = resolve;
      }),
  ) as unknown as typeof fetch;
  return (ok: boolean) => answer({ ok, status: ok ? 200 : 502 });
};

const settles = async (promise: Promise<unknown>) => {
  let settled = false;
  void promise.then(() => {
    settled = true;
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  return settled;
};

describe('formatSaved', () => {
  it('holds a new conversation until the change of format is answered', async () => {
    const answer = heldPut();
    const saving = setReplyFormat('a2ui');
    expect(await settles(formatSaved())).toBe(false);

    answer(true);
    await saving;
    expect(await settles(formatSaved())).toBe(true);
  });

  it('lets a conversation start after a change that failed', async () => {
    const answer = heldPut();
    const saving = setReplyFormat('markdown');
    answer(false);
    await expect(saving).rejects.toThrow('could not change');
    expect(await settles(formatSaved())).toBe(true);
  });
});
