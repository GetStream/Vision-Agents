import { ToolCallContext } from '@stream-io/stream-agents';

import { createDeviceTools } from '../agent/deviceTools';

jest.mock('stream-chat-expo/ai-components', () => jest.requireActual('./sdkMock').aiComponentsMock());

const call = (id: string) => new ToolCallContext({ id });
const signal = () => new AbortController().signal;

describe('createDeviceTools', () => {
  it('declares the phone clock on the session, with its step title', () => {
    const tools = createDeviceTools({ clientId: 'install', userId: 'user' }).sessionTools();
    expect(tools.has('get_local_context')).toBe(true);
    expect(tools.declared()).toEqual([
      expect.objectContaining({ display_title: "Checking your phone's time", name: 'get_local_context' }),
    ]);
  });

  it('runs a call itself when no step addresses it to this phone', async () => {
    const deviceTools = createDeviceTools({ clientId: 'install', userId: 'user' });
    const output = await deviceTools.sessionTools().call('get_local_context', '{}', signal(), call('c1'));
    expect(JSON.parse(output as string)).toEqual(
      expect.objectContaining({ timeZone: expect.any(String), utc: expect.any(String) }),
    );
  });

  it('answers an addressed call with what the runner reported, before or after it is asked', async () => {
    const deviceTools = createDeviceTools({ clientId: 'install', userId: 'user' });
    deviceTools.addressed = true;
    const tools = deviceTools.sessionTools();

    await deviceTools.send('early', { output: { from: 'runner' } });
    await expect(tools.call('get_local_context', '{}', signal(), call('early'))).resolves.toBe(
      JSON.stringify({ from: 'runner' }),
    );

    const late = tools.call('get_local_context', '{}', signal(), call('late'));
    await deviceTools.send('late', { output: { from: 'later' } });
    await expect(late).resolves.toBe(JSON.stringify({ from: 'later' }));
  });

  it('fails a call the runner reported as failed', async () => {
    const deviceTools = createDeviceTools({ clientId: 'install', userId: 'user' });
    deviceTools.addressed = true;
    const asked = deviceTools.sessionTools().call('get_local_context', '{}', signal(), call('c1'));
    await deviceTools.send('c1', { failure: 'Location not shared' });
    await expect(asked).rejects.toThrow('Location not shared');
  });
});
