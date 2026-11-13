import { ToolCallContext } from '@stream-io/stream-agents';

import { crewBackend } from '../agent/agents';
import { createDeviceTools } from '../agent/deviceTools';

jest.mock('stream-chat-expo/ai-components', () => jest.requireActual('./sdkMock').aiComponentsMock());
jest.mock('../agent/agents', () => ({ crewBackend: jest.fn(async () => ({ id: 'ticket-308' })) }));

const call = (id: string) => new ToolCallContext({ id });
const signal = () => new AbortController().signal;
const create = () => createDeviceTools({ clientId: 'install', userId: 'crew-jordan' });

describe('createDeviceTools', () => {
  it('asks the person first for every tool that acts in their name, and only for those', () => {
    const tools = create().sessionTools();
    expect(tools.approval('get_local_context')).toBeUndefined();
    for (const name of ['request_shift_swap', 'create_support_ticket', 'book_check_in']) {
      expect(tools.approval(name)).toEqual(expect.objectContaining({ title: expect.any(String) }));
    }
  });

  it('titles the step of every tool it declares', () => {
    const declared = create().sessionTools().declared();
    expect(declared).toHaveLength(4);
    for (const tool of declared) {
      expect(tool).toEqual(expect.objectContaining({ display_title: expect.any(String) }));
    }
  });

  it('drafts a ticket through the backend with the arguments the agent wrote', async () => {
    const args = { asset: 'FRY-02', details: 'E-41 after a reset', summary: 'Fryer down' };
    const output = await create()
      .sessionTools()
      .call('create_support_ticket', JSON.stringify(args), signal(), call('c1'));
    expect(crewBackend).toHaveBeenCalledWith('crew/tickets', { body: args });
    expect(JSON.parse(output as string)).toEqual({ id: 'ticket-308' });
  });

  it('answers an addressed call with what the runner reported', async () => {
    const deviceTools = create();
    deviceTools.addressed = true;
    const asked = deviceTools.sessionTools().call('get_local_context', '{}', signal(), call('c2'));
    await deviceTools.send('c2', { output: { timeZone: 'Europe/Skopje' } });
    await expect(asked).resolves.toBe(JSON.stringify({ timeZone: 'Europe/Skopje' }));
  });
});
