import { imagesOf } from '../agent/photos';

jest.mock('expo-image-manipulator', () => ({
  manipulateAsync: jest.fn(async (uri: string) => ({ base64: `b64(${uri})` })),
  SaveFormat: { JPEG: 'jpeg' },
}));

describe('imagesOf', () => {
  it('sends the photos as JPEG data URIs and leaves other files out', async () => {
    const images = await imagesOf([
      { type: 'image/png', uri: 'file:///a.png' },
      { type: 'application/pdf', uri: 'file:///b.pdf' },
      { uri: 'file:///c' },
    ]);
    expect(images).toEqual([
      { url: 'data:image/jpeg;base64,b64(file:///a.png)' },
      { url: 'data:image/jpeg;base64,b64(file:///c)' },
    ]);
  });
});
