// Hermes has no `crypto.randomUUID`. Ported from stream-chat-js's `generateUUIDv4`.
/* eslint-disable no-bitwise -- RFC 4122 version and variant bits */

type CryptoLike = {
  getRandomValues?: (bytes: Uint8Array) => Uint8Array;
  randomUUID?: () => string;
};

const globalWithCrypto = globalThis as { crypto?: CryptoLike };

const getRandomValuesWithMathRandom = (bytes: Uint8Array) => {
  const max = Math.pow(2, (8 * bytes.byteLength) / bytes.length);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = Math.random() * max;
  }
  return bytes;
};

const hex = (bytes: Uint8Array) => {
  let s = '';
  for (let i = 0; i < bytes.length; i++) {
    s += bytes[i].toString(16).padStart(2, '0');
  }
  return s;
};

// https://tools.ietf.org/html/rfc4122
const generateUUIDv4 = () => {
  const getRandomValues =
    globalWithCrypto.crypto?.getRandomValues?.bind(globalWithCrypto.crypto) ??
    getRandomValuesWithMathRandom;
  const bytes = getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40; // version
  bytes[8] = (bytes[8] & 0xbf) | 0x80; // variant

  return (
    hex(bytes.subarray(0, 4)) +
    '-' +
    hex(bytes.subarray(4, 6)) +
    '-' +
    hex(bytes.subarray(6, 8)) +
    '-' +
    hex(bytes.subarray(8, 10)) +
    '-' +
    hex(bytes.subarray(10, 16))
  );
};

if (typeof globalWithCrypto.crypto?.randomUUID !== 'function') {
  globalWithCrypto.crypto = Object.assign(globalWithCrypto.crypto ?? {}, {
    randomUUID: generateUUIDv4,
  });
}
