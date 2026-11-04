import type { ImageSource } from "@stream-io/calypso";

/**
 * Photos from the composer, as the router takes them.
 *
 * The composer hands over files; a question takes an image by URL, and a page has nowhere
 * public to put one. So each is shrunk and sent inline, as a data URI: a photo straight off a
 * phone's camera is several megabytes, which is more than a model needs to read it.
 */

/** The longest side a photo is sent at, in pixels. */
export const LONGEST_SIDE = 1280;

/** The size an image of `width` by `height` is drawn at, so that its longest side fits `longest`. */
export function fitWithin(width: number, height: number, longest: number): { width: number; height: number } {
  const scale = Math.min(1, longest / Math.max(width, height));
  return { width: Math.round(width * scale), height: Math.round(height * scale) };
}

/** Each file that reads as an image, ready to ask with. One that does not is left out. */
export async function photoSources(files: readonly File[]): Promise<ImageSource[]> {
  const sources = await Promise.all(files.map((file) => source(file).catch(() => undefined)));
  return sources.filter((each): each is ImageSource => each !== undefined);
}

async function source(file: File): Promise<ImageSource | undefined> {
  if (!file.type.startsWith("image/")) {
    return undefined;
  }
  // Turned the way the camera held it, as the phones do.
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  const size = fitWithin(bitmap.width, bitmap.height, LONGEST_SIDE);
  const canvas = document.createElement("canvas");
  canvas.width = size.width;
  canvas.height = size.height;
  canvas.getContext("2d")?.drawImage(bitmap, 0, 0, size.width, size.height);
  bitmap.close();
  return { url: canvas.toDataURL("image/jpeg", 0.7) };
}
