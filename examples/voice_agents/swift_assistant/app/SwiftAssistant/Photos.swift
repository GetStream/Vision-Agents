import Photos
import UIKit
import VisionAgentsCore

/// Photos from the composer, as the router takes them.
///
/// The composer hands over files on the phone; a question takes an image by URL, and a phone
/// has nowhere public to put one. So each is shrunk and sent inline, as a data URI: a photo
/// straight off the camera is several megabytes, which is more than a model needs to read it.
enum Photos {
    /// The longest side a photo is sent at, in pixels.
    private static let longestSide: CGFloat = 1280

    /// The longest side a photo is shown at in the chat before Stream has it, in pixels.
    private static let shownSide: CGFloat = 540

    /// Each file that reads as an image, ready to ask with. One that does not is left out.
    static func sources(for files: [URL]) async -> [ImageSource] {
        files.compactMap(source(for:))
    }

    /// Each file that reads as an image, small enough to show in the chat at once.
    static func thumbnails(for files: [URL]) -> [UIImage] {
        files.compactMap { UIImage(contentsOfFile: $0.path).map { resized($0, longestSide: shownSide) } }
    }

    /// The newest picture in the photo library, ready to show the agent, and when it was taken.
    /// Nil when the library has none.
    static func latest() async throws -> (image: ImageSource, taken: Date?)? {
        let access = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
        guard access == .authorized || access == .limited else {
            throw ToolFailure("They have not let Juno see their photos. It is in Settings.")
        }
        let newest = PHFetchOptions()
        newest.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        newest.fetchLimit = 1
        guard let asset = PHAsset.fetchAssets(with: .image, options: newest).firstObject else {
            return nil
        }
        let options = PHImageRequestOptions()
        // One answer, at full quality, even when the original is in iCloud.
        options.deliveryMode = .highQualityFormat
        options.isNetworkAccessAllowed = true
        let image = await withCheckedContinuation { (done: CheckedContinuation<UIImage?, Never>) in
            PHImageManager.default().requestImage(
                for: asset, targetSize: CGSize(width: longestSide, height: longestSide),
                contentMode: .aspectFit, options: options
            ) { image, _ in done.resume(returning: image) }
        }
        guard let image, let source = source(for: image) else {
            throw ToolFailure("Their newest photo could not be read.")
        }
        return (image: source, taken: asset.creationDate)
    }

    private static func source(for file: URL) -> ImageSource? {
        UIImage(contentsOfFile: file.path).flatMap(source(for:))
    }

    private static func source(for image: UIImage) -> ImageSource? {
        guard let jpeg = resized(image, longestSide: longestSide).jpegData(compressionQuality: 0.7)
        else { return nil }
        return ImageSource(url: "data:image/jpeg;base64," + jpeg.base64EncodedString())
    }

    private static func resized(_ image: UIImage, longestSide: CGFloat) -> UIImage {
        let scale = min(1, longestSide / max(image.size.width, image.size.height))
        let size = CGSize(width: image.size.width * scale, height: image.size.height * scale)
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        return UIGraphicsImageRenderer(size: size, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: size))
        }
    }
}
