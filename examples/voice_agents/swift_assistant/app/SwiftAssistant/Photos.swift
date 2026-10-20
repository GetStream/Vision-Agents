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

    private static func source(for file: URL) -> ImageSource? {
        guard let image = UIImage(contentsOfFile: file.path),
            let jpeg = resized(image, longestSide: longestSide).jpegData(compressionQuality: 0.7)
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
