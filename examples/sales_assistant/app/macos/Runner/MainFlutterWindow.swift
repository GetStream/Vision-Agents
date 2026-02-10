import Cocoa
import FlutterMacOS

class MainFlutterWindow: NSWindow {
  override func awakeFromNib() {
    let flutterViewController = FlutterViewController()

    // Capture XIB frame, then assign the Flutter content view controller
    // and re-apply the frame (standard Flutter macOS pattern).
    let windowFrame = self.frame
    self.contentViewController = flutterViewController
    self.setFrame(windowFrame, display: true)

    // Now resize to our compact overlay dimensions.
    let screenFrame = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
    let windowWidth: CGFloat = 420
    let windowHeight: CGFloat = 640
    let windowX = screenFrame.maxX - windowWidth - 24
    let windowY = screenFrame.maxY - windowHeight - 24
    let overlayFrame = NSRect(x: windowX, y: windowY, width: windowWidth, height: windowHeight)
    self.setFrame(overlayFrame, display: true)

    // Translucent chrome
    self.styleMask.insert(.fullSizeContentView)
    self.titlebarAppearsTransparent = true
    self.titleVisibility = .hidden
    self.isOpaque = false
    self.backgroundColor = NSColor(calibratedWhite: 0.08, alpha: 0.92)
    self.hasShadow = true

    // Always-on-top floating overlay, draggable by background
    self.level = .floating
    self.isMovableByWindowBackground = true

    // Rounded corners
    if let contentView = self.contentView {
      contentView.wantsLayer = true
      contentView.layer?.cornerRadius = 16
      contentView.layer?.masksToBounds = true
    }

    RegisterGeneratedPlugins(registry: flutterViewController)
    super.awakeFromNib()
  }
}
