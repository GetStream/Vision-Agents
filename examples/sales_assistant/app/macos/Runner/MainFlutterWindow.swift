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
    self.backgroundColor = .clear
    self.hasShadow = true

    // Always-on-top floating overlay, draggable by background
    self.level = .floating
    self.isMovableByWindowBackground = true

    // Hide this window from all screen-capture / screen-sharing APIs
    // (Zoom, OBS, QuickTime, etc.) so coaching suggestions stay private.
    self.sharingType = .none

    // Insert an NSVisualEffectView behind the Flutter content for real
    // translucency / frosted-glass blur.
    if let contentView = self.contentView {
      let blurView = NSVisualEffectView()
      blurView.material = .hudWindow
      blurView.blendingMode = .behindWindow
      blurView.state = .active
      blurView.wantsLayer = true
      blurView.frame = contentView.bounds
      blurView.autoresizingMask = [.width, .height]
      contentView.addSubview(blurView, positioned: .below, relativeTo: nil)

      // Rounded corners on the whole window content
      contentView.wantsLayer = true
      contentView.layer?.cornerRadius = 16
      contentView.layer?.masksToBounds = true
      contentView.layer?.borderWidth = 0.5
      contentView.layer?.borderColor = NSColor(white: 1.0, alpha: 0.08).cgColor
    }

    RegisterGeneratedPlugins(registry: flutterViewController)
    super.awakeFromNib()
  }
}
