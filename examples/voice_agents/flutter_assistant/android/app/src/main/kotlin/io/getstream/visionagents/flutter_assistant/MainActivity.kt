package io.getstream.visionagents.flutter_assistant

import android.content.pm.PackageManager
import android.os.Build
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    /** The app's question for the local network, answered once the person has. */
    private var asking: MethodChannel.Result? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val messenger = flutterEngine.dartExecutor.binaryMessenger
        val calendar = CalendarChannel(applicationContext)
        MethodChannel(messenger, CalendarChannel.NAME).setMethodCallHandler(calendar::handle)
        MethodChannel(messenger, LOCAL_NETWORK).setMethodCallHandler { _, result -> askForLocalNetwork(result) }
    }

    /**
     * Asks for the local network, which an app targeting Android 17 needs to reach the Mac at its
     * address on the Wi-Fi, as android_assistant does. Earlier Androids need no asking.
     */
    private fun askForLocalNetwork(result: MethodChannel.Result) {
        // Android 17, which this compileSdk knows only as a number.
        if (Build.VERSION.SDK_INT < 37 || checkSelfPermission(PERMISSION) == PackageManager.PERMISSION_GRANTED) {
            result.success(true)
            return
        }
        asking = result
        requestPermissions(arrayOf(PERMISSION), REQUEST)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != REQUEST) return
        asking?.success(grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED)
        asking = null
    }

    private companion object {
        const val LOCAL_NETWORK = "io.getstream.visionagents.flutterassistant/localNetwork"
        const val PERMISSION = "android.permission.ACCESS_LOCAL_NETWORK"
        const val REQUEST = 37
    }
}
