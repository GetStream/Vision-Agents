package io.getstream.visionagents.androidassistant

import android.content.Context
import android.content.pm.PackageManager
import androidx.activity.ComponentActivity
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Asks the person for a permission when a tool first needs it.
 *
 * The tools run on the session's socket, which outlives any one screen, but only an activity can
 * ask. So the activity on screen lends itself here while it is there, and a tool that needs the
 * calendar or the location asks through it and waits for the answer.
 */
object Permissions {
    private var launcher: ActivityResultLauncher<Array<String>>? = null
    private var answer: CompletableDeferred<Map<String, Boolean>>? = null
    private val asking = Mutex()

    /** Lends [activity] for asking. Call it from `onCreate`, before the activity starts. */
    fun attach(activity: ComponentActivity) {
        val registered = activity.registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
            answer?.complete(it)
        }
        launcher = registered
        activity.lifecycle.addObserver(
            object : DefaultLifecycleObserver {
                override fun onDestroy(owner: LifecycleOwner) {
                    if (launcher === registered) launcher = null
                    // Nobody is left to answer, which counts as no.
                    answer?.complete(emptyMap())
                }
            },
        )
    }

    /**
     * Whether the app holds at least one of [permissions], asking for all of them first when it
     * holds none: the location is either precise or approximate. False when the person refuses,
     * or when no screen is there to ask on.
     */
    suspend fun anyOf(context: Context, vararg permissions: String): Boolean =
        permissions.any { context.holds(it) } || ask(permissions.toList()).values.any { it }

    /** Whether the app holds every one of [permissions], asking for those it does not. */
    suspend fun allOf(context: Context, vararg permissions: String): Boolean {
        val missing = permissions.filterNot { context.holds(it) }
        return missing.isEmpty() || ask(missing).let { granted -> missing.all { granted[it] == true } }
    }

    /** Asks once at a time, since a screen shows one question. */
    private suspend fun ask(permissions: List<String>): Map<String, Boolean> = asking.withLock {
        withContext(Dispatchers.Main) {
            val launch = launcher ?: return@withContext emptyMap()
            val pending = CompletableDeferred<Map<String, Boolean>>()
            answer = pending
            launch.launch(permissions.toTypedArray())
            pending.await()
        }
    }

    private fun Context.holds(permission: String): Boolean =
        checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED
}
