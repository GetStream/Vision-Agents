package io.getstream.visionagents.androidassistant

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.browser.auth.AuthTabIntent
import androidx.browser.customtabs.CustomTabsClient
import androidx.browser.customtabs.CustomTabsIntent
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.lifecycle.DefaultLifecycleObserver
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.CompletableDeferred

/**
 * The browser half of a connector's sign-in: opens the router's sign-in page and waits for the
 * router to hand the browser back to this app, at [Assistant.RETURN_URL].
 *
 * The page is opened in an Auth Tab where the browser has one, which hands the redirect to the
 * app itself and closes. Where it does not, it is a Custom Tab, and the redirect comes back
 * through [SignInCallbackActivity], which the app's scheme opens. Either way the page is the
 * browser's, so the provider's credentials never pass through the app.
 *
 * Make one in the activity's `onCreate`, before it starts, since that is when it can listen for
 * the Auth Tab's answer.
 */
class SignIn(private val activity: ComponentActivity) {
    private val authTab = AuthTabIntent.registerActivityResultLauncher(activity) { result ->
        pending?.complete(result.resultUri.takeIf { result.resultCode == AuthTabIntent.RESULT_OK })
    }

    /**
     * Opens [url], the sign-in's `launch_url` with `#handoff_token=` and its handoff token after
     * it, and returns where the router sent the browser back to, with the `connection_id` and
     * `status` it added, or null when the person closed the page first. A sign-in started while
     * another is open ends that one with null.
     */
    suspend fun authenticate(url: Uri): Uri? {
        pending?.complete(null)
        val answer = CompletableDeferred<Uri?>()
        pending = answer
        val browser = CustomTabsClient.getPackageName(activity, null)
        if (browser != null && CustomTabsClient.isAuthTabSupported(activity, browser)) {
            AuthTabIntent.Builder().build()
                .apply { intent.setPackage(browser) }
                .launch(authTab, url, Assistant.CALLBACK_SCHEME)
        } else {
            endsOnReturn(answer)
            CustomTabsIntent.Builder().build().launchUrl(activity, url)
        }
        return try {
            answer.await()
        } finally {
            if (pending === answer) pending = null
        }
    }

    /**
     * Ends [answer] with null once the app is back without a redirect, the Custom Tab having been
     * closed. A redirect arrives first: [SignInCallbackActivity] hands it over before it brings
     * this activity back.
     */
    private fun endsOnReturn(answer: CompletableDeferred<Uri?>) {
        val lifecycle = activity.lifecycle
        lifecycle.addObserver(
            object : DefaultLifecycleObserver {
                private var left = false

                override fun onPause(owner: LifecycleOwner) {
                    left = true
                }

                override fun onResume(owner: LifecycleOwner) {
                    if (!left) return
                    answer.complete(null)
                    lifecycle.removeObserver(this)
                }
            },
        )
    }

    internal companion object {
        /**
         * The sign-in waiting for its redirect. One for the process, since the redirect may
         * arrive in an activity made after the one that opened the page.
         */
        @Volatile
        private var pending: CompletableDeferred<Uri?>? = null

        fun redirected(uri: Uri?) {
            pending?.complete(uri)
        }
    }
}

/** The app's [SignIn], for a sign-in card's Connect button. */
val LocalSignIn = staticCompositionLocalOf<SignIn> { error("SignIn is provided by MainActivity") }

/**
 * Where the router sends a Custom Tab back to, by the app's own scheme: it hands the redirect to
 * the sign-in waiting for it and brings the app back over the tab, which closes it.
 */
class SignInCallbackActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        SignIn.redirected(intent.data)
        startActivity(
            Intent(this, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP),
        )
        finish()
    }
}
