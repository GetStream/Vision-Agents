package io.getstream.visionagents.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch

/**
 * Where you type.
 *
 * Sending is a function rather than a session, so the same field works for a conversation, a
 * search box or anything else the host wants it for.
 */
@Composable
public fun Composer(
    send: suspend (String) -> Unit,
    modifier: Modifier = Modifier,
    prompt: String = "Message",
    enabled: Boolean = true,
) {
    var text by rememberSaveable { mutableStateOf("") }
    val scope = rememberCoroutineScope()
    val canSend = enabled && text.isNotBlank()
    val submit = {
        if (canSend) {
            val sending = text
            text = ""
            scope.launch { send(sending) }
        }
    }
    Row(
        modifier = modifier,
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            modifier = Modifier.weight(1f),
            enabled = enabled,
            placeholder = { Text(prompt) },
            maxLines = 5,
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
            keyboardActions = KeyboardActions(onSend = { submit() }),
        )
        Button(onClick = { submit() }, enabled = canSend) {
            Text("Send")
        }
    }
}
