package io.getstream.visionagents.core.generated

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
internal data class SessionConnectorBinding(
    @SerialName("name") val name: String,
    @SerialName("connection_id") val connectionId: String,
)
