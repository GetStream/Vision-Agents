import asyncio
from typing import Any, Optional

import boto3
from smithy_aws_core.identity.components import (
    AWSCredentialsIdentity,
    AWSIdentityProperties,
)
from smithy_core.aio.interfaces.identity import IdentityResolver


class Boto3CredentialsResolver(
    IdentityResolver[AWSCredentialsIdentity, AWSIdentityProperties]
):
    """IdentityResolver that delegates to boto3.Session for credential resolution.

    Supports the full boto3 credential chain: env vars, shared credentials files,
    AWS profiles, SSO, EC2 instance profiles, etc.
    """

    def __init__(self, profile_name: Optional[str] = None) -> None:
        self._session = boto3.Session(profile_name=profile_name)

    async def get_identity(
        self, *, properties: AWSIdentityProperties, **kwargs: Any
    ) -> AWSCredentialsIdentity:
        # Both calls can block: get_credentials() walks the provider chain
        # (file I/O, IMDS, SSO, STS) on first access, and get_frozen_credentials()
        # triggers refresh on RefreshableCredentials.
        credentials = await asyncio.to_thread(self._session.get_credentials)
        if not credentials:
            raise ValueError("Unable to load AWS credentials via boto3")

        creds = await asyncio.to_thread(credentials.get_frozen_credentials)
        if not creds.access_key or not creds.secret_key:
            raise ValueError("AWS credentials are incomplete")

        return AWSCredentialsIdentity(
            access_key_id=creds.access_key,
            secret_access_key=creds.secret_key,
            session_token=creds.token or None,
            expiration=None,
        )
