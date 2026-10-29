from enum import StrEnum


class AuditResourceType(StrEnum):
    AGENT_CONFIG = "agent_config"
    END_USER = "end_user"
    KNOWLEDGE = "knowledge"
    KNOWLEDGE_URL = "knowledge_url"
    LIMIT = "limit"
    PLUGIN = "plugin"
    POLICY = "policy"
    PRICE_LIST = "price_list"
    PROVIDER_KEY = "provider_key"
    ROUTER_CONFIG = "router_config"
    SKILL = "skill"

    def __str__(self) -> str:
        return str(self.value)
