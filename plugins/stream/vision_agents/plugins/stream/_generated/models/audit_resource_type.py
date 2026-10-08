from enum import StrEnum


class AuditResourceType(StrEnum):
    AGENT_CONFIG = "agent_config"
    KNOWLEDGE = "knowledge"
    KNOWLEDGE_URL = "knowledge_url"
    PLUGIN = "plugin"
    POLICY = "policy"
    ROUTER_CONFIG = "router_config"
    SKILL = "skill"

    def __str__(self) -> str:
        return str(self.value)
