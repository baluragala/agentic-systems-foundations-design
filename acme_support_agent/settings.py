"""
settings.py — configuration, validated at boot.
===============================================

WHY this file exists
--------------------
The teaching packages read `os.getenv(...)` wherever they need something. That
is fine for a notebook and unacceptable in a service, for one specific reason:
**a misconfigured agent fails at 3am on the fourth request, not at startup.**

`os.getenv("MAX_REFUND_AUTO_APPROVE_USD")` returns a *string*, or `None`. If it
is `None` because someone forgot the variable, and your comparison is
`amount > float(value or 0)`, you have just built an agent that auto-approves
every refund. That bug is invisible in code review and expensive in production.

So: one settings object, typed, validated on import, that refuses to start if
the configuration is wrong. This is the cheapest reliability win available and
it is the first thing to add when an agent leaves a notebook.

WHAT IS CONFIGURABLE — AND WHAT DELIBERATELY IS NOT
---------------------------------------------------
Configurable: model, budgets, thresholds, storage paths, feature flags.

**Not configurable: the policy rules themselves.** You will notice there is no
`ALLOW_ENTERPRISE_AUTO_REFUND` setting. Rules that exist to stop the agent doing
something harmful do not get an environment variable, because an environment
variable is a thing someone can flip at 2am under pressure. If a rule is load-
bearing for safety, it belongs in code, in review, behind a test.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Every knob the service has, in one validated object.

    Read once at startup via `get_settings()`. Nothing in the package calls
    `os.getenv` directly — so "what is this service configured to do?" has
    exactly one answer, and it is printable.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="ACME_", extra="ignore", case_sensitive=False
    )

    # -- model ---------------------------------------------------------------
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    model: str = "gpt-4o-mini"
    # Temperature 0 in the ACT step is not a style preference. Creativity in
    # tool selection is variance you will spend your on-call rotation debugging.
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    request_timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)

    # -- loop budgets --------------------------------------------------------
    max_steps: int = Field(default=8, ge=1, le=50)
    max_tool_calls: int = Field(default=12, ge=1, le=100)
    # LangGraph's own backstop. Roughly 2 nodes per step, plus the fixed nodes.
    recursion_limit: int = Field(default=40, ge=5, le=200)

    # -- the approval gate ---------------------------------------------------
    # Refunds at or below this amount may be auto-approved IF policy otherwise
    # allows. Above it, a human is interrupted. Set to 0 to require approval for
    # every refund — which is a perfectly reasonable way to launch.
    auto_approve_refund_under_usd: float = Field(default=200.0, ge=0)
    require_approval_for_enterprise: bool = True

    # -- guardrails ----------------------------------------------------------
    max_input_chars: int = Field(default=4000, ge=100)
    redact_pii_in_logs: bool = True
    # Refuse to return an answer whose figures are not supported by tool output.
    enforce_grounding: bool = True

    # -- persistence ---------------------------------------------------------
    # SQLite is the honest default for a reference implementation: durable
    # across restarts, zero infrastructure. Swap for Postgres in production —
    # the checkpointer interface is the same, which is the point.
    checkpoint_db: str = "var/checkpoints.sqlite"
    audit_log_path: str = "var/audit.jsonl"

    # -- observability -------------------------------------------------------
    langsmith_api_key: Optional[str] = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = "acme-support-agent"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"

    # -- environment ---------------------------------------------------------
    environment: Literal["dev", "staging", "prod"] = "dev"

    @field_validator("auto_approve_refund_under_usd")
    @classmethod
    def _sane_threshold(cls, value: float) -> float:
        # A guard against the fat-finger that matters most here. Nobody
        # intentionally auto-approves ten-thousand-dollar refunds, but an extra
        # zero in a config file is an easy mistake and an expensive one.
        if value > 5000:
            raise ValueError(
                f"auto_approve_refund_under_usd={value} is implausibly high. "
                "If this is deliberate, raise the cap in code with a comment "
                "explaining why — do not do it from an environment variable."
            )
        return value

    # -- derived -------------------------------------------------------------
    @property
    def has_model_key(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def is_prod(self) -> bool:
        return self.environment == "prod"

    def summary(self) -> str:
        """Printable configuration — log this at startup, every time.

        When an incident starts, the first question is always "what was it
        configured to do?". Answering that from a log line beats reconstructing
        it from a deploy pipeline.
        """
        key = "set" if self.has_model_key else "MISSING"
        return (
            f"env={self.environment} model={self.model} key={key} "
            f"budget={self.max_steps}steps/{self.max_tool_calls}calls "
            f"auto_approve<=${self.auto_approve_refund_under_usd:.0f} "
            f"grounding={'on' if self.enforce_grounding else 'OFF'} "
            f"langsmith={'on' if self.langsmith_api_key else 'off'}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The single settings instance. Cached, so validation runs exactly once."""
    return Settings()
