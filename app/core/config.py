from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # PostgreSQL
    database_url: str = "postgresql://localhost:5432/retro_rag"

    # MySQL (moalog-server, read-only)
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "readonly"
    mysql_password: SecretStr = SecretStr("")
    mysql_database: str = "moalog"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # OpenAI
    openai_api_key: SecretStr = SecretStr("")

    # JWT
    jwt_secret: SecretStr = SecretStr("")

    # CORS
    cors_allowed_origins: str = "http://localhost:3000"

    # Logging
    log_level: str = "INFO"

    # Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # LLM
    llm_model: str = "gpt-4o"
    llm_fallback_model: str = "gpt-4o-mini"

    # Hybrid Search
    hybrid_search_alpha: float = 0.7

    # Relevance
    relevance_threshold: float = 0.4

    # Embedding Cache
    embedding_cache_ttl: int = 86400  # 24 hours in seconds

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",")]


settings = Settings()
