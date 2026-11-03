from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "PrepZio Interview System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    API_PREFIX: str = "/api/v1"

    MODEL_NAME: str = "PrepZio-CS-LLM"
    MODEL_CHECKPOINT: str = "checkpoints/best_model.pt"

    DEVICE: str = "auto"

    MAX_CONTEXT_LENGTH: int = 512
    MAX_NEW_TOKENS: int = 256

    TEMPERATURE: float = 0.7
    TOP_K: int = 40
    TOP_P: float = 0.9

    ALLOWED_ORIGINS: str = "http://localhost:5173"

    SUPABASE_URL: str | None = None
    SUPABASE_KEY: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.ALLOWED_ORIGINS.split(",")
            if origin.strip()
        ]


settings = Settings()