from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "AgeEstAI"
    MAX_FACES: int = 5
    AGE_BIN_LABELS: tuple[str, ...] = (
        "0-12",
        "13-19",
        "20-29",
        "30-39",
        "40-49",
        "50-64",
        "65+",
    )

    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
