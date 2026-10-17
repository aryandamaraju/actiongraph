from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Item(BaseModel):
    task: str = Field(min_length=3)
    owners: List[str] = Field(default_factory=list)
    due: Optional[str] = None
    evidence: str = Field(min_length=3)

    @field_validator("owners", mode="before")
    @classmethod
    def _owners(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [] if v.strip().lower() in ("", "none", "null", "unassigned") else [v.strip()]
        return [str(x).strip() for x in v if str(x).strip()]

    @field_validator("due", mode="before")
    @classmethod
    def _due(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return None if v.lower() in ("", "none", "null", "n/a", "unspecified") else v


class Extraction(BaseModel):
    items: List[Item] = Field(default_factory=list)
