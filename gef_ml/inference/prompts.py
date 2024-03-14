from enum import Enum
from typing import Optional, Literal

from pydantic import BaseModel, Field

from .descriptions import (
    DESC_CAPACITY_DEV,
    DESC_FINANCE,
    DESC_INDUSTRY_LEADERSHIP,
    DESC_KNOWLEDGE_INFO_SHARING,
    DESC_NO_PRIVATE_SECTOR,
    DESC_POLICY_DEV,
)

QUERY_PRIVATE_SECTOR_INVOLVEMENT = f"""Determine the level of private sector involvement in the provided context. Private sector involvement is defined by the following levels:

Below are the in depth descriptions of each level along with discintions between them:

{DESC_NO_PRIVATE_SECTOR}

{DESC_KNOWLEDGE_INFO_SHARING}

{DESC_POLICY_DEV}

{DESC_CAPACITY_DEV}

{DESC_FINANCE}

{DESC_INDUSTRY_LEADERSHIP}

Please select the primary level of private sector involvement in this document exerpt. If there is no clear primary level, also select the secondary level. If there is no private sector involvement, select "No private sector involvement" and provide a reason for the choice. 

Respond in JSON with the keys "involvement_level", "secondary_involvement_level" (if applicable), "reason", and "extra_info" (if applicable).
"""


class InvolvementLevelEnum(str, Enum):
    NO_INVOLVEMENT = "No private sector involvement"
    KNOWLEDGE_SHARING = "Knowledge & Information Sharing"
    POLICY_DEVELOPMENT = "Policy Development"
    CAPACITY_DEVELOPMENT = "Capacity Development"
    FINANCE = "Finance"
    INDUSTRY_LEADERSHIP = "Industry Leadership"


class ResponseObject(BaseModel):
    """Data model for the response to a private sector involvement query."""

    involvement_level: Literal[
        "No private sector involvement",
        "Knowledge & Information Sharing",
        "Policy Development",
        "Capacity Development",
        "Finance",
        "Industry Leadership",
    ] = Field(..., description="The primary level of private sector involvement")
    secondary_involvement_level: Optional[
        Literal[
            "No private sector involvement",
            "Knowledge & Information Sharing",
            "Policy Development",
            "Capacity Development",
            "Finance",
            "Industry Leadership",
        ]
    ] = Field(
        ...,
        description="The secondary level of private sector involvement if there is not a clear primary level",
    )
    reason: str = Field(..., description="The reason for the chosen involvement level")
    extra_info: Optional[str] = Field(
        ...,
        description="Any additional relevant information about the private sector involvement",
    )


class ResponseObject2(BaseModel):
    primary_involvement_level: InvolvementLevelEnum = Field(
        ..., description="The primary level of private sector involvement"
    )
