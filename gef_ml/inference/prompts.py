from typing import Optional
from pydantic import BaseModel, Field, Literal
from enum import Enum

from .descriptions import (
    DESC_CAPACITY_DEV,
    DESC_FINANCE,
    DESC_INDUSTRY_LEADERSHIP,
    DESC_KNOWLEDGE_INFO_SHARING,
    DESC_NO_PRIVATE_SECTOR,
    DESC_POLICY_DEV,
)

QUERY_PRIVATE_SECTOR_INVOLVEMENT = f"""Determine instances of private sector involvement in this document exerpt. Private sector involvement is defined by the following levels:

- 1: No private sector involvement - Projects which do not involve the private sector
- 2: Knowledge & Information Sharing - Projects in which the private sector is informed of initiatives and results
- 3: Policy Development - Projects where the private sector is consulted as part of an intervention run by someone else
- 4: Capacity Development - Projects focused on building the capacity of private sector actors, especially SMEs
- 5: Finance - Projects where government or civil society engages with private sector for finance and expertise
- 6: Industry Leadership - Projects that focus directly on the Private Sector as leader (private sector coming up with solutions; not only trained/capacitated)

Below are the in depth descriptions of each level along with discintions between them:

{DESC_NO_PRIVATE_SECTOR}

{DESC_KNOWLEDGE_INFO_SHARING}

{DESC_POLICY_DEV}

{DESC_CAPACITY_DEV}

{DESC_FINANCE}

{DESC_INDUSTRY_LEADERSHIP}
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

    involvement_level: InvolvementLevelEnum = Field(
        ..., description="The primary level of private sector involvement"
    )
    secondary_involvement_level: Optional[InvolvementLevelEnum] = Field(
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
