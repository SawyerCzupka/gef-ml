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

QUERY_PRIVATE_SECTOR_SUMMARY = f"""Provide a summary of the private sector involvement in the provided context. The summary should include all relevant information that can be used to identify the private sector involvement level so that the decision can be made using only the summary itself.

Private sector involvement is defined by the following levels and their in depth descriptions:

{DESC_NO_PRIVATE_SECTOR}

{DESC_KNOWLEDGE_INFO_SHARING}

{DESC_POLICY_DEV}

{DESC_CAPACITY_DEV}

{DESC_FINANCE}

{DESC_INDUSTRY_LEADERSHIP}

Please provide a summary of the private sector involvement in this document exerpt.
"""

QUERY_PRIVATE_SECTOR_SUMMARY_V2 = """## Categories of Private Sector Involvement

**1. No Private Sector Involvement:**
   - Projects without any engagement with the private sector.
   - Mentioning the private sector in stakeholder identification without direct engagement.

**2. Knowledge & Information Sharing:**
   - Projects where the private sector participates only in awareness-raising or knowledge/information-sharing.
   - Projects offering information to enhance the private sector's profit-making capacity should be classified as Category 4.

**3. Policy Development:**
   - Projects consulting the private sector for policy development (e.g., regulations, frameworks, reports).
   - Projects involving private sector in governance but not considering their views fall under Category 2.

**4. Capacity Development:**
   - Projects building the private sector's capacity for profit-generating activities.
   - Training in auxiliary topics or informing investment opportunities without capacity building should be Category 2.

**5. Finance:**
   - Projects involving the private sector for finance/expertise through Public-Private Partnerships.
   - Routine expenses covered by the private sector are not included. Leading roles by the private sector should be Category 6.

**6. Industry Leadership:**
   - Projects where the private sector leads in proposing solutions and advancing the project.
   - The private sector contributes financing/expertise and spearheads development/implementation of project activities.

---

**Query:**

Provide a summary of the private sector involvement in the provided document excerpt based on the categories above.
"""


class InvolvementLevelEnum(str, Enum):
    NO_INVOLVEMENT = "No private sector involvement"
    KNOWLEDGE_SHARING = "Knowledge & Information Sharing"
    POLICY_DEVELOPMENT = "Policy Development"
    CAPACITY_DEVELOPMENT = "Capacity Development"
    FINANCE = "Finance"
    INDUSTRY_LEADERSHIP = "Industry Leadership"


class PrivSectorClassResponseObj(BaseModel):
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


class PrivSectorSummaryResponseObj(BaseModel):
    """Data model for a summary of the private sector involvement of a project."""

    summary: str = Field(
        ..., description="The summary of the private sector involvement"
    )


class ResponseObject2(BaseModel):
    primary_involvement_level: InvolvementLevelEnum = Field(
        ..., description="The primary level of private sector involvement"
    )
