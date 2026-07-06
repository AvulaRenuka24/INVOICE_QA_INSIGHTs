from typing import List, Optional

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str = Field(default="")
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None


class Invoice(BaseModel):
    invoice_number: str
    vendor: str
    invoice_date: str
    total_amount: float
    currency: str
    line_items: List[LineItem] = Field(default_factory=list)