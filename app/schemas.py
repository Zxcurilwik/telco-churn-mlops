"""Request/response contracts. Pydantic rejects bad input with HTTP 422
before it ever reaches the model."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_BATCH_SIZE = 1000

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]
INTERNET_ADDONS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

EXAMPLE_CUSTOMER = {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


class Customer(BaseModel):
    """One customer, in the same format as the IBM Telco dataset."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    customerID: str | None = Field(None, max_length=64, description="Echoed back, not used")
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=1000, description="Current monthly bill")
    TotalCharges: float = Field(ge=0, le=200_000, description="Total billed so far")

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        """Enforce the same invariants that hold in the training data."""
        if (self.PhoneService == "No") != (self.MultipleLines == "No phone service"):
            raise ValueError(
                "MultipleLines must be 'No phone service' exactly when PhoneService is 'No'"
            )
        no_internet = self.InternetService == "No"
        for field in INTERNET_ADDONS:
            if no_internet != (getattr(self, field) == "No internet service"):
                raise ValueError(
                    f"{field} must be 'No internet service' exactly when InternetService is 'No'"
                )
        return self


class Prediction(BaseModel):
    customerID: str | None = None
    churn_probability: float = Field(ge=0, le=1)
    churn: bool = Field(description="churn_probability >= decision threshold")
    risk_level: Literal["low", "medium", "high"]


class PredictionResponse(Prediction):
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    threshold: float
    model_version: str
    predictions: list[Prediction]


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str
