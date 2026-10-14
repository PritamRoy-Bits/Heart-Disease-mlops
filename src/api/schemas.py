"""Pydantic request/response schemas for the prediction API."""
from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    age: float = Field(..., examples=[63])
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=3, description="chest pain type (0-3)")
    trestbps: float = Field(..., description="resting blood pressure (mm Hg)")
    chol: float = Field(..., description="serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="resting ECG result (0-2)")
    thalach: float = Field(..., description="maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="exercise induced angina")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="slope of the peak exercise ST segment")
    ca: float = Field(..., ge=0, le=4, description="number of major vessels colored by fluoroscopy")
    thal: float = Field(..., description="thalassemia (3 = normal, 6 = fixed defect, 7 = reversible)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1,
                "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0,
                "ca": 0, "thal": 6,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
