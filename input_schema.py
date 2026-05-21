
class DiabetesInput(BaseModel):

    Pregnancies: int = Field(ge=0, le=20)

    Glucose: float = Field(
        gt=0,
        le=500,
        description="Blood glucose level"
    )

    BloodPressure: float = Field(
        gt=0,
        le=300
    )

    SkinThickness: float = Field(
        ge=0,
        le=100
    )

    Insulin: float = Field(
        ge=0,
        le=1000
    )

    BMI: float = Field(
        gt=0,
        le=100
    )

    DiabetesPedigreeFunction: float = Field(
        ge=0,
        le=5
    )

    Age: int = Field(
        ge=1,
        le=120
    )
