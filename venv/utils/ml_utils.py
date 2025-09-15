# utils/ml_utils.py
import numpy as np
from sklearn.linear_model import LinearRegression

def compute_predictions(profile: dict) -> dict:
    """
    Computes agricultural predictions based on the user's profile.
    NOTE: This is a DUMMY model for demonstration purposes.
    """
    print(f"[ML] Computing predictions for profile: {profile}")
    try:
        land_size = profile.get('land_size', 2.0)
        ph = profile.get('ph', 6.5)

        model = LinearRegression()
        X_train = np.array([[1.0, 5.5], [3.0, 7.0], [2.0, 6.0]])
        y_train = np.array([300, 650, 450])
        model.fit(X_train, y_train)

        predicted_yield = model.predict(np.array([[land_size, ph]]))[0]

        predictions = {
            "yield": round(predicted_yield, 2),
            "pest_risk": round(np.random.uniform(0.1, 0.4), 2),
            "soil_fertility": "Moderate"
        }
        print(f"[ML] Predictions calculated: {predictions}")
        return predictions
    except Exception as e:
        print(f"❌ ERROR [ML]: Failed to compute predictions. Reason: {e}")
        return {"error": "Could not compute predictions."}