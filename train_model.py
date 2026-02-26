
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)
data.head
# Save model
joblib.dump(model, "model.joblib")

print("Model trained and saved!")
