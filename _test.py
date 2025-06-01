import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Fixture to load the iris dataset and split it
@pytest.fixture
def iris_data():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Test model training
def test_model_training(iris_data):
    X_train, _, y_train, _ = iris_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    assert hasattr(model, "predict"), "Model should have a predict method"

# Test prediction shape
def test_model_prediction_shape(iris_data):
    X_train, X_test, y_train, _ = iris_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test), "Number of predictions should match number of test samples"

# Test model saving
def test_model_saving(tmp_path):
    model = RandomForestClassifier(n_estimators=10)
    model.fit([[0, 0, 0, 0]], [0])
    file_path = tmp_path / "model.pkl"
    joblib.dump(model, file_path)
    assert file_path.exists(), "Model file should be saved"

# Test model loading and inference
def test_model_loading_and_inference(tmp_path):
    model = RandomForestClassifier(n_estimators=10)
    model.fit([[0, 0, 0, 0]], [0])
    file_path = tmp_path / "model.pkl"
    joblib.dump(model, file_path)
    
    loaded_model = joblib.load(file_path)
    prediction = loaded_model.predict([[0, 0, 0, 0]])
    assert prediction[0] == 0, "Prediction from loaded model should match expected"
