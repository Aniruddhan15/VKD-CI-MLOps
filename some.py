# model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    iris = load_iris()
    model = RandomForestClassifier()
    model.fit(iris.data, iris.target)
    joblib.dump(model, "iris_model.pkl")

if __name__ == "__main__":
    train_model()
