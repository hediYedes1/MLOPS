# model_pipeline.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib


def prepare_data():

    data_filename = "Churn_Modelling.csv"

    df = pd.read_csv(data_filename)

    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    columns_to_drop = ["Surname", "Geography"]
    df = df.drop(columns_to_drop, axis=1)

    X = df.drop(["Exited"], axis=1)
    y = df["Exited"]
    X = X.drop(columns=["RowNumber", "CustomerId"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_model(X_train, y_train):

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy score: {accuracy * 100:.2f}%")

    matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot()
    plt.show()

    return accuracy


def save_model(model, filename="classifier.joblib"):

    joblib.dump(model, filename)
    print(f" Modèle sauvegardé avec succes dans {filename}")


def load_model(filename="classifier.joblib"):

    model = joblib.load(filename)
    print(f" Modèle chargé depuis {filename}")
    return model
