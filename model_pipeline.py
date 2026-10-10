import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_FILE = "onlinefoods.csv"
TARGET = "Output"


def load_data(path=DATA_FILE):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


def prepare_features(df):
    y = df[TARGET].str.strip().str.capitalize().map({"Yes": 1, "No": 0})
    X = pd.get_dummies(df.drop(columns=[TARGET])).fillna(0)
    return X, y


def train_model(df):
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return model, accuracy


if __name__ == "__main__":
    data = load_data()
    _, acc = train_model(data)
    print("Test accuracy:", round(acc, 4))
