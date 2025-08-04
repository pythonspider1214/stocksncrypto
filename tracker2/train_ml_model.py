
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def train_model(data_path):
    try:
        data = pd.read_csv(data_path)
        X = data.drop('target', axis=1)
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        print(f"Model trained with score: {model.score(X_test, y_test)}")
    except FileNotFoundError:
        print(f"Data file {data_path} not found.")
    except pd.errors.EmptyDataError:
        print(f"No data found in the file {data_path}.")
    except Exception as e:
        print(f"An unexpected error occurred during model training: {e}")
