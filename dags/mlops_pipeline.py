from __future__ import annotations

import os
import json
import pickle
from datetime import datetime

import kagglehub

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import mlflow
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException

BASE_DIR = "/opt/airflow/mlops_demo"
DATA_DIR = f"{BASE_DIR}/data"
ARTIFACTS_DIR = f"{BASE_DIR}/artifacts"
REGISTRY_DIR = f"{BASE_DIR}/registry"
DATA_ID = "andpereira/portuguese-car-market" 
CSV_FILENAME = "market_analysis_cars_nov2025.csv"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(REGISTRY_DIR, exist_ok=True)

# MLflow local tracking folder
MLFLOW_URI = f"file://{BASE_DIR}/mlruns"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("airflow-mlops-pro1d-assignment-regression")

R2_THRESHOLD = 0.80

default_args = {"owner": "mlops", "depends_on_past": False, "retries": 0}

with DAG(
    dag_id="mlops_pro1d_assignment_regression",
    description="MLOps PRO1d assignment saving artifacts to Windows",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args=default_args,
    tags=["mlops", "assignment"],
) as dag:

    @task
    def ingest_data() -> str:
        DATA_DIR_PATH = kagglehub.dataset_download(DATA_ID) 
        full_source_path = os.path.join(DATA_DIR_PATH, CSV_FILENAME) 
        local_path = f"{DATA_DIR}/{CSV_FILENAME}"
        df = pd.read_csv(full_source_path)
        df.to_csv(local_path, index=False)
        return local_path

    @task
    def validate_data(csv_path: str) -> str:
        df = pd.read_csv(csv_path)

        report = {
            "rows": len(df),
            "cols": list(df.columns),
            "null_counts": df.isnull().sum().to_dict(),
        }

        report_path = f"{ARTIFACTS_DIR}/validation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        output_path = f"{ARTIFACTS_DIR}/validated_raw.csv"
        df.to_csv(output_path, index=False)
        return output_path

    @task
    def preprocess_data(csv_path: str) -> str:
        df = pd.read_csv(csv_path)

        brands = [
            "Alfa Romeo", "Audi", "BMW", "Chevrolet", "Citroen", "Dacia", "Fiat",
            "Ford", "Honda", "Hyundai", "Jaguar", "Jeep", "Kia", "Land Rover",
            "Lexus", "Mazda", "Mercedes", "Mini", "Mitsubishi", "Nissan",
            "Opel", "Peugeot", "Porsche", "Renault", "Seat", "Skoda",
            "Smart", "Suzuki", "Tesla", "Toyota", "Volkswagen", "Volvo"
        ]

        models = [
            "147", "159",
            "A1", "A3", "A4", "A5", "A6", "A7", "A8", "Q2", "Q3", "Q5", "Q7",
            "1 Series", "116", "118", "120", "3 Series", "316", "318", "320",
            "C1", "C2", "C3", "C4", "C5", "DS3",
            "Duster", "Sandero", "Logan",
            "Punto", "Tipo", "500", "500L", "500X",
            "Focus", "Fiesta", "Mondeo", "Kuga",
            "Civic", "Accord", "CR-V", "Jazz",
            "i20", "i30", "Tucson",
            "Renegade", "Compass",
            "Ceed", "Sportage", "Rio",
            "Ypsilon",
            "Discovery", "Freelander",
            "CT200h", "IS200", "IS250",
            "C200", "C180", "E200", "E220", "A180", "A200", "B180",
            "Cooper", "Countryman",
            "ASX", "Lancer", "Outlander",
            "Qashqai", "Micra", "Juke",
            "Corsa", "Astra", "Insignia", "Mokka",
            "208", "2008", "3008", "308", "508",
            "Cayenne",
            "Clio", "Megane", "Twingo", "Kadjar",
            "Ibiza", "Leon",
            "Fabia", "Octavia", "Superb",
            "ForTwo", "ForFour",
            "Tivoli",
            "Impreza", "Forester",
            "Swift", "Vitara", "SX4",
            "Auris", "Yaris", "Corolla", "Avensis", "RAV4",
            "Golf", "Polo", "Passat", "Tiguan", "Scirocco",
            "S60", "V40", "V60", "XC60"
        ]

        #Ekstrakcja marek aut
        def extract_brand(title):
            for b in brands:
                if title.startswith(b):
                    return b
            return "Unknown"

        # Ekstrakcja modeli aut
        def extract_model(title):
            for m in models:
                if m in title:
                    return m
            return "Unknown"

        #Utworzenie nowych kolumn brand, model i uzupełnienie ich danymi
        df["brand"] = df["title"].apply(extract_brand)
        df["model"] = df["title"].apply(extract_model)

        #Usunięcie kolumny currency bo zawiera tylko jedną wartość i się nie przyda do treningu modeli
        df = df.drop(columns=['currency', 'index', 'title'])

        df = df.rename(columns={'displacement': 'displacement_cc'})

        #Uzupełnienie brakujących wartości liczbowych medianą dla każdej kolumny liczbowej
        numeric_columns = df.select_dtypes(include = ["int64", "float64"]).columns
        for column in numeric_columns:
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)

        #Zmiana danych w 2 kolumnach z tekstowych na numeryczne
        mapping_fuel = {'Diesel': 1, 'Gasoline': 2, 'Hybrid Plug-In': 3, 'Hybrid (Gasoline)': 4, 'Hybrid (Diesel)': 5, 'LPG': 6, 'GNC': 7}
        df['fuel'] = (
            df['fuel']
            .replace(mapping_fuel)
            .fillna(0)
            .astype(int)
        )

        mapping_transmission = {'Manual': 1, 'Automatic': 2}
        df['transmission'] = (
            df['transmission']
            .replace(mapping_transmission)
            .fillna(0)
            .astype(int)
        )

        columns_to_numerical = ['brand', 'model', 'location']
        df = pd.get_dummies(df, columns=columns_to_numerical, drop_first=True)

        #Zapis nowego datasetu
        new_csv_path = f"{ARTIFACTS_DIR}/preprocessed.csv"
        df.to_csv(new_csv_path, index=False)
        return new_csv_path

    @task
    def split_data(csv_path: str) -> dict:
        df = pd.read_csv(csv_path)

        X = df.drop(columns=["price"])
        y = df["price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        paths = {
            "X_train": f"{ARTIFACTS_DIR}/X_train.parquet",
            "X_test": f"{ARTIFACTS_DIR}/X_test.parquet",
            "y_train": f"{ARTIFACTS_DIR}/y_train.parquet",
            "y_test": f"{ARTIFACTS_DIR}/y_test.parquet",
        }

        X_train.to_parquet(paths["X_train"])
        X_test.to_parquet(paths["X_test"])
        pd.DataFrame({"price": y_train}).to_parquet(paths["y_train"])
        pd.DataFrame({"price": y_test}).to_parquet(paths["y_test"])

        return paths

    @task
    def train_model(paths: dict) -> str:
        X_train = pd.read_parquet(paths["X_train"])
        y_train = pd.read_parquet(paths["y_train"])["price"]

        with mlflow.start_run(run_name="rf-regressor-train") as run:
            params = {"n_estimators": 100, "random_state": 42}
            mlflow.log_params(params)

            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)

            model_path = f"{ARTIFACTS_DIR}/model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            mlflow.log_artifact(model_path, artifact_path="model")

            meta_path = f"{ARTIFACTS_DIR}/mlflow_run.json"
            with open(meta_path, "w") as f:
                json.dump({"run_id": run.info.run_id}, f)

        return model_path

    @task
    def evaluate_model(paths: dict, model_path: str) -> dict:
        X_test = pd.read_parquet(paths["X_test"])
        y_test = pd.read_parquet(paths["y_test"])["price"]

        model = pickle.load(open(model_path, "rb"))

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        meta_path = f"{ARTIFACTS_DIR}/mlflow_run.json"
        run_id = json.load(open(meta_path))["run_id"]

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("r2_score", float(r2))

        metrics = {"r2_score": float(r2), "threshold": R2_THRESHOLD}
        json.dump(metrics, open(f"{ARTIFACTS_DIR}/metrics.json", "w"), indent=2)

        return metrics

    @task
    def promote_if_good(model_path: str, metrics: dict) -> str:
        r2 = metrics["r2_score"]

        if r2 < R2_THRESHOLD:
            raise AirflowSkipException(f"R2 score {r2} < threshold {R2_THRESHOLD}")

        version = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        dest = f"{REGISTRY_DIR}/model_{version}_r2{r2:.3f}.pkl"

        with open(model_path, "rb") as src, open(dest, "wb") as dst:
            dst.write(src.read())

        return dest

    csv_path = ingest_data()
    validated_csv_path = validate_data(csv_path)
    preprocessed_csv_path = preprocess_data(validated_csv_path)
    splits = split_data(preprocessed_csv_path)
    model_path = train_model(splits)
    metrics = evaluate_model(splits, model_path)
    _ = promote_if_good(model_path, metrics)