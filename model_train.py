import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import time

INPUT_FEATURES = [
    "Track_1", "Track_2", "Track_3", "Track_4", "Track_5", 
    "Track_6", "Track_7", "Track_8", "Track_9", "Track_10", 
    "Track_11", "Track_12", "Track_13", "Track_14", "Track_15",
    "Track_16", "Track_17", "Track_18", "Track_19",
    "SpeedX", "SpeedY", "SpeedZ", "Angle", "TrackPosition",
    "RPM", "WheelSpinVelocity_1", "WheelSpinVelocity_2",
    "WheelSpinVelocity_3", "WheelSpinVelocity_4", "DistanceCovered",
    "DistanceFromStart", "CurrentLapTime", "Damage",
    "Opponent_9", "Opponent_10", "Opponent_11", "Opponent_19"
]

OUTPUT_FEATURES = ["Steering", "Acceleration", "Braking"]

def read_csv_files(data_dir):
    print(f"Looking for CSV files in: {data_dir}")
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print("No CSV files found in the data directory. Exiting.")
        return {}
    
    print(f"Found {len(csv_files)} CSV files in the data directory")
    
    dataframes = {}
    for file_path in csv_files:
        try:
            print(f"Reading file: {file_path}")
            df = pd.read_csv(file_path, skipinitialspace=True)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            dataframes[os.path.basename(file_path)] = df
            print(f"Successfully loaded {len(df)} rows from {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    return dataframes

def preprocess_data(dataframes):

    if not dataframes:
        print("No data to process. Exiting.")
        return None, None
    
    combined_df = pd.concat(dataframes.values(), ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    
    combined_df = combined_df[combined_df["CurrentLapTime"] >= 0] 
    print(f"Rows after filtering invalid CurrentLapTime: {len(combined_df)}")
    
    missing_values = combined_df.isnull().sum().sum()
    print(f"Total missing values: {missing_values}")
    
    if missing_values > 0:
        column_means = combined_df.mean(numeric_only=True)
        combined_df = combined_df.fillna(column_means)
        print("Filled missing values with column means")
    
    combined_df["Acceleration"] = (combined_df["Acceleration"] > 0.5).astype(int)  
    combined_df["Braking"] = (combined_df["Braking"] > 0.5).astype(int)  
    combined_df["Steering"] = np.clip(combined_df["Steering"], -1, 1)  
    
    available_input_features = [col for col in INPUT_FEATURES if col in combined_df.columns]
    available_output_features = [col for col in OUTPUT_FEATURES if col in combined_df.columns]
    
    if len(available_input_features) < len(INPUT_FEATURES):
        missing_features = set(INPUT_FEATURES) - set(available_input_features)
        print(f"Warning: {len(missing_features)} input features are missing from the data: {missing_features}")
    
    if len(available_output_features) < len(OUTPUT_FEATURES):
        missing_outputs = set(OUTPUT_FEATURES) - set(available_output_features)
        print(f"Warning: {len(missing_outputs)} output features are missing from the data: {missing_outputs}")
    
    print(f"Using {len(available_input_features)} input features: {available_input_features}")
    print(f"Predicting {len(available_output_features)} output features: {available_output_features}")
    
    X = combined_df[available_input_features].copy()
    y = combined_df[available_output_features].copy()
    
    return X, y

def build_and_train_model(X_train, y_train, X_val, y_val):

    print("\n=== Building and Training Model ===")
    
    steering_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_train_steering_scaled = steering_scaler.fit_transform(y_train[["Steering"]])
    y_val_steering_scaled = steering_scaler.transform(y_val[["Steering"]])
    
    y_train_processed = pd.DataFrame({
        "Steering": y_train_steering_scaled.flatten(),
        "Acceleration": y_train["Acceleration"],
        "Braking": y_train["Braking"]
    })
    
    y_val_processed = pd.DataFrame({
        "Steering": y_val_steering_scaled.flatten(),
        "Acceleration": y_val["Acceleration"],
        "Braking": y_val["Braking"]
    })
    
    model = MLPRegressor(
        hidden_layer_sizes=(64, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        tol=1e-5,
        shuffle=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=True
    )
    
    start_time = time.time()
    
    print("Training neural network model...")
    model.fit(X_train, y_train_processed)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    val_predictions = model.predict(X_val)
    val_predictions_df = pd.DataFrame(val_predictions, columns=OUTPUT_FEATURES, index=y_val.index)
    
    val_steering_mse = mean_squared_error(y_val_processed["Steering"], val_predictions_df["Steering"])
    val_steering_r2 = r2_score(y_val_processed["Steering"], val_predictions_df["Steering"])
    
    print("\nSteering Evaluation (Validation):")
    print(f"MSE: {val_steering_mse:.4f}")
    print(f"R²: {val_steering_r2:.4f}")
    
    val_predictions_df["Acceleration_Binary"] = (val_predictions_df["Acceleration"] > 0.5).astype(int)
    val_predictions_df["Braking_Binary"] = (val_predictions_df["Braking"] > 0.5).astype(int)
    
    acc_accuracy = (val_predictions_df["Acceleration_Binary"].values == y_val["Acceleration"].values).mean()
    brake_accuracy = (val_predictions_df["Braking_Binary"].values == y_val["Braking"].values).mean()
    
    print("\nAcceleration Accuracy (Validation): {:.2%}".format(acc_accuracy))
    print("Braking Accuracy (Validation): {:.2%}".format(brake_accuracy))
    
    return model, steering_scaler

def evaluate_model(model, steering_scaler, X_test, y_test):

    print("\n=== Evaluating Model on Test Data ===")
    
    y_test_steering_scaled = steering_scaler.transform(y_test[["Steering"]])
    y_test_processed = pd.DataFrame({
        "Steering": y_test_steering_scaled.flatten(),
        "Acceleration": y_test["Acceleration"],
        "Braking": y_test["Braking"]
    })
    
    test_predictions = model.predict(X_test)
    test_predictions_df = pd.DataFrame(test_predictions, columns=OUTPUT_FEATURES, index=y_test.index)
    
    test_steering_mse = mean_squared_error(y_test_processed["Steering"], test_predictions_df["Steering"])
    test_steering_r2 = r2_score(y_test_processed["Steering"], test_predictions_df["Steering"])
    
    print("\nSteering Evaluation (Test):")
    print(f"MSE: {test_steering_mse:.4f}")
    print(f"R²: {test_steering_r2:.4f}")
    
    test_predictions_df["Acceleration_Binary"] = (test_predictions_df["Acceleration"] > 0.5).astype(int)
    test_predictions_df["Braking_Binary"] = (test_predictions_df["Braking"] > 0.5).astype(int)
    
    acc_accuracy = (test_predictions_df["Acceleration_Binary"].values == y_test["Acceleration"].values).mean()
    brake_accuracy = (test_predictions_df["Braking_Binary"].values == y_test["Braking"].values).mean()
    
    print("\nAcceleration Accuracy (Test): {:.2%}".format(acc_accuracy))
    print("Braking Accuracy (Test): {:.2%}".format(brake_accuracy))
    
    plt.figure(figsize=(10, 6))
    
    original_steering_pred = steering_scaler.inverse_transform(test_predictions_df[["Steering"]])
    original_steering_actual = y_test[["Steering"]]
    
    plt.scatter(original_steering_actual.values[:100], original_steering_pred[:100], alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.xlabel('Actual Steering')
    plt.ylabel('Predicted Steering')
    plt.title('Predicted vs Actual Steering Values (First 100 samples)')
    plt.savefig('steering_prediction.png')
    plt.close()
    
    print("Visualization saved as 'steering_prediction.png'")

def main():
    print("=== Building Racing Simulation Model ===")
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    dataframes = read_csv_files(data_dir)
    
    if not dataframes:
        print("No data to process. Exiting.")
        return
    
    X, y = preprocess_data(dataframes)
    
    if X is None or y is None:
        print("Failed to preprocess data. Exiting.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    print(f"\nData split into:")
    print(f"- Training: {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
    print(f"- Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(X):.1%})")
    print(f"- Testing: {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%})")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("Data scaled with StandardScaler")
    
    model, steering_scaler = build_and_train_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    evaluate_model(model, steering_scaler, X_test_scaled, y_test)
    
    os.makedirs("r", exist_ok=True)
    
    print("\n=== Saving Model and Scalers ===")
    joblib.dump(model, 'models/racing_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(steering_scaler, 'models/steering_scaler.joblib')
    
    print("\nModel and scalers saved to:")
    print("- models/racing_model.joblib")
    print("- models/scaler.joblib")
    print("- models/steering_scaler.joblib")
    
    print("\n=== Model Building Complete ===")

if __name__ == "__main__":
    main() 