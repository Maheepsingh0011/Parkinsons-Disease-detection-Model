# Parkinsons-Disease-detection-Model
This project implements a machine learning model to predict the presence of Parkinson's disease based on various biomedical voice measurements.
mport pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore') # Suppress warnings

# --- 1. Data Loading ---
try:
    from google.colab import files
    print("Please upload 'parkinsons data.csv' when prompted.")
    uploaded = files.upload()
    file_path = "parkinsons data.csv"
except ImportError:
    print("Not in Google Colab. Assuming 'parkinsons data.csv' is in the current directory.")
    file_path = "parkinsons data.csv"
except Exception as e:
    print(f"Error during file access: {e}")
    print("Ensure 'parkinsons data.csv' is available.")
    exit()

data = pd.read_csv(file_path)

# --- 2. Exploratory Data Analysis (EDA) ---
print("\n--- Data Head ---")
print(data.head())
print("\n--- Data Info ---")
data.info()
print("\n--- Data Shape ---")
print(f"Dataset shape: {data.shape}")
print("\n--- Missing Values ---")
print(data.isnull().sum())
print("\n--- Data Description ---")
print(data.describe())
print("\n--- Target Variable Value Counts ('status' column) ---")
print(data['status'].value_counts())

# --- 3. Data Preprocessing: Feature and Target Separation ---
X = data.drop(columns=['name', 'status'], axis=1)
y = data['status']

print("\n--- X Head (Features) ---")
print(X.head())
print("\n--- y Head (Target) ---")
print(y.head())

# --- 4. Data Splitting ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# --- 5. Data Standardization (Scaling) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit on training data, then transform
X_test_scaled = scaler.transform(X_test)     # Transform test data using the same scaler

X_train = pd.DataFrame(X_train_scaled, columns=X.columns) # Convert back to DataFrame
X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

print("\n--- X_train after scaling (first 5 rows) ---")
print(X_train.head())

# --- 6. Model Training ---
model = SVC(kernel='linear') # Initialize SVM classifier
model.fit(X_train, y_train)  # Train the model with scaled training data

# --- 7. Model Evaluation ---
pred_train = model.predict(X_train)
train_acc = accuracy_score(y_train, pred_train)
print(f"\nTraining Accuracy: {train_acc:.4f}")

pred_test = model.predict(X_test)
test_acc = accuracy_score(y_test, pred_test)
print(f"Test Accuracy: {test_acc:.4f}")

# --- 8. Predictive Model for New Input Data ---
# Input data for prediction. It must have 22 features, matching the trained model.
# The original tuple had 23 elements; the 17th element (index 16) is removed.
input_data_full = (122.4, 148.65, 113.819, 0.00968, 0.00008, 0.00465, 0.00696, 0.01394, 0.06134, 0.626, 0.03134, 0.04518, 0.04368, 0.09403, 0.01929, 19.085, 1, 0.458359, 0.819521, -4.075192, 0.33559, 2.486855, 0.368674)
input_data = input_data_full[:16] + input_data_full[17:]

print(f"\nOriginal input_data_full had {len(input_data_full)} features.")
print(f"Corrected input_data for prediction now has {len(input_data)} features.")

input_array = np.asarray(input_data)
reshaped_data = input_array.reshape(1, -1) # Reshape for single sample prediction

std_data = scaler.transform(reshaped_data) # Apply same scaling to new input

prediction = model.predict(std_data) # Make the prediction

# --- 9. Output Prediction Result ---
print(f"\nModel's raw prediction: {prediction[0]}")

if prediction[0] == 0:
    print("Prediction for new input: The person is Healthy.")
else:
    print("Prediction for new input: The person has Parkinson's disease.")
