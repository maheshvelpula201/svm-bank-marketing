
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
file_path = r"C:\Users\mahes\Downloads\archive (7)\bank-additional-full.csv"
df = pd.read_csv(file_path, sep=";")

print("Original dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# 2. Remove duplicate rows
df = df.drop_duplicates()

# 3. Separate features and target
X = df.drop(columns=["y"])
y = df["y"].astype(str).str.strip().str.strip('"')

# Handle unknown values as missing
X = X.replace("unknown", pd.NA)

# 4. Identify numerical and categorical columns
numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

# 5. Preprocess numerical and categorical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# 6. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7. Build and train SVM
model = Pipeline([
    ("preprocessor", preprocessor),
    ("svm", SVC(kernel="rbf", C=1.0))
])

print("\nTraining SVM...")
model.fit(X_train, y_train)

# 8. Evaluate model
y_pred = model.predict(X_test)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("SVM Accuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nSVM training and evaluation completed!")


# Predict for a new customer
new_customer = X_test.iloc[[0]]

prediction = model.predict(new_customer)

print("\nNew Customer Prediction:", prediction[0])
print("Actual Outcome:", y_test.iloc[0])

