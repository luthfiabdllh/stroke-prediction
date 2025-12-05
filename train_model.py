import pandas as pd
import numpy as np
import kagglehub
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

def main():
    print("Downloading dataset...")
    try:
        path = kagglehub.dataset_download('fedesoriano/stroke-prediction-dataset')
        csv_path = path + '/healthcare-dataset-stroke-data.csv'
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        # Fallback: check local directory
        csv_path = 'healthcare-dataset-stroke-data.csv'
        
    print(f"Loading dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Dataset not found. Please ensure 'healthcare-dataset-stroke-data.csv' is in the directory.")
        return

    # Drop ID column
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # Define features and target
    X = df.drop('stroke', axis=1)
    y = df['stroke']

    # Define column groups
    categorical_onehot = ['gender', 'work_type', 'smoking_status']
    categorical_ordinal = ['ever_married', 'Residence_type']
    numerical = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

    # Create Preprocessing Pipeline
    # 1. Impute BMI (and other nums) with mean
    # 2. Encode Categoricals
    # 3. Scale everything
    
    # We need separate transformers for different column types
    
    # Numerical Transformer: Impute -> Scale (Wait, scaling should be applied to ALL features at the end? 
    # The notebook applied StandardScaler to X_train which included encoded features.
    # So we should Encode first, then Scale result.)
    
    # But ColumnTransformer runs in parallel.
    # So we need:
    # Branch 1 (Num): Impute -> Pass
    # Branch 2 (Cat OneHot): OneHot
    # Branch 3 (Cat Ordinal): Ordinal
    # Then combine -> Scale.
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    categorical_ordinal_transformer = OrdinalEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical),
            ('cat_onehot', categorical_onehot_transformer, categorical_onehot),
            ('cat_ordinal', categorical_ordinal_transformer, categorical_ordinal)
        ],
        remainder='drop' # Should be nothing left, but good practice
    )

    # Full Pipeline (Preprocessing + Scaling)
    # Note: SMOTE cannot be in sklearn Pipeline (it changes n_samples). 
    # We apply SMOTE manually on train set.
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler())
    ])

    # Split Data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Fit and Transform Train Data
    print("Preprocessing training data...")
    X_train_prep = pipeline.fit_transform(X_train)
    
    # Apply SMOTE
    print("Applying SMOTE...")
    sm = SMOTE(random_state=2)
    X_train_res, y_train_res = sm.fit_resample(X_train_prep, y_train)

    # Train Model
    print("Training Random Forest...")
    # Parameters from notebook: criterion='gini', n_estimators=100, random_state=0
    clf = RandomForestClassifier(criterion='gini', n_estimators=100, random_state=0)
    clf.fit(X_train_res, y_train_res)

    # Evaluate
    print("Evaluating model...")
    X_test_prep = pipeline.transform(X_test)
    y_pred = clf.predict(X_test_prep)
    y_prob = clf.predict_proba(X_test_prep)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob)}")

    # Save Artifacts
    print("Saving artifacts...")
    joblib.dump(pipeline, 'preprocessing_pipeline.joblib')
    joblib.dump(clf, 'model.joblib')
    print("Done!")

if __name__ == "__main__":
    main()
