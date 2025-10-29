import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

#Load dataset
df = pd.read_csv("data_modified.csv")
print("Data loaded successfully!")
print(df.head())

#Remove duplicates and handle missing data
df = df.drop_duplicates()
df = df.dropna()

#Split features and labels
X = df.drop(['Label'], axis=1)
y = df['Label']

#Train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



#Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "SVM (RBF Kernel)": SVC(kernel='rbf', class_weight='balanced'),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=10)
}

#Evaluate models
results = []

for name, model in models.items():
    print(f"\n Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4)
    })

#Display results as a table
results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:\n")
print(results_df.to_string(index=False))

#Save results to CSV
results_df.to_csv("model_performance.csv", index=False)
print("\nResults saved to 'model_performance.csv'")

#Save the best model (based on F1-score)
best_model_name = results_df.loc[results_df['F1 Score'].idxmax(), 'Model']
best_model = models[best_model_name]
import pickle
pickle.dump(best_model, open("best_model.pickle", "wb"))
print(f"\nBest model saved: {best_model_name}")
