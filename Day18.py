import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
df = pd.read_csv('Iris.csv')
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
X = df.drop(['Species', 'Id'], axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = GaussianNB()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print("🔍 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))
scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5)
print("\n🔁 Cross Validation Scores:", scores)
print("✅ Average CV Accuracy:", scores.mean())
probs = model.predict_proba(X_test_scaled)
plt.figure(figsize=(10, 6))
for i in range(len(probs)):
    plt.bar(np.arange(3) + i*4, probs[i], width=0.8)
plt.title("Prediction Probabilities for Each Test Instance")
plt.xlabel("Class Index (0, 1, 2)")
plt.ylabel("Probability")
plt.tight_layout()
plt.show()
