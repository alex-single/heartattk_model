import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

df = pd.read_csv('/Medicaldataset.csv')

labelencoder = LabelEncoder()


df.dropna(inplace=True)
print(df.columns)

df["Result"] = labelencoder.fit_transform(df["Result"])

X = df.drop(columns=['Result','Systolic blood pressure','Diastolic blood pressure', 'Blood sugar','Heart rate','Gender'])
y = df['Result']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=.2, random_state=12)




model = RandomForestClassifier(random_state=42)

model.fit(X_train, Y_train)


ypred = model.predict(X_test)

cm = confusion_matrix(Y_test, ypred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labelencoder.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


from sklearn.metrics import classification_report, accuracy_score


print("Classification Report:\n")
print(classification_report(Y_test, ypred, target_names=["Positive", "Negative"]))

# Print accuracy
print(f"Accuracy: {accuracy_score(Y_test, ypred):.2f}")


import seaborn as sns

cols_for_corr = list(X.columns) + ['Result']
df_corr_selected = df[cols_for_corr].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(df_corr_selected, annot=True, fmt=".2f", cmap="RdBu_r")
plt.title("Feature Correlation Matrix (after dropping BP & Blood sugar)")
plt.show()