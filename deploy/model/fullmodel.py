import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_excel(r'C:\Users\Admin\Downloads\web7_9\deploy\model\Tested-full.xlsx')

# Drop unnecessary columns
df = df.drop(columns=['ID', 'Genus', 'No', 'Sex', 'Gena color', 'Body color'])

# Define feature columns for the full model
shape_columns = ['a(1-2)', 'b(2-3)', 'c(3-4)', 'd(4-5)', 'e(6-7)', 'f(7-10)', 
                 'g(9-10)', 'h(9-15)', 'i(15-16)', 'j(14-15)', 'k(13-14)', 
                 'l(13-17)', 'm(17-18)', 'n(1-18)', 'o(2-13)', 'p(3-12)', 
                 'q(12-13)', 'r(5-12)', 's(11-14)', 't(8-11)', 'u(7-8)', 
                 'v(8-9)', 'w(11-12)']

# Rescale features a+b+c+d to sum to 5000
def rescale_abcd(row):
    cols = [col for col in ['a(1-2)', 'b(2-3)', 'c(3-4)', 'd(4-5)'] if col in row.index]
    total = sum(row[cols])
    if total != 0:
        factor = 5000 / total
        for col in cols:
            row[col] *= factor
    return row

# Apply rescaling to the dataset
df = df.apply(rescale_abcd, axis=1)

# Features and target for Species classification
X_full = df[shape_columns]
y_full = df['Species']

# Split data into training and test sets
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# Train the full model
full_model = DecisionTreeClassifier(random_state=42)
full_model.fit(X_train_full, y_train_full)

# Test the full model
y_pred_full = full_model.predict(X_test_full)
print(f"Full Model Accuracy: {accuracy_score(y_test_full, y_pred_full):.4f}")
print(f"Full Model Report:\n{classification_report(y_test_full, y_pred_full)}")

# Save the full model
try:
    joblib.dump(full_model, 'full_model.pkl')
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving the model: {str(e)}")
