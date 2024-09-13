import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def encode_colors(data):
    data['Gena color'] = data['Gena color'].map({'Orange': 0, 'White': 1})
    data['Body color'] = data['Body color'].map({
        'Metallic green': 0,
        'Metallic blue': 1,
        'Cupreous': 2,
        'Grey': 3
    })
    return data

def rescale_abcd(row):
    cols = ['a(1-2)', 'b(2-3)', 'c(3-4)', 'd(4-5)']
    total = sum(row[cols])
    if total != 0:
        factor = 5000 / total
        for col in cols:
            row[col] *= factor
    return row

def train_model():
    df = pd.read_excel(r'C:\\Users\\Admin\\Downloads\\web7_9\\deploy\\model\\Trained-minimal.xlsx')
    
    # Apply rescaling to a(1-2), b(2-3), c(3-4), d(4-5)
    df = df.apply(rescale_abcd, axis=1)
    
    df = encode_colors(df)

    # Use all features for X, but only specific ones for model training
    X_all = df[['a(1-2)', 'b(2-3)', 'c(3-4)', 'd(4-5)', 'e(6-7)', 'g(9-10)', 'Gena color', 'Body color']]
    X_model = df[['a(1-2)', 'b(2-3)', 'e(6-7)', 'g(9-10)', 'Gena color', 'Body color']]
    y = df['Species']

    X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    return model, X_all.columns.tolist()

def predict(model, input_data, feature_names):
    # Ensure input_data is a DataFrame with the correct column names
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Apply rescaling to a(1-2), b(2-3), c(3-4), d(4-5)
    input_df = input_df.apply(rescale_abcd, axis=1)
    
    # Encode colors
    input_df = encode_colors(input_df)
    
    # Select only the features used by the model
    model_features = ['a(1-2)', 'b(2-3)', 'e(6-7)', 'g(9-10)', 'Gena color', 'Body color']
    X_pred = input_df[model_features]
    
    return model.predict(X_pred)[0]

if __name__ == "__main__":
    model, feature_names = train_model()
    
    model_filename = 'minimal_model.pkl'
    joblib.dump((model, feature_names), model_filename)
    print(f"Model and feature names saved to {model_filename}")

    # Example usage of predict function
    sample_input = {
        'a(1-2)': 1000, 'b(2-3)': 1000, 'c(3-4)': 1000, 'd(4-5)': 1000,
        'e(6-7)': 100, 'g(9-10)': 200,
        'Gena color': 'Orange', 'Body color': 'Metallic green'
    }
    prediction = predict(model, sample_input, feature_names)
    print(f"Prediction for sample input: {prediction}")