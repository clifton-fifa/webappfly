import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ฟังก์ชันการเข้ารหัสสี Gena และ Body
def encode_colors(data):
    data['Gena color'] = data['Gena color'].map({'Orange': 0, 'White': 1})
    data['Body color'] = data['Body color'].map({
        'Metallic green': 0,
        'Metallic blue': 1,
        'Cupreous': 2,
        'Grey': 3
    })
    return data

# ฟังก์ชันสำหรับ rescale คอลัมน์ a(1-2), b(2-3), c(3-4), d(4-5)
def rescale_abcd(row):
    cols = [col for col in ['a(1-2)', 'b(2-3)', 'c(3-4)', 'd(4-5)'] if col in row.index]
    total = sum(row[cols])
    if total != 0:
        factor = 5000 / total
        for col in cols:
            row[col] *= factor
    return row

# ฟังก์ชันการฝึกโมเดล
def train_model():
    # อ่านข้อมูลจากไฟล์ Excel
    df = pd.read_excel(r'C:\\Users\\Admin\\Downloads\\web7_9\\deploy\\model\\Trained-minimal.xlsx')  # ใช้ read_excel()
    
    # Rescale columns a(1-2), b(2-3), c(3-4), d(4-5)
    df = df.apply(rescale_abcd, axis=1)
    
    df = encode_colors(df)

    # แบ่งข้อมูลฝึก/ทดสอบ
    X = df[['a(1-2)', 'b(2-3)', 'e(6-7)', 'g(9-10)', 'Gena color', 'Body color']]
    y = df['Species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # สร้างโมเดล Decision Tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # ทดสอบโมเดล
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # แสดงผลลัพธ์การทดสอบ
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    return model

# บันทึกโมเดลหลังฝึกฝน
if __name__ == "__main__":
    model = train_model()
    
    # บันทึกโมเดลด้วย joblib
    model_filename = 'minimal_model.pkl'
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

