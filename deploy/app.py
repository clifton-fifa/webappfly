from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np 
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)

# โหลดโมเดลที่ฝึกมาแล้ว
minimal_model = joblib.load(r'C:\Users\Admin\Downloads\webappfly\deploy\model\minimal_model.pkl')  # Minimal mode model
full_model = joblib.load(r'C:\Users\Admin\Downloads\webappfly\deploy\model\full_model.pkl')  # Full mode model

# ฟีเจอร์ที่ใช้ในแต่ละโหมด
minimal_shape_columns = ['a(1-2)', 'b(2-3)', 'c(3-4)', 'd(4-5)', 'e(6-7)', 'g(9-10)']
categorical_columns = ['Gena color', 'Body color']

full_shape_columns = ['a(1-2)', 'b(2-3)', 'c(3-4)', 'd(4-5)', 'e(6-7)', 'f(7-10)', 'g(9-10)', 'h(9-15)', 
                      'i(15-16)', 'j(14-15)', 'k(13-14)', 'l(13-17)', 'm(17-18)', 'n(1-18)', 'o(2-13)', 
                      'p(3-12)', 'q(12-13)', 'r(5-12)', 's(11-14)', 't(8-11)', 'u(7-8)', 'v(8-9)', 'w(11-12)']

# ฟังก์ชันปรับขนาดฟีเจอร์ a+b+c+d = 5000
def rescale_abcd(row):
    cols = [col for col in ['a(1-2)', 'b(2-3)', 'c(3-4)', 'd(4-5)'] if col in row.index]
    total = sum(row[cols])
    if total != 0:
        factor = 5000 / total
        for col in cols:
            row[col] *= factor
    return row

# ฟังก์ชันเตรียมข้อมูลสำหรับการทำนาย
def prepare_input_data(data, mode):
    df = pd.DataFrame([data])
    
    # ปรับขนาดฟีเจอร์สำหรับทั้ง minimal และ full modes
    df = df.apply(rescale_abcd, axis=1)

    # One-hot encode categorical data ในโหมด minimal
    if mode == 'minimal':
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=False)

        # ตรวจสอบว่า categorical columns ถูกเข้ารหัสครบถ้วนหรือไม่
        expected_columns = [f"{col}_{val}" for col in categorical_columns for val in ['orange', 'white', 
                        'Metallic Green', 'Metallic Blue', 'Cupreous', 'Gray']]
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0  # เติมค่า 0 สำหรับคอลัมน์ที่ขาด

    return df

# ฟังก์ชันทำนายผลลัพธ์
def predict(data, model, mode):
    df = prepare_input_data(data, mode)

    # เติมคอลัมน์ที่ขาดหายด้วย 0
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    df = df[model.feature_names_in_]
    prediction = model.predict(df)[0]
    
    # ดึงความน่าจะเป็นสูงสุดจากการทำนาย
    probabilities = model.predict_proba(df)[0]
    max_probability = np.max(probabilities)
    
    return prediction, max_probability

# สร้าง Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict_species_family():
    try:
        logging.debug(f"Received form data: {request.form}")

        mode = request.form.get('mode')  # ตรวจสอบโหมดที่เลือก ('minimal' หรือ 'full')
        measurements = {}

        if mode == 'minimal':
            # รับข้อมูลจากฟิลด์โหมด minimal
            measurements['a(1-2)'] = float(request.form.get('minimal_a12') or 0)
            measurements['b(2-3)'] = float(request.form.get('minimal_b23') or 0)
            measurements['c(3-4)'] = float(request.form.get('minimal_c34') or 0)
            measurements['d(4-5)'] = float(request.form.get('minimal_d45') or 0)
            measurements['e(6-7)'] = float(request.form.get('minimal_e67') or 0)
            measurements['g(9-10)'] = float(request.form.get('minimal_g910') or 0)
            measurements['Gena color'] = request.form.get('minimal_genaColor', '')
            measurements['Body color'] = request.form.get('minimal_bodyColor', '')

            # ใช้โมเดล minimal สำหรับการทำนาย
            model = minimal_model

        elif mode == 'full':
            # รับข้อมูลจากฟิลด์โหมด full
            measurements['a(1-2)'] = float(request.form.get('full_a12') or 0)
            measurements['b(2-3)'] = float(request.form.get('full_b23') or 0)
            measurements['c(3-4)'] = float(request.form.get('full_c34') or 0)
            measurements['d(4-5)'] = float(request.form.get('full_d45') or 0)
            measurements['e(6-7)'] = float(request.form.get('full_e67') or 0)
            measurements['f(7-10)'] = float(request.form.get('full_f710') or 0)
            measurements['g(9-10)'] = float(request.form.get('full_g910') or 0)
            measurements['h(9-15)'] = float(request.form.get('full_h915') or 0)
            measurements['i(15-16)'] = float(request.form.get('full_i1516') or 0)
            measurements['j(14-15)'] = float(request.form.get('full_j1415') or 0)
            measurements['k(13-14)'] = float(request.form.get('full_k1314') or 0)
            measurements['l(13-17)'] = float(request.form.get('full_l1317') or 0)
            measurements['m(17-18)'] = float(request.form.get('full_m1718') or 0)
            measurements['n(1-18)'] = float(request.form.get('full_n118') or 0)
            measurements['o(2-13)'] = float(request.form.get('full_o213') or 0)
            measurements['p(3-12)'] = float(request.form.get('full_p312') or 0)
            measurements['q(12-13)'] = float(request.form.get('full_q1213') or 0)
            measurements['r(5-12)'] = float(request.form.get('full_r512') or 0)
            measurements['s(11-14)'] = float(request.form.get('full_s1114') or 0)
            measurements['t(8-11)'] = float(request.form.get('full_t811') or 0)
            measurements['u(7-8)'] = float(request.form.get('full_u78') or 0)
            measurements['v(8-9)'] = float(request.form.get('full_v89') or 0)
            measurements['w(11-12)'] = float(request.form.get('full_w1112') or 0)

            # ใช้โมเดล full สำหรับการทำนาย
            model = full_model

        # ทำการทำนาย
        species, species_probability = predict(measurements, model, mode)

        logging.debug(f"Prediction result: Species={species}")
        logging.debug(f"Probability: Species={species_probability}")

        result = {
            'success': True,
            'species': species,
            'probability': float(species_probability)
        }

        logging.debug(f"Sending result to template: {result}")
        return jsonify(result)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logging.error(error_message)
        return jsonify({'success': False, 'error_message': error_message})

# รัน Flask app
if __name__ == "__main__":
    app.run(debug=True)


