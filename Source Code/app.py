from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model using joblib
model = joblib.load('mymodel.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        age = int(request.form['age'])
        occupation = int(request.form['occupation'])
        stay_in_current_city_years = int(request.form['stay_in_current_city_years'])
        product_category_1 = int(request.form['product_category_1'])
        product_category_2 = int(request.form['product_category_2'])
        product_category_3 = int(request.form['product_category_3'])

        # Perform any necessary data preprocessing here

        # Make a prediction using the loaded model
        prediction = model.predict([[age, occupation, stay_in_current_city_years,
                                     product_category_1, product_category_2, product_category_3]])

        # Return the prediction result
        return render_template('result.html', prediction=prediction[0])

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
