from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('./model/trained_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    # get imput data from the request
    data= request.get_json()
    # Create a pandas dataframe from the imputs
    input_df = pd.DataFrame(data)
    # make prediction using the model
    prediction = model.predict(input_df)
    
    # Return the predicted price as JSON response
    return jsonify({'Predicted_price': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)