from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('./model/trained_model.pkl')
scaler = joblib.load('./model/scaler.pkl')

numerical_features = ['bedrooms', 'bathrooms','sqft_living','sqft_lot', 'floors', 
'waterfront', 'view', 'condition', 'grade','sqft_above','sqft_basement', 'yr_built', 
'yr_renovated', 'zipcode', 'lat', 'long','sqft_living15','sqft_lot15']

@app.route('/predict', methods=['POST'])
def predict():
    # get imput data from the request
    data= request.get_json()
    # Create a pandas dataframe from the imputs
    input_df = pd.DataFrame(data)

     # Scale the input features using the same StandardScaler object
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

   
    # make prediction using the model
    prediction = model.predict(input_df)
    
    # Return the predicted price as JSON response
    return jsonify({'Predicted_price': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)