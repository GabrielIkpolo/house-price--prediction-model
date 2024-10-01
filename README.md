## This is a simple model to Predict house prices


The features

The features are the columns in the X dataframe, which are:

    bedrooms: The number of bedrooms in the house
    bathrooms: The number of bathrooms in the house
    sqft_living: The living area of the house in square feet
    sqft_lot: The size of the lot on which the house is built
    floors: The number of floors in the house
    waterfront: Whether the house has a waterfront view (0 or 1)
    view: The quality of the view from the house (0, 1, 2, 3, or 4)
    condition: The condition of the house (1-5, where 1 is the worst and 5 is the best)
    grade: The grade of the house (1-13, where 1 is the lowest and 13 is the highest)
    sqft_above: The square footage of the house above ground level
    sqft_basement: The square footage of the house below ground level
    yr_built: The year the house was built
    yr_renovated: The year the house was last renovated
    zipcode: The zip code of the house's location
    lat: The latitude of the house's location
    long: The longitude of the house's location
    sqft_living15: The average living area of the 15 nearest neighbors
    sqft_lot15: The average lot size of the 15 nearest neighbors


# You can test the model manually for prediction by entering:

How to predict a value manually

To predict a value manually, you would need to provide the values for each of these features, and then use the trained model to make a prediction.


    bedrooms: 4
    bathrooms: 2.5
    sqft_living: 2500
    sqft_lot: 5000
    floors: 2
    waterfront: 0
    view: 2
    condition: 4
    grade: 8
    sqft_above: 2500
    sqft_basement: 0
    yr_built: 2010
    yr_renovated: 0
    zipcode: 98103
    lat: 47.5
    long: -122.2
    sqft_living15: 2400
    sqft_lot15: 5000

```
# In Code:
import pandas as pd

# Create a new dataframe with the input features
input_df = pd.DataFrame({
    'bedrooms': [4],
    'bathrooms': [2.5],
   'sqft_living': [2500],
   'sqft_lot': [5000],
    'floors': [2],
    'waterfront': [0],
    'view': [2],
    'condition': [4],
    'grade': [8],
   'sqft_above': [2500],
   'sqft_basement': [0],
    'yr_built': [2010],
    'yr_renovated': [0],
    'zipcode': [98103],
    'lat': [47.5],
    'long': [-122.2],
   'sqft_living15': [2400],
   'sqft_lot15': [5000]
})

# Load the trained model
model = joblib.load('best_model.pkl')

# Make a prediction
prediction = model.predict(input_df)

print("Predicted price:", prediction)

```

###  After deployment test using the api end point, 

`localhost:5000/predict` 

add the features in the body of your api client

```
[{
    "bedrooms": 4,
    "bathrooms": 2.5,
    "sqft_living": 2500,
    "sqft_lot": 5000,
    "floors": 2,
    "waterfront": 0,
    "view": 2,
    "condition": 4,
    "grade": 8,
    "sqft_above": 2500,
    "sqft_basement": 0,
    "yr_built": 2010,
    "yr_renovated": 0,
    "zipcode": 98103,
    "lat": 47.5,
    "long": -122.2,
    "sqft_living15": 2400,
    "sqft_lot15": 5000
}]

```

### Note 
You may need to greate a virtual environment and install the required libraries eg:

`python3 -m venv my-vrtual-env`

`source ~/my-virtual-env/bin/activate`

cd into the project directory

` pip install -r requirements.txt`

You can install jupyter lab if you like it also

`pip install jupyter-lab`