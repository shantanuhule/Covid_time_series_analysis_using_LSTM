from flask import Flask, request, render_template
import pandas as pd
import matplotlib.pyplot as plt

import io
import base64
import numpy as np
from keras.models import load_model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Read the CSV file
        df = pd.read_csv('COVID-19 cases worldwide - daily-6.csv')

        # Filter the data based on which countries are selected
        selected_country = request.form['country']
        model=None
        if 'India' == selected_country:
            model=load_model('india_model.h5')
            df=df[df['countriesAndTerritories']=='India']
        elif 'USA' == selected_country:
            df=df[df['countriesAndTerritories']=='United_States_of_America']
            model=load_model('usa_model.h5')
        
        elif 'Brazil' == selected_country:
            df=df[df['countriesAndTerritories']=='Brazil']
            model=load_model('brazil_model.h5')
        else:
            model=load_model('india_model.h5')
            df=df[df['countriesAndTerritories']=='India']

        
        df=df[['dateRep','cases']]
        # df1['dateRep']=pd.to_datetime(df1['dateRep'])
        # df1.sort_values(by='dateRep',inplace=True)
        data=df.filter(['cases']).values
        train_data_len = int(np.ceil(len(data) * .8))
        # Normalize the data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data)
        # Create the training data set
        train_data = scaled_data[0:train_data_len , : ]

        # Split the data into x_train and y_train data sets
        x_train = []
        y_train = []

        for i in range(12,len(train_data)):
            x_train.append(train_data[i-12:i,0])
            y_train.append(train_data[i,0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        # Reshape the data to fit the model
        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

                # Create the testing data set
        test_data = scaled_data[train_data_len - 12: , : ]

        # Create the x_test and y_test data sets
        x_test = []
        y_test = data[train_data_len : , : ]

        for i in range(12,len(test_data)):
            x_test.append(test_data[i-12:i,0])

        x_test = np.array(x_test)

        # Reshape the data to fit the model
        x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


                # Get the model's predicted price values
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Evaluate the model
        rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
        print(rmse)

        train = df[:train_data_len]
        valid = df[train_data_len:]
        valid['Predictions'] = predictions

        plt.figure()
        plt.title('Model')
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Cases for '+selected_country, fontsize=10)
        plt.plot(train['cases'])
        plt.plot(valid[['cases', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'])#, loc='lower right'
        
        


        # Convert the plot to an image that can be displayed in HTML
        figfile = io.BytesIO()
        plt.savefig(figfile, format='png')
        plt.close()
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue()).decode()
        # Display the plot on the webpage
        return render_template('index.html', image=figdata_png,rmse=rmse)

    # Display the upload form if the user has not yet submitted the form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
