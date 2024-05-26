import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

# Load the training data
df = pd.read_csv('co2_emissions (1).csv')

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Encode 'transmission' column
df['transmission'] = label_encoder.fit_transform(df['transmission'])

# Encode 'fuel_type' column
df['fuel_type'] = label_encoder.fit_transform(df['fuel_type'])


# Split the data into features (X_train) and target variable (y_train)
X_train = df[['engine_size', 'cylinders', 'transmission', 'fuel_type', 'fuel_consumption_comb(mpg)']]
y_train = df['co2_emissions']

randomforest = RandomForestRegressor()
randomforest.fit(X_train, y_train)

# Load the trained model
with open('randomforest.pkl', 'rb') as f:
    randomforest = pickle.load(f)

if hasattr(randomforest, 'estimators_'):
    print("Model is fitted.")
else:
    print("Model is not fitted. Fitting the model now...")
    # Split the data into features (X_train) and target variable (y_train)
    X_train = df[['engine_size', 'cylinders', 'transmission', 'fuel_type', 'fuel_consumption_comb(mpg)']]
    y_train = df['co2_emissions']
    # Fit the model to the training data
    randomforest.fit(X_train, y_train)
    print("Model fitted successfully.")

# Function to predict CO2 emissions
def predict_co2_emissions(engine_size, cylinders, transmission, fuel_type, fuel_consumption):
    # Encode categorical variables
    transmission_mapping = {"Automatic": 0, "Manual": 1, "Automatic Select Shift": 2, "Continuously Variable": 3, "Automated Manual": 4}
    fuel_type_mapping = {"Regular Gasoline": 0, "Premium Gasoline": 1, "Diesel": 2, "Ethanol": 3, "Natural Gas": 4}
    transmission_encoded = transmission_mapping[transmission]
    fuel_type_encoded = fuel_type_mapping[fuel_type]
    # Make prediction
    input_data = [[engine_size, cylinders, transmission_encoded, fuel_type_encoded, fuel_consumption]]
    predicted_co2 = randomforest.predict(input_data)
    return predicted_co2[0]

# Define the plot_bar function outside of the main function
def plot_bar(data, x_label, y_label, title):
    plt.figure(figsize=(23, 5))
    fig = sns.barplot(data=data, x=x_label, y=y_label)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.bar_label(plt.gca().containers[0], fontsize=9)
    st.pyplot()

def main():
    css = """
    <style>
    body {
       background-image: url("background.jpg");
       background-size: cover;
    }
    .sidebar .sidebar-content {
       background-image: url("sidebar.jpg");
       background-size: cover;
    }
    </style>
    """

    # Display the CSS
    st.markdown(css, unsafe_allow_html=True)

    st.title("CO2 Emissions Predictor")

    # Sidebar
    st.set_option('deprecation.showPyplotGlobalUse', False)

    user_input = st.sidebar.selectbox('Please select', ('Project Details', 'Visualization', 'Prediction'))

    if user_input == 'Project Details':
        st.header("CO2 Emissions by Vehicle")
        st.subheader("CO2 Emissions")
        st.write("The main objective of the project is to model the CO2 emissions as a function of several car engine features. The data is as follows")
        st.write(df)
        st.write("The Description of Numerical Variables")
        st.write(df.describe(include=np.number))
        st.write("The Description of Categorical Variables")
        st.write(df.describe(include=object))

       # Box Plots-------------------------------------------------------------------------------------------
        st.header("Box Plots")

        st.subheader("Box Plots for Numerical Features")
        features = ['engine_size', 'cylinders', 'fuel_consumption_comb(mpg)', 'co2_emissions']

       # Adjust the layout based on the number of features
        num_features = len(features)
        num_rows = (num_features + 2) // 3  
        num_cols = min(num_features, 3)     

        plt.figure(figsize=(20, 10))

        for i, feature in enumerate(features, start=1):
            plt.subplot(num_rows, num_cols, i)
            plt.boxplot(df[feature])  
            plt.title(feature)


        st.pyplot()

       # Calculate the IQR (Interquartile Range) for each feature
        Q1 = df[features].quantile(0.25)
        Q3 = df[features].quantile(0.75)
        IQR = Q3 - Q1

        # Define the lower and upper bounds for outlier detection
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Remove outliers from the DataFrame
        df_no_outliers = df[~((df[features] < lower_bound) | (df[features] > upper_bound)).any(axis=1)]

        # Print the number of data points before and after outlier removal
        st.write("Before removing outliers, we have", len(df), "data points.")
        st.write("After removing outliers, we have", len(df_no_outliers), "data points.")

        # Boxplot after removing outliers
        st.subheader("Boxplot after removing outliers")
        plt.figure(figsize=(20, 10))
        for i, feature in enumerate(features, start=1):
            plt.subplot(2, 2, i)
            plt.boxplot(df_no_outliers[feature])
            plt.title(feature)
        st.pyplot()


    elif user_input == 'Visualization':

        st.sidebar.subheader("Visualization Options")
        user_input = st.sidebar.selectbox('Please select', ('Brands of Cars', 'Model', 'Vehicle Class', 'Engine Size', 'Cylinders', 'Transmission', 'Fuel Type', 'Prediction'))

        def plot_bar(data, x_label, y_label, title):
            plt.figure(figsize=(23, 5))
            fig = sns.barplot(data=data, x=x_label, y=y_label)
            plt.xticks(rotation=90)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.bar_label(plt.gca().containers[0], fontsize=9)
            st.pyplot()

        # Brands of Cars-------------------------------------------------------------------------------------------
        if user_input == 'Brands of Cars':

            st.subheader('Brands of Cars')
            st.write("There are 42 unique brands of cars with with 20 brands owning fewer than 100 cars and ranging from 2 to 575 cars per brand.")
            df['make'] = df['make'].astype(str)
            df_brand = df['make'].value_counts().reset_index().rename(columns={'count': 'Count'})
            plt.figure(figsize=(15, 6))
            fig1 = sns.barplot(data=df_brand, x="make", y="Count")
            plt.xticks(rotation=75)
            plt.title("All Car Companies and their Cars")
            plt.xlabel("Companies")
            plt.ylabel("Cars")
            plt.bar_label(fig1.containers[0], fontsize=7)
            st.pyplot()
            st.write(df_brand)

            # CO2 Emission variation with Brand----------------------------------------------------------------------
            st.header('Variation in CO2 emissions with different features')
            st.subheader('CO2 Emission with Brand ')
            df_co2_make = df.groupby(['make'])['co2_emissions'].mean().sort_values().reset_index()
            plt.figure(figsize=(20, 5))
            fig9 = sns.barplot(data=df_co2_make, x="make", y="co2_emissions")
            plt.xticks(rotation=90)
            plt.title("CO2 Emissions variation with Brand")
            plt.xlabel("Brands")
            plt.ylabel("CO2 Emissions(g/km)")
            plt.bar_label(plt.gca().containers[0], fontsize=8, fmt='%.1f')
            st.pyplot()

            def plot_bar(data, x_label, y_label, title):
                plt.figure(figsize=(23, 5))
                fig10 = sns.barplot(data=data, x=x_label, y=y_label)
                plt.xticks(rotation=90)
                plt.title(title)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.bar_label(plt.gca().containers[0], fontsize=9)
                st.pyplot()

        # Top 25 Models of Cars------------------------------------------------------------------------------------
        elif user_input == 'Model':
            st.subheader('Top 25 Models of Cars')
            st.write("There are 2053 unique models across the 42 brands. While 5 brands possess fewer than 10 cars and 5 brands have over 100, the majority of brands (32) have model counts between 10 and 100.")
            df_model = df['model'].value_counts().reset_index().rename(columns={'count': 'Count'})
            plt.figure(figsize=(20, 6))
            fig2 = sns.barplot(data=df_model[:25], x="model", y="Count")
            plt.xticks(rotation=75)
            plt.title("Top 25 Car Models")
            plt.xlabel("Models")
            plt.ylabel("Cars")
            plt.bar_label(fig2.containers[0])
            st.pyplot()
            st.write(df_model)

        # Vehicle Class--------------------------------------------------------------------------------------------
        elif user_input == 'Vehicle Class':
            st.subheader('Vehicle Class')
            st.write("We observe 16 distinct vehicle classes, with SUV-Small, Mid-size, and Compact being the most prevalent classes, each with over 900 cars.")
            df_vehicle_class = df['vehicle_class'].value_counts().reset_index().rename(columns={'count': 'Count'})
            plt.figure(figsize=(20, 5))
            fig3 = sns.barplot(data=df_vehicle_class, x="vehicle_class", y="Count")
            plt.xticks(rotation=75)
            plt.title("All Vehicle Class")
            plt.xlabel("Vehicle Class")
            plt.ylabel("Cars")
            plt.bar_label(fig3.containers[0])
            st.pyplot()
            st.write(df_vehicle_class)

            # CO2 Emissions variation with Vehicle Class-------------------------------------------------------------
            st.subheader('CO2 Emissions variation with Vehicle Class')
            df_co2_vehicle_class = df.groupby(['vehicle_class'])['co2_emissions'].mean().sort_values().reset_index()
            fig11 = plot_bar(df_co2_vehicle_class, "vehicle_class", "co2_emissions", "CO2 Emissions variation with Vehicle Class")
            st.pyplot()

        # Engine Sizes of Cars-------------------------------------------------------------------------------------
        elif user_input == 'Engine Size':
            st.subheader('Engine Sizes of Cars')
            st.write("There are 51 different engine sizes present, with 2-liter and 3-liter engines being the most common, while 2.2-liter and 5.8-liter engines are the least common.")
            df_engine_size = df['engine_size'].value_counts().reset_index().rename(columns={'count': 'Count'})
            plt.figure(figsize=(20, 6))
            fig4 = sns.barplot(data=df_engine_size, x="engine_size", y="Count")
            plt.xticks(rotation=90)
            plt.title("All Engine Sizes")
            plt.xlabel("Engine Size(L)")
            plt.ylabel("Cars")
            plt.bar_label(fig4.containers[0])
            st.pyplot()
            st.write(df_engine_size)

        # Cylinders-----------------------------------------------------------------------------------------------
        elif user_input == 'Cylinders':
            st.subheader('Cylinders')
            st.write("Our dataset features 8 different cylinder counts, ranging from 3 to 16 cylinders. The most common cylinder counts are 4 and 6, with 16 cylinders appearing only twice, notably in the Bugatti Chiron.")
            df_cylinders = df['cylinders'].value_counts().reset_index().rename(columns={'count': 'Count'})
            plt.figure(figsize=(20, 6))
            fig5 = sns.barplot(data=df_cylinders, x="cylinders", y="Count")
            plt.xticks(rotation=90)
            plt.title("All Cylinders")
            plt.xlabel("Cylinders")
            plt.ylabel("Cars")
            plt.bar_label(fig5.containers[0])
            st.pyplot()
            st.write(df_cylinders)

        # Transmission of Cars------------------------------------------------------------------------------------
        elif user_input == 'Transmission':
            transmission_mapping = {"A": "Automatic", "M": "Manual", "AS": "Automatic Select Shift", "AV": "Continuously Variable", "AM": "Automated Manual"}
            df["transmission"] = df["transmission"].map(transmission_mapping)
            st.subheader('Transmission')
            st.write("Five transmission types are represented in our dataset: Automatic Select shift is the most oftenly used, while Continuously Variable is the least common.")
            df_transmission = df['transmission'].value_counts().reset_index().rename(columns={'count': 'Count'})
            fig6 = plt.figure(figsize=(20, 5))
            sns.barplot(data=df_transmission, x="transmission", y="Count")
            plt.title("All Transmissions")
            plt.xlabel("Transmissions")
            plt.ylabel("Cars")
            plt.bar_label(plt.gca().containers[0])
            st.pyplot()
            st.write(df_transmission)

            def plot_bar(data, x_label, y_label, title):
                plt.figure(figsize=(23, 5))
                fig10 = sns.barplot(data=data, x=x_label, y=y_label)
                plt.xticks(rotation=90)
                plt.title(title)
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.bar_label(plt.gca().containers[0], fontsize=9)
                st.pyplot()

            # CO2 Emission variation with Transmission---------------------------------------------------------------
            st.subheader('CO2 Emission variation with Transmission')
            df_co2_transmission = df.groupby(['transmission'])['co2_emissions'].mean().sort_values().reset_index()
            fig12 = plot_bar(df_co2_transmission, "transmission", "co2_emissions", "CO2 Emission variation with Transmission")
            st.pyplot()

        # Fuel Type of Cars--------------------------------------------------------------------------------------
        elif user_input == 'Fuel Type':
            fuel_mapping = {"Z": "Premium Gasoline", "D": "Diesel", "X": "Regular Gasoline", "E": "Ethanol", "AM": "Natural Gas"}
            df["fuel_type"] = df["fuel_type"].map(fuel_mapping)
            st.subheader('Fuel Type')
            st.write("There are 5 fuel types: Regular Gasoline being the most common and Natural Gas being the least common..")
            df_fuel_type = df['fuel_type'].value_counts().reset_index().rename(columns={'count': 'Count'})
            plt.figure(figsize=(20, 5))
            fig7 = sns.barplot(data=df_fuel_type, x="fuel_type", y="Count")
            plt.title("All Fuel Types")
            plt.xlabel("Fuel Types")
            plt.ylabel("Cars")
            plt.bar_label(plt.gca().containers[0])
            st.pyplot()
            st.text("We have only one data on natural gas. So we cannot predict anything using only one data. That's why we have to drop this row.")
            st.write(df_fuel_type)


            # CO2 Emissions variation with Fuel Type--------------------------------------------------------------
            st.subheader('CO2 Emissions variation with Fuel Type')
            df_co2_fuel_type = df.groupby(['fuel_type'])['co2_emissions'].mean().sort_values().reset_index()
            fig13 = plot_bar(df_co2_fuel_type, "fuel_type", "co2_emissions", "CO2 Emissions variation with Fuel Type")
            st.pyplot()

    elif user_input == 'Prediction':
        st.header("CO2 Emission Prediction")
        st.write('Enter the vehicle specifications to predict CO2 emissions.')

        engine_size = st.number_input('Engine Size (L)', step=0.1)
        cylinders = st.number_input('Cylinders', min_value=2, max_value=16, step=1)
        transmission = st.selectbox('Transmission', ['Automatic', 'Manual', 'Automatic Select Shift', 'Continuously Variable', 'Automated Manual'])
        fuel_type = st.selectbox('Fuel Type', ['Regular Gasoline', 'Premium Gasoline', 'Diesel', 'Ethanol', 'Natural Gas'])
        fuel_consumption = st.number_input('Fuel Consumption (mpg)', step=0.1)

        if st.button('Predict CO2 Emissions'):
            predicted_co2 = predict_co2_emissions(engine_size, cylinders, transmission, fuel_type, fuel_consumption)
            st.write(f'Predicted CO2 Emissions: {predicted_co2:.2f} g/km')

if __name__ == "__main__":
  main()
