# !pip install -q streamlit
# !npm install localtunnel
# SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
# Core Pkgs
import sklearn
import streamlit as st
import scipy
import scipy.stats
import itertools
import pickle

# Importing Libraries for EDA
import pandas as pd
import numpy as np


# Importing Libraries for Data Visualisation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

def main():
  activities = ["EDA", "Plots", "ML"]
  choice = st.sidebar.selectbox("Select Activities", activities)

  if choice == "EDA":
    st.subheader("EDA")
    data = st.file_uploader("Upload a Dataset", type = ["csv"])



    if data is not None:
      df = pd.read_csv(data)
      st.dataframe(df.head())

      if st.checkbox("Show Datatypes"):
        st.write(df.dtypes)

      if st.checkbox("Show Shape"):
        st.write(df.shape)

      if st.checkbox("Show Columns"):
        all_columns = df.columns.to_list()
        st.write(all_columns)

      if st.checkbox("Summary"):
        st.write(df.describe())

      if st.checkbox("Short Information"):
        st.write(df.info())

      if st.checkbox("Show Selected Columns"):
        selected_columns = st.multiselect("Select Columns", df.columns.to_list())
        new_df = df[selected_columns]
        st.dataframe(new_df)

      if st.checkbox("Value Counts"):
        selected_columns = st.selectbox("Select Columns", df.columns.to_list())
        new_df = df[selected_columns].value_counts()
        new_df

      if st.checkbox("Correlation Plot(Seaborn)"):
        st.write(sns.heatmap(df.corr(),annot=True))
        plt.show()
        st.pyplot()

      if st.checkbox("Pie Plot"):
        all_columns = df.columns.to_list()
        column_to_plot = st.selectbox("Select 1 Column",all_columns)
        pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%", wedgeprops = {"edgecolor" : "black", 'linewidth': 1, 'antialiased': True})
        st.write(pie_plot)
        st.pyplot()

  elif choice == "Plots":
    st.subheader("Data Visualization")
    data = st.file_uploader("Upload a Dataset", type = ["csv"])

    if data is not None:
      df = pd.read_csv(data)
      st.dataframe(df.head())

      if st.checkbox("Show Value Counts"):
        selected_columns = st.selectbox("Select Columns", df.columns.to_list())
        st.write(df[selected_columns].value_counts().plot(kind = "bar"))
        st.pyplot()

# Customizable Plot

      all_columns_names = df.columns.tolist()
      type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
      selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

      if st.button("Generate Plot"):
        st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

# Plot By Streamlit
        if type_of_plot == 'area':
          cust_data = df[selected_columns_names].value_counts().plot(kind = "area")
          st.write(cust_data)
          st.pyplot()

        elif type_of_plot == 'bar':
          cust_data = df[selected_columns_names].value_counts().plot(kind = "bar")
          st.write(cust_data)
          st.pyplot()

        elif type_of_plot == 'line':
          cust_data = df[selected_columns_names].value_counts().plot(kind = "line")
          st.write(cust_data)
          st.pyplot()

# Custom Plot
        elif type_of_plot:
          cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
          st.write(cust_plot)
          st.pyplot()


  elif choice == "ML":
    st.subheader("ML Web App")
    data = st.file_uploader("Upload a Dataset", type = ["csv"])

    if data is not None:
      df = pd.read_csv(data)
      st.dataframe(df.head())

      name = st.selectbox("Select Car Name", options = df["name"].unique())
      st.write(name)
      split = name.split(" ")
      car_maker = split[0]
      car_model = split[1]


      option = list(itertools.chain(range(1980, 2024, 1)))

      years = st.selectbox("Select year of model", options = option)
      st.write(years)

      current_year = 2023
      no_of_total_years = current_year-years

      km_driven = st.slider('Select km driven Length', 0.0, 850000.0, step = 1000.0)

      fuel_options = ["Diesel", "Petrol", "CNG", "LPG", "Electric"]
      fuel = st.select_slider("Select fuel Width", options = fuel_options)

      seller_type_options = ["Individual", "Dealer", "Trustmark Dealer"]
      seller_type = st.select_slider('Select seller_type Length', options = seller_type_options)

      transmission_options = ["Manual", "Automatic"]
      transmission = st.select_slider('Select transmission Width', options = transmission_options)

      owner_options = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]
      owner = st.select_slider('Select owner Width', options = owner_options)

      test  = [[ name, years, km_driven, fuel, seller_type, transmission, owner]]
      st.write('Test_Data', test)


      if st.button('Predict', key = "int"):
        input_data = {"car_maker": [car_maker],
                    "car_model": [car_model],
                    "no_of_total_years":[no_of_total_years],
                    'km_driven': [km_driven],
                    'fuel': [fuel],
                    'seller_type': [seller_type],
                    'transmission': [transmission],
                    'owner': [owner]}

        input_df = pd.DataFrame(input_data)

        # Update the file path to reflect the correct location in Streamlit cloud
        pkl_file_path = "pipeline_model.pkl"

        # Load the pickle file
        with open(pkl_file_path, "rb") as file:
          pipeline = pickle.load(file)


        predictions = pipeline.predict(input_df)
        
        if predictions<0:
            st.success("We have identified that there were inaccuracies in the details entered by you.\U0001F600")
        else:
            st.success(round(predictions[0]))

      #  st.success(predictions[0])


if __name__ == "__main__":
    main()
