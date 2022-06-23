import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#create a title and sub title

st.write("""
# DIABETES DETECTION
DETECT IF SOMEONE HAS DIABETES USING MACHINE LEARNING AND PYTHON
""")

#open and display an image

image = Image.open('C:/Users/kay/Desktop/project/diabetes.jpg')
st.image(image, caption='machine learning', use_column_width=True)

#GET THE DATA
df = pd.read_csv('C:/Users/kay/Desktop/project/diabetes.csv')
#set sub header

st.subheader('Data Information:')
#show the data as a table
st.dataframe(df)

#show statistics on the data
st.write(df.describe())
#show the data as a chart
chart = st.bar_chart(df)

#split the data into independent 'X' and dependent 'Y' variables
X = df.iloc[:, 0:8].values
Y = df.iloc[:,-1].values
#split the dataset into 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
#get the features input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 10, 5)
    glucose = st.sidebar.slider('glucose', 0, 200, 120)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 25)
    insulin = st.sidebar.slider('insulin', 0.0, 845.0, 30.5)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 30)

    #store dictionary into a variable
    user_data = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'BMI': BMI,
        'DPF': DPF,
        'age': age
         }
    #transform the data into a data frame
    features = pd.DataFrame(user_data, index = [0])
    return features
#store the user input into a variable
user_input = get_user_input()
#set a subheader and display the user input
st.write(user_input)

#create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#show the model metrics
st.subheader('Model Test Accuracy:')
st.write( str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100)+'%')
#store the model prediction in a variable
prediction = RandomForestClassifier.predict(user_input)

#set a subheader and display the classification
st.subheader('classification: ')
st.write(prediction)
    
if (prediction[0] == 0):
    st.write("""
    ### The Person is not Diabetic""")
else:
    st.write("""
    ### The Person is Diabetic""")