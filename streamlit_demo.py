#importing necessary modules
from cgitb import html
from click import style
import streamlit as st
import pickle as pk
import numpy as np
import time
import matplotlib as mp
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

#importing dataframe from dataframe.py
import dataframe as dfd

#load the pickled model file
load_model=pk.load(open('path to the trained model .sav','rb'))
diabetes_df = dfd.diabetes_df_copy


#functions for visualization
def get_hist(features, to, from1):
    if to == 0 :
        dfa = diabetes_df
    else :
        dfa = diabetes_df.loc[from1:to]
    fig, ax = plt.subplots()
    sb.histplot(dfa[features], bins=13)
    ax.set_xlabel(features)
    ax.set_ylabel('no:of persons')
    st.pyplot(fig)

def get_scat(features_1 , features_2 , to, from1):
    if to == 0 :
        dfa = diabetes_df
    else :
        dfa = diabetes_df.loc[from1:to]
    c = alt.Chart(dfa).mark_circle().encode(x= features_1 ,y= features_2, color='Outcome', tooltip=[features_1, features_2,'Outcome'])
    st.altair_chart(c, use_container_width=True)

def to_from():
    html_tp ="""
        <h4 style ="color:black;font-size:18px;">please enter the range of data pointers
        </h4>
        </div>
        """
    st.write(html_tp,unsafe_allow_html= True)
    cols = st.columns((2,1,2))
    from1 = cols[0].number_input('From',0,769)
    to = cols[2].number_input('To',0,769)
    return(to , from1)

    

opt1 = st.sidebar.selectbox('Pages', ['Home','Visualization','Prediction','Feedback'])

#page 1
if opt1 == 'Home': 
    html_tp =""" 
        <p style ="color:black;font-size:16px;font-family:Georgia , serif;">Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy. Sometimes your body doesn’t make enough—or any—insulin or doesn’t use insulin well. Glucose then stays in your blood and doesn’t reach your cells.
        </p>
        <p style ="color:black;font-size:16px;font-family:Georgia , serif;">Diabetes causes vary depending on your genetic makeup, family history,ethnicity, health and environmental factors. There is no common diabetes cause that fits every type of diabetes as the causes of diabetes vary depending on the individual and the type.
        </p> 
        </div>
        """

    st.title('Diabetes Prediction')
    st.write('Analyse data and prediction')
    st.write(html_tp,unsafe_allow_html= True)
    cols = st.columns((1,1))
    cols[0].image('.../images$video/07.jpg')
    cols[0].subheader('Type 1 diabetes causes')
    cols[0].write('Type 1 diabetes is caused by the immune system destroying the cells in the pancreas that make insulin. This causes diabetes by leaving the body without enough insulin to function normally.his is called an autoimmune reaction, or autoimmune cause, because the body is attacking itself.Underlying genetic disposition may also be a type 1 diabetes cause.')
    cols[1].image('.../images$video/08.jpg')
    cols[1].subheader('Type 2 diabetes causes')
    cols[1].write('Type 2 diabetes causes are usually multifactorial – more than one diabetes cause is involved. Often, the most overwhelming factor is a family history of type 2 diabetes.This is the most likely type 2 diabetes cause.There are a variety of risk factors for type 2 diabetes, any or all of which increase the chances of developing the condition.')
    cols = st.columns((1,2))
    cols[0].subheader('symptoms include:')
    cols[0].write('Obesity')
    cols[0].write('Living a sedentary')
    cols[0].write('lifestyle')
    cols[0].write('Increasing aging')
    cols[0].write(' Bad diet')
    cols[1].image('.../images$video/01.webp')
    
    html_tp1 ="""
        <h2 style ="color:black;font-size:24px;">This is a data science powered prediction software that can predict your daibetic status in no time just fill th health pointer and get your result.
        </h2>
        <h2 style ="color:black;font-size:24px;">We also provide the data visualization section through which you can go through the data and understand various patterns,  relationships.This would definetly make a clear view about diabetes Symptoms and its stages.
        </h2>
        
        </div>
        """
 
    st.write(html_tp1,unsafe_allow_html= True)
    cols = st.columns((1,7,1))
    video_file = open('.../images$video/02.mp4', 'rb')
    video_bytes = video_file.read()
    cols[1].video(video_bytes)

    html_tp =""" 
        <p style ="color:black;font-size:18px;font-family:Georgia;">Lets fight Diabetes with proper understanding and predetection.
        </p> 
        </div>
        """
    html_tp3 =""" 
        <p style ="color:black;font-size:12px;font-family:Georgia , serif;">This content is provided as a service of the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), part of the National Institutes of Health. The NIDDK translates and disseminates research findings to increase knowledge and understanding about health and disease among patients, health professionals, and the public. Content produced by the NIDDK is carefully reviewed by NIDDK scientists and other experts.
        </p> 
        </div>
        """
    
    cols[1].write(html_tp,unsafe_allow_html= True)
    st.write(html_tp3,unsafe_allow_html= True)
    cols = st.columns((3,1,3,1,3,1,3,1))
    cols[0].image('.../images$video/03.jpg')
    cols[2].image('.../images$video/04.jpg')
    cols[4].image('.../images$video/05.jpg')
    cols[6].image('.../images$video/06.jpg')


#page 2
if opt1 == 'Visualization':
    options = st.sidebar.radio('Select', ['Histogram','Line Chart','Scatter Plot','Heat Map'])
    st.sidebar.markdown('Data visualization helps to tell stories by curating data into a format that''s easier to understand, highlighting the trends and outliers.')
    if options == 'Histogram':
        html_tp ="""
        <div style ="background-color:purple;padding:13px">
        <h1 style ="color:white;text-align:center;font-family:montserrat">Histogram</h1>
        </div>
        """
        st.markdown(html_tp,unsafe_allow_html= True)
        st.subheader('Select a feature')
        features = st.selectbox('Feature', ['Select','Glucose_level','Diastolic_BloodPressure','SkinThickness','Insulin_level','BMI','DiabetesPedigreeFunction','Age','Pregnancies'])
        
        
        if features == 'Select':
            html_tp ="""
            <div
            <p style ="color:black;text-align:center;font-size:20px;font-family:montserrat">The histogram is a popular graphing tool. It is used to summarize discrete or continuous data that are measured on an interval scale. It is often used to illustrate the major features of the distribution of the data in a convenient form.</p>
            <p style ="color:black;text-align:center;font-size:20px;font-family:montserrat">It is a bar graph-like representation of data that buckets a range of classes into columns along the horizontal x-axis. The vertical y-axis represents the number count or percentage of occurrences in the data for each column. Columns can be used to visualize patterns of data distributions.</p>
            </div>
            """
            st.markdown(html_tp,unsafe_allow_html= True)
        else :
            to , from1 = to_from()
            if to == 0 :
                get_hist(features, to, from1)
                        
            else :
                get_hist(features, to, from1)
               
    if options == 'Line Chart':
        html_tp1 ="""
        <div style ="background-color:blue;padding:13px">
        <h1 style ="color:white;text-align:center;font-family:montserrat">Line Chart Plot</h1>
        </div>
        """
        st.markdown(html_tp1,unsafe_allow_html= True)
        
        to , from1 = to_from()
                    
        if to == 0:    
            st._legacy_line_chart(diabetes_df)
        else :
            dfa = diabetes_df.loc[from1:to]
            st.line_chart(dfa)
    
                    
        
    if options == 'Scatter Plot':
        html_tp2 ="""
        <div style ="background-color:green;padding:13px">
        <h1 style ="color:white;text-align:center;font-family:montserrat">Scatter Plot</h1>
        </div>
        """
        st.markdown(html_tp2,unsafe_allow_html= True)
        cols = st.columns((1,1))
        
        features_1 = cols[0].selectbox('Feature1', ['Select','Glucose_level','Diastolic_BloodPressure','SkinThickness','Insulin_level','BMI','DiabetesPedigreeFunction','Age','Pregnancies'])
        features_2 = cols[1].selectbox('Feature2', ['Select','Glucose_level','Diastolic_BloodPressure','SkinThickness','Insulin_level','BMI','DiabetesPedigreeFunction','Age','Pregnancies'])
        if features_1 == 'Select' or features_2 == 'Select' or features_1 == features_2 :
            html_tp ="""
            <h4 style ="color:black;font-size:18px;">please select two distinct features to cross check with outcome
            </h4>
            </div>
            """
            st.markdown(html_tp,unsafe_allow_html= True)
            html_tp ="""
            <div
            <p style ="color:black;text-align:center;font-size:20px;font-family:montserrat">Scatter plots are used to plot data points on a horizontal and a vertical axis in the attempt to show how much one variable is affected by another. Each row in the data table is represented by a marker whose position depends on its values in the columns set on the X and Y axes.</p>
            <p style ="color:black;text-align:center;font-size:20px;font-family:montserrat"> It represents data points on a two-dimensional plane or on a Cartesian system. The independent variable or attribute is plotted on the X-axis, while the dependent variable is plotted on the Y-axis.It is cross checkd with the outcome to find the stage at the data point.</p>
            </div>
            """
            st.markdown(html_tp,unsafe_allow_html= True)
        else :
            to , from1 = to_from()
            get_scat(features_1 , features_2 , to, from1)
        
    if options == 'Heat Map':
        html_tp2 ="""
        <div style ="background-color:silver;padding:13px">
        <h1 style ="color:white;text-align:center;font-family:montserrat">Heat Map</h1>
        </div>
        """
        st.markdown(html_tp2,unsafe_allow_html= True)
        html_tp ="""
            <div
            <p style ="color:black;text-align:left;font-size:18px;font-family:montserrat">Correlation heatmaps are a type of plot that visualize the strength of relationships between numerical variables. Correlation plots are used to understand which variables are related to each other and the strength of this relationship.</p>
            </div>
            """
        st.markdown(html_tp,unsafe_allow_html= True)
        cols1 = st.columns((1,7,1))
        fig, ax = plt.subplots()
        sb.heatmap(diabetes_df.corr(), ax=ax)
        with st.spinner('Wait for it...'):
            time.sleep(3)
        cols1[1].write(fig)
       
        html_tp5 ="""
            <div
            <p style ="color:black;text-align:left;font-size:18px;font-family:montserrat">A heat map is a data visualization technique that shows magnitude of a phenomenon as color in two dimensions. The variation in color may be by hue or intensity, giving obvious visual cues to the reader about how the phenomenon is clustered or varies over space.
            </p>
            </div>
            """
        st.markdown(html_tp5,unsafe_allow_html= True)


#page 3
if opt1 == 'Prediction':
    st.title('Predict diabetics')
    form = st.form(key="annotation")
    with form:
        cols1 = st.columns((2, 1))
        input_1 = cols1[0].number_input("Pregnancy",0,100)
        input_2 = cols1[0].number_input('Glucose level',0,1000)
        input_3 = cols1[0].number_input('DiastolicBlood Pressure',0,1000)
        input_4 = cols1[0].number_input('Skin Thickness',0,100)
        input_5 = cols1[0].number_input('Insulin level',0,200)
        input_6 = cols1[0].number_input('BMI')
        input_7 = cols1[0].number_input('Diabetic Pedegree Function')
        input_8 = cols1[0].number_input('Age',0,200)
        
        if input_2 == 0 and input_3 == 0 and input_4 == 0 and input_5 == 0 and input_6 == 0 and input_8 == 0:
            
            if st.form_submit_button('Predict') == 1:
                st.error('Please enter valid information')
        else :
            input_data = [input_1,input_2,input_3,input_4,input_5,input_6,input_7,input_8]
        
            if st.form_submit_button('Predict') == 1:
                prediction = load_model.predict([input_data])
                with st.spinner('Wait for it...'):
                    time.sleep(2)
        
                    if prediction == 0:
                        st.success("You have tested Negative !")
                        st.balloons()
                    else:
                        st.warning("You have tested positive")
                        st.info("Please contact a Doctor")
                    html_tp =""" 
                    <meta charset="UTF-8">
                    <p style ="color:black;font-size:18px;font-family: 'Brush Script MT', cursive;">Thank you for using our app. Hope its helpfull &#128512;
                    </p> 
                    </div>
                    """
                    st.write(html_tp,unsafe_allow_html= True)
#page 4
if opt1 == 'Feedback':
        st.write('Contact us') 
        st.text_area("Tell us how we can help:")
        st.write("We will respond to you via mail or phone")
        cols = st.columns((1, 1))
        mailr = cols[0].text_input("Email:")
        phone = cols[1].text_input("phone:")
        st.write('<img width=50 src="https://emojipedia-us.s3.amazonaws.com/source/skype/289/squid_1f991.png" style="margin-left: 5px; filter: hue-rotate(230deg) brightness(1.1);">',
        unsafe_allow_html=True,)
