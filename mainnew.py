# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st
import shap

st.header("Development and validation of an AI model for estimating early death among lung cancer patients with bone metastases")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")

age = st.sidebar.slider("Age", 30, 100)
primarysite = st.sidebar.selectbox("Primary site", ("Main bronchus", "Upper lobe", "Middle lobe", "Lower lobe", "Overlapping lesion", "Lung, NOS"))
Histology = st.sidebar.selectbox("Histology", ("Unspecified neoplasms", "Epithelial neoplasms, NOS", "Squamous cell neoplasms", "Adenomas and adenocarcinomas", "Others"))
race = st.sidebar.selectbox("Race", ("Black", "Others", "Unknown", "White"))
Sex = st.sidebar.selectbox("Sex", ("Female", "Male"))
tstage= st.sidebar.selectbox("T stage", ("T0", "T1", "T2", "T3", "T4", "Tx"))
nstage= st.sidebar.selectbox("N stage", ("N0", "N1", "N2", "N3", "Nx"))
brainm = st.sidebar.selectbox("Brain metastasis", ("No", "Unknown", "Yes"))
liverm = st.sidebar.selectbox("Liver metastasis", ("No", "Unknown", "Yes"))
surgery = st.sidebar.selectbox("Cancer directed surgery", ("No", "Unknown", "Yes"))
Radiation = st.sidebar.selectbox("Radiation", ("No/Unknown", "Yes"))
Chemotherapy = st.sidebar.selectbox("Chemotherapy", ("No/Unknown", "Yes"))

if st.button("Submit"):
    rf_clf = jl.load("Xgbc_clf_final_round-web.pkl")
    x = pd.DataFrame([[primarysite, Histology, race, Sex, tstage, nstage, brainm, liverm, surgery, Radiation, Chemotherapy, age]],
                     columns=['primarysite', 'Histology', 'race', 'Sex', 'tstage', 'nstage', 'brainm', 'liverm', 'surgery', 'Radiation', 'Chemotherapy', 'age'])
    x = x.replace(["Main bronchus", "Upper lobe", "Middle lobe", "Lower lobe", "Overlapping lesion", "Lung, NOS"], [1, 2, 3, 4, 5, 6])
    x = x.replace(["Unspecified neoplasms", "Epithelial neoplasms, NOS", "Squamous cell neoplasms", "Adenomas and adenocarcinomas", "Others"], [1, 2, 3, 4, 5])
    x = x.replace(["Female", "Male"], [1, 2])
    x = x.replace(["Black", "Others", "Unknown", "White"], [1, 2, 3, 4])
    x = x.replace(["T0", "T1", "T2", "T3", "T4", "Tx"], [1, 2, 3, 4, 5, 6])
    x = x.replace(["N0", "N1", "N2", "N3", "Nx"], [1, 2, 3, 4, 5])
    x = x.replace(["No", "Unknown", "Yes"], [1, 2, 3])
    x = x.replace(["No/Unknown", "Yes"], [1, 2])


    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.text(f"Probability of experiencing early death: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.37:
        st.text(f"Risk group: low-risk group")
    else:
        st.text(f"Risk group: High-risk group")
    if prediction < 0.37:
        st.markdown(f"Recommendations: For patients in the low-risk groups, invasive surgery, such as excisional surgery, and long-course radiotherapy were recommended, because those patients might suffer from poor quality of life for a very long time, if only palliative interventions were performed.")
    else:
        st.markdown(f"Recommendations: Patients in the high-risk groups were 4.5-fold chances to suffer from early death than patients in the low-risk groups (P<0.001). Open surgery was not recommended to those patients. They should better be treated with radiotherapy alone, best supportive care, or minimal invasive techniques such as cementoplasty to palliatively alleviate pain.")

    st.subheader('Model explanation: contribution of each model predictor')
    star = pd.read_csv('X_train.csv', low_memory=False)
    y_trainy = pd.read_csv('y_train.csv', low_memory=False)
    data_train_X = star.loc[:, ['age', 'primarysite', 'Histology', 'race', 'Sex', 'tstage', 'nstage', 'brainm', 'liverm', 'surgery', 'Radiation', 'Chemotherapy']]
    y_train = y_trainy.mortality
    model = rf_clf.fit(data_train_X, y_train)
    explainer = shap.Explainer(model)
    shap_value = explainer(x)
    #st.text(shap_value)

    shap.initjs()
    #image = shap.plots.force(shap_value)
    #image = shap.plots.bar(shap_value)

    shap.plots.waterfall(shap_value[0])
    st.pyplot(bbox_inches='tight')
    st.set_option('deprecation.showPyplotGlobalUse', False)


st.subheader('About the model')
st.markdown('This online calculator is freely accessible, and it’s algorithm was based on the gradient boosting machine. Internal validation showed that the AUC of the model was 0.820; External validation also showed good results. However, this model was designed for research purpose, and clinical treatment for bone metastases should not rely the AI platform only')