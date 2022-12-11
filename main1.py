import streamlit as st

import warnings
warnings.filterwarnings("ignore")

from io import StringIO, BytesIO
import urllib.request
import joblib
import time
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import os,sys
from scipy import stats

import pickle

from sklearn import metrics

st.set_page_config(
    page_title="20-029_Data Mining"
)



tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Preprocessing", "Modelling", "Implementation"])


with tab1:
    st.write("Nama : Alvina Maharani")
    st.write("NIM     :200411100029")
    st.write("Kelas : Penambangan Data A")
    st.title('Klasifikasi penyakit Stroke')
    st.write("""
    
    Stroke adalah kondisi yang terjadi ketika pasokan darah ke otak mengalami gangguan atau berkurang akibat penyumbatan (stroke iskemik) atau pecahnya pembuluh darah (stroke hemoragik).
    """)
    st.write("Stroke merupakan kondisi gawat darurat yang perlu ditangani secepatnya, karena sel otak dapat mati hanya dalam hitungan menit. Tindakan penanganan yang cepat dan tepat dapat meminimalkan tingkat kerusakan otak dan mencegah kemungkinan munculnya komplikasi.")

    st.markdown("""
    Link Dataset
    <a href="https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset"> Klik Disini</a>
    """, unsafe_allow_html=True)


    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    st.write("Dataset Penyakit Stroke : ")
    st.write(df)
    st.write("Note Nama - Nama Kolom : ")

    st.write("""
    <ol>
    <li>gender : Jenis Kelamin (1=Laki-laki, 0=Perempuan)</li>
    <li>age : Usia (di sesuikan langsung dengan angka tipe float)</li>
    <li>hypertention : Tekanan darah tinggi ( 0 = tidak mengalami hipertensi, 1 = mengalami hipertensi))</li>
    <li>heart_disease : sakit jantung ( 0 = tidak mengalami sakit jantung, 1 = mengalami sakit jantung)</li>
    <li>ever_married : kondisi pernah menikah. (0 = belum menikah, 1 = sudah menikah) </li>
    <li>work_type : jenis pekerjaan keseharian apakah wiraswasta atau privat(pribadi). (0 = privat, 1 = wiraswasta) </li>
    <li>Residence_type : tempat tinggal. urban artinya perkotaan, rural artinya pedesaan (0 = urban, 1 = rural) </li>
    <li>avg_glucose_level : rata-rata gula darah yang dimiliki</li>
    <li>bmi : perkiraan lemak tubuh </li>
    <li>smoking_status : status apakah pernah merokok atau tidak. formerly smoked artinya pernah merokok, never smoked artinya tidak pernah merokok, smokes artinya merokok.</li>
    </ol>
    """,unsafe_allow_html=True)

with tab2:
    st.write("Data preprocessing adalah proses yang mengubah data mentah ke dalam bentuk yang lebih mudah dipahami. Proses ini penting dilakukan karena data mentah sering kali tidak memiliki format yang teratur. Selain itu, data mining juga tidak dapat memproses data mentah, sehingga proses ini sangat penting dilakukan untuk mempermudah proses berikutnya, yakni analisis data.")
    st.write("Data preprocessing adalah proses yang penting dilakukan guna mempermudah proses analisis data. Proses ini dapat menyeleksi data dari berbagai sumber dan menyeragamkan formatnya ke dalam satu set data.")
    
    # intial template
    px.defaults.template = "plotly_dark"
    px.defaults.color_continuous_scale = "reds"

    st.header("Import Data")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    
    for uploaded_file in uploaded_files:
        # uplod file
        data = pd.read_csv(uploaded_file)
        st.write(" **Nama File Anda :** ", uploaded_file.name)

        # view dataset asli
        st.header("Dataset Asli")
        X = data.drop(columns=["stroke"])
        st.dataframe(X)
        row, col = data.shape 
        st.caption(f"({row} rows, {col} cols)")

        # view dataset NORMALISASI
        st.header("Dataset Normalisasi dan Transformasi")
        #  Tahap Normalisasi data string ke kategori
        X = pd.DataFrame(X)
        X['gender'] = X['gender'].astype('category')
        X['ever_married'] = X['ever_married'].astype('category')
        X['work_type'] = X['work_type'].astype('category')
        X['Residence_type'] = X['Residence_type'].astype('category')
        X["bmi"] = X["bmi"].astype('category')
        X["smoking_status"] = X["smoking_status"].astype('category')
        cat_columns = X.select_dtypes(['category']).columns
        X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)

        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaled = scaler.fit_transform(X)
        features_names = X.columns.copy()
        #features_names.remove('label')
        scaled_features = pd.DataFrame(scaled, columns=features_names)

        st.dataframe(scaled_features)
        row, col = data.shape
        st.caption(f"({row} rows, {col} cols")

        import joblib
        filename = "noemalisasi_stroke.sav"
        joblib.dump(scaler, filename)

        y = df['stroke'].values

with tab3:
    st.write("""
    <h5>Modelling</h5>
    <br>
    """, unsafe_allow_html=True)


    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    y_baru = le.fit_transform(y)
    y_baru

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test=train_test_split(X, y_baru, test_size=0.2, random_state=1)

    #inisialisasi KNN
    my_param_grid = {'n_neighbors':[2,3,5,7], 'weights': ['distance', 'uniform']}
    GridSearchCV(estimator=KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
    knn = GridSearchCV(KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
    knn.fit(X_train, y_train)

    pred_test = knn.predict(X_test)

    vknn = f'Hasil akurasi dari pemodelan K-Nearest Neighbour = {accuracy_score(y_test, pred_test) * 100 :.2f} %'
    vknn

    filenameModelKnnNorm = 'modelKnnNorm.pkl'
    joblib.dump(knn, filenameModelKnnNorm)


    #inisialisasi model gausian
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    filenameModelGau = 'modelGau.pkl'
    joblib.dump(gnb, filenameModelGau)

    y_pred = gnb.predict(X_test)

    vgnb = f'Hasil akurasi dari pemodelan Gausian = {accuracy_score(y_test, y_pred) * 100 :.2f} %'
    vgnb

    #inisialisasi model Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    d3 = DecisionTreeClassifier()
    d3.fit(X_train, y_train)

    filenameModeld3 = 'modeld3.pkl'
    joblib.dump(d3, filenameModeld3)

    y_pred = d3.predict(X_test)

    vd3 = f'Hasil akurasi dari pemodelan decision tree : {accuracy_score(y_test, y_pred) * 100 :.2f} % '
    vd3

    K_Nearest_Naighbour, Gausian, Decision_Tree = st.tabs(["K-Nearest aighbour", "Naive Bayes Gausian", "Decision Tree"])
    with K_Nearest_Naighbour:
        st.header("K-Nearest Neighbour")
        st.write("Algoritma KNN mengasumsikan bahwa sesuatu yang mirip akan ada dalam jarak yang berdekatan atau bertetangga. Artinya data-data yang cenderung serupa akan dekat satu sama lain. KNN menggunakan semua data yang tersedia dan mengklasifikasikan data atau kasus baru berdasarkan ukuran kesamaan atau fungsi jarak. Data baru kemudian ditugaskan ke kelas tempat sebagian besar data tetangga berada.")
        st.header("Pengkodean")

        st.text("""
        my_param_grid = {'n_neighbors':[2,3,5,7], 'weights': ['distance', 'uniform']}
        GridSearchCV(estimator=KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
        knn = GridSearchCV(KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
        knn.fit(X_train, y_train)

        pred_test = knn.predict(X_test)

        vknn = f'Hasil akurasi dari pemodelan K-Nearest Neighbour : {accuracy_score(y_test, pred_test) * 100 :.2f} %'
        """)

        st.header("Hasil Akurasi")
        st.write(vknn)
    

    with Gausian:
        st.header("Naive Bayes Gausian")
        st.write("Metode yang juga dikenal sebagai Naive Bayes Classifier ini menerapkan teknik supervised klasifikasi objek di masa depan dengan menetapkan label kelas ke instance/catatan menggunakan probabilitas bersyarat. \nProbabilitas bersyarat adalah ukuran peluang suatu peristiwa yang terjadi berdasarkan peristiwa lain yang telah (dengan asumsi, praduga, pernyataan, atau terbukti) terjadi \nRumus: P(A│B) = P(B│A)P(A)P(B). Adapun salah satu jenis naive bayes adalah gausian. Distribusi Gaussian adalah asumsi pendistribusian nilai kontinu yang terkait dengan setiap fitur berisi nilai numerik. Ketika diplot, akan muncul kurva berbentuk lonceng yang simetris tentang rata-rata nilai fitur.")
        st.header("Pengkodean")
        st.text("""
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)

        filenameModelGau = 'modelGau.pkl'
        joblib.dump(gnb, filenameModelGau)

        y_pred = gnb.predict(X_test)

        vgnb = f'Hasil akurasi dari pemodelan Gausian : {accuracy_score(y_test, y_pred) * 100 :.2f} %'
        """)
        st.header("Hasil Akurasi")
        st.write(vgnb)   
    
    with Decision_Tree:
        st.header("Decision Tree")
        st.write("Konsep Decision tree  adalah dengan cara menyajikan algoritma dengan pernyataan bersyarat, \nyang meliputi cabang untuk mewakili langkah-langkah pengambilan keputusan yang dapat mengarah pada hasil yang menguntungkan.")
        st.header("Pengkodean")
        st.text(""" 
        from sklearn.tree import DecisionTreeClassifier
        d3 = DecisionTreeClassifier()
        d3.fit(X_train, y_train)

        filenameModeld3 = 'modeld3.pkl'
        joblib.dump(d3, filenameModeld3)

        y_pred = d3.predict(X_test)

        vd3 = f'Hasil akurasi dari pemodelan decision tree : {accuracy_score(y_test, y_pred) * 100 :.2f} %'
        """)
        st.header("Hasil Akurasi")
        st.write(vd3) 


with tab4:
    st.write("""
    <h5>Implementation</h5>
    <br>
    """, unsafe_allow_html=True)
    st.title("Implementasi Model")
    st.write("Sebagai bahan eksperimen silahkan inputkan beberapa data yang akan digunakan sebagai data testing untuk pengklasifikasian")

    st.header("Input Data Testing")
    
    #create input
    id = st.number_input("id")
    gender = st.number_input("gender")
    age    = st.number_input("age")
    hypertension = st.number_input("hypertension")
    heart_disease = st.number_input("heart_disease")
    ever_married = st.number_input("ever_married")
    work_type = st.number_input("work_type")
    Residence_type = st.number_input("Residence_type")
    avg_glucose_level = st.number_input("avg_glucose_level")
    bmi = st.number_input("bmi")
    smoking_status = st.number_input("smoking_status")

    def submit():
        #olahan inputan 
        a = np.array([[id, gender, age, hypertension, heart_disease, ever_married,	work_type,	Residence_type, avg_glucose_level, bmi,	smoking_status ]])
        test_data = np.array(a).reshape(1, -1)
        test_data = pd.DataFrame(test_data, columns =['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',	'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status' ])

        #  Tahap Normalisasi data sting ke kategori
        test_data = pd.DataFrame(test_data)
        test_data['gender'] = test_data['gender'].astype('category')
        test_data['ever_married'] = test_data['ever_married'].astype('category')
        test_data['work_type'] = test_data['work_type'].astype('category')
        test_data['Residence_type'] = test_data['Residence_type'].astype('category')
        test_data["bmi"] = test_data["bmi"].astype('category')
        test_data["smoking_status"] = test_data["smoking_status"].astype('category')
        cat_columns = test_data.select_dtypes(['category']).columns
        test_data[cat_columns] = test_data[cat_columns].apply(lambda x: x.cat.codes)
    
        scaler = joblib.load(filename)
        test_d = scaler.fit_transform(test_data)
        # pd.DataFrame(test_d)

        # load knn
        knn = joblib.load(filenameModelKnnNorm)
        pred = knn.predict(test_d)

        # load gausian
        gnb = joblib.load(filenameModelGau)
        pred = gnb.predict(test_d)

        # load gdecision tree
        d3 = joblib.load(filenameModeld3)
        pred = d3.predict(test_d)

        # button
        st.header("Data Input")
        st.write("Berikut ini tabel hasil input data testing yang akan diklasifikasi:")
        st.dataframe(a)

        st.header("Hasil Prediksi")
        K_Nearest_Naighbour, Naive_Bayes, Decision_Tree = st.tabs(["K-Nearest aighbour", "Naive Bayes Gausian", "Decision Tree"])

        with K_Nearest_Naighbour:
            st.subheader("Model K-Nearest Neighbour")
            pred = knn.predict(test_d)
            if pred[0]== 0:
                st.write("Hasil Klasifikaisi : Tidak Stroke")
            elif pred[0]== 1 :
                st.write("Hasil Klasifikaisi : Stroke")
            else:
                st.write("Hasil Klasifikaisi : New Category")
    
        with Naive_Bayes:
            st.subheader("Model Naive Bayes Gausian")
            pred = gnb.predict(test_d)
            if pred[0]== 0:
                st.write("Hasil Klasifikaisi : Tidak Stroke")
            elif pred[0]== 1 :
                st.write("Hasil Klasifikaisi : Stroke")
            else:
                st.write("Hasil Klasifikaisi : New Category")


        with Decision_Tree:
            st.subheader("Model Decision Tree")
            pred = d3.predict(test_d)
            if pred[0]== 0:
                st.write("Hasil Klasifikaisi : Tidak Stroke")
            elif pred[0]== 1 :
                st.write("Hasil Klasifikaisi : Stroke")
            else:
                st.write("Hasil Klasifikaisi : New Category")
                
    submitted = st.button("Submit")
    if submitted:
        submit()
    



   