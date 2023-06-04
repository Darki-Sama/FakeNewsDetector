import streamlit as st
from PIL import Image
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

image = Image.open('IsFake.png')
imageT = Image.open('true.png')
imageF = Image.open('fake.png')

vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con

def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction






if __name__ == '__main__':
    st.image(image)
    st.title('Detector de Fake News ')
    st.subheader("Introduce la noticia que deseas revisar")
    sentence = st.text_area("Introduce tu noticia aqui", "",height=200)
    predict_btt = st.button("Revisar")
    if predict_btt:
        prediction_class=fake_news(sentence)
        print(prediction_class)
        if prediction_class == [0]:
            st.success('La noticia es confiable')
            st.image(imageT)
            
        if prediction_class == [1]:
            st.warning('La noticia no es confiable')
            st.image(imageF)
            st.warning('Recomendamos que corrobores la informacion con tus sitios de confianza')