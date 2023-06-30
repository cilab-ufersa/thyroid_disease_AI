import streamlit as st

st.set_page_config(
            page_title="Hipotireoidismo",
            page_icon='app/icon/cilab.png',
            layout="wide",
            )

st.markdown('<style>h1{font-size: 33px;}</style>', unsafe_allow_html=True)
st.title("Classificação de doenças da tireoide com técnicas de aprendizado de máquina")
st.markdown('---')
st.image('app/icon/icon.png')
st.write("O hipotireoidismo é uma condição em que a glândula tireoide não produz hormônio tireoidiano suficiente, o que pode resultar em uma diminuição do metabolismo e uma série de sintomas associados. A glândula tireoide está localizada na base do pescoço e é responsável por produzir hormônios que regulam o metabolismo do corpo.")

st.markdown('<style>h3{font-size: 15px;}</style>', unsafe_allow_html=True)
st.subheader("Apoio")
st.image('app/icon/Ufersa.png', caption='UFERSA', width=70)