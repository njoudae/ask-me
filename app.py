import streamlit as st
import fitz
import pytesseract
from PIL import Image
import os
from dotenv import load_dotenv
#from langchain.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI


load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")


# إعداد صفحة Streamlit
st.set_page_config(page_title="اسألني", layout="wide")
st.title("📄 تفاعل مع ملفك باللغة العربية")
st.markdown(
    """
    <style>
    body {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📎 ارفع الملف", type="pdf")


def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        page_text = page.get_text()
        if page_text.strip():
            text += page_text
        else:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img, lang='ara')
    return text

# عند رفع الملف
if uploaded_file:
    raw_text = extract_text(uploaded_file)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    # استخدام المفتاح من البيئة
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectordb = FAISS.from_documents(docs, embeddings)

    # خانة السؤال
    question = st.text_input("🗨️ اكتب سؤالك هنا:")

    if question:
        docs_similar = vectordb.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs_similar])

        prompt = f"""
        اقرأ النص التالي وأجب على السؤال بدقة باللغة العربية:

        السياق:
        {context}

        السؤال:
        {question}

        الإجابة:
        """

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=openai_key
        )

        response = llm.predict(prompt)

        st.markdown("### ✅ الإجابة:")
        st.write(response)
