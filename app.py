import streamlit as st
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from dotenv import load_dotenv
import os

import pytesseract


# ---------------------------
# Load Embedding Model
# ---------------------------
@st.cache_resource
def loadModel():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = loadModel()


# ---------------------------
# OCR
# ---------------------------
def extractTextFromImage(uploadedFile):
    image = Image.open(uploadedFile)
    text = pytesseract.image_to_string(image)
    return text


# ---------------------------
# Chunk Text
# ---------------------------
def chunkText(text):
    chunks = []
    size = 500

    for i in range(0, len(text), size):
        chunks.append(text[i:i+size])

    return chunks


# ---------------------------
# Create Embedding
# ---------------------------
def createEmbedding(texts):
    return model.encode(texts)


def connectDB():
    conn = psycopg2.connect(
        st.secrets["DATABASE_URL"]
    )

    register_vector(conn)
    return conn

    register_vector(conn)

    return conn


# ---------------------------
# Store Chunks
# ---------------------------
def replaceDocument(chunks, vectors):

    conn = connectDB()
    cur = conn.cursor()

    try:

        cur.execute(
            "TRUNCATE TABLE documents RESTART IDENTITY"
        )

        for chunk, vector in zip(chunks, vectors):

            cur.execute(
                """
                INSERT INTO documents
                (chunk_text, embedding)
                VALUES (%s,%s)
                """,
                (chunk, vector)
            )

        conn.commit()

    finally:

        cur.close()
        conn.close()


# ---------------------------
# Retrieve Similar Chunks
# ---------------------------
def findRelatedVector(questionVector):

    conn = connectDB()
    cur = conn.cursor()

    try:

        cur.execute(
            """
            SELECT chunk_text
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT 3
            """,
            (questionVector,)
        )

        return cur.fetchall()

    finally:

        cur.close()
        conn.close()


# ---------------------------
# OpenRouter
# ---------------------------
def initModel():

    load_dotenv()

    api_key = os.getenv("API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    return client


client = initModel()


# ---------------------------
# Prompt
# ---------------------------
def generatePrompt(context, question):

    prompt = f"""
Answer the question only from the given context.

If answer is not available say:
"Answer not found"

Context:
{context}

Question:
{question}
"""

    return prompt


# ---------------------------
# Streamlit UI
# ---------------------------

if "processed" not in st.session_state:
    st.session_state.processed = False


st.title("VISUALRAG")


uploadedFile = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)


# ---------------------------
# Upload Button
# ---------------------------
if st.button("Upload"):

    if uploadedFile is not None:

        try:

            text = extractTextFromImage(uploadedFile)

            chunks = chunkText(text)

            vectors = createEmbedding(chunks)

            replaceDocument(chunks, vectors)

            st.session_state.processed = True

            st.success(
                "Image Processed Successfully"
            )

        except Exception as e:

            st.error(e)

    else:

        st.error("Upload an Image")


# ---------------------------
# Ask Question
# ---------------------------
if st.session_state.processed:

    question = st.text_input(
        "Ask Question About Image"
    )

    if st.button("Ask"):

        if question.strip():

            questionVector = createEmbedding(
                [question]
            )[0]

            result = findRelatedVector(
                questionVector
            )

            prompt = generatePrompt(
                result,
                question
            )

            try:

                response = client.chat.completions.create(
                    model="openrouter/free",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                st.success(
                    response.choices[0].message.content
                )

            except Exception as e:

                st.error(e)

        else:

            st.error("Enter Question")