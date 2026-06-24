import streamlit as st
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import OpenAI
from dotenv import load_dotenv
import os


@st.cache_resource
def loadModel():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = loadModel()


def connectDB():

    import os
    import psycopg2

    DATABASE_URL = os.getenv("DATABASE_URL")

    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn

def chunkText(text):

    chunks = []
    size = 500

    for i in range(0, len(text), size):
        chunks.append(text[i:i+size])

    return chunks


def createEmbedding(data):
    return model.encode(data)


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


def initModel():

    load_dotenv()

    api_key = os.getenv("API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    return client

client = initModel()


def generatePrompt(context, question):

    return f"""
Answer the question only based on the given context.


Context:
{context}

Question:
{question}
"""


if "processed" not in st.session_state:
    st.session_state.processed = False


st.title("VISUALRAG")

uploadedFiles = st.file_uploader(
    "Upload Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)


if st.button("Upload"):

    if uploadedFiles:

        try:

            all_text = ""

            for uploadedFile in uploadedFiles:

                image = Image.open(uploadedFile)

                text = pytesseract.image_to_string(image)

                all_text += text + "\n"

            chunks = chunkText(all_text)

            vectors = createEmbedding(chunks)

            replaceDocument(chunks, vectors)

            st.session_state.processed = True

            st.success(
                f"{len(uploadedFiles)} Images Processed Successfully"
            )

        except Exception as e:

            st.error(e)

    else:

        st.error("Upload Images")


if st.session_state.processed:

    question = st.text_input(
        "Ask Question About Uploaded Images"
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
                    model="openai/gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                st.subheader("Answer")

                st.success(
                    response.choices[0]
                    .message.content
                )

            except Exception as e:

                st.error(e)

        else:

            st.error("Enter Question")