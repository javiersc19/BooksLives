📚🎵 BooksLives — El soundtrack de tu lectura

App desarrollada en Python + Streamlit, desplegada en Hugging Face Spaces, que analiza texto de un PDF, detecta emociones y genera música y arte usando IA (Transformers MusicGen, Replicate, OpenAI).

🚀 ¿Qué hace esta app?

✔️ Sube un PDF ✔️ Extrae el texto ✔️ Analiza sentimientos del texto ✔️ Genera música basada en el estado emocional ✔️ Produce imágenes basadas en la lectura ✔️ Muestra insights del capítulo/libro

🛠️ Tecnologías usadas

Streamlit – interfaz

Transformers + MusicGen – generación musical

PyPDF2 + pdf2image – lectura de PDF

Replicate API – generación multimedia

OpenAI API – análisis de sentimientos, embeddings e insights

Pydantic – validación de datos

📁 Estructura del proyecto 📦 BooksLives ├── BooksLives.py # Tu app Streamlit principal ├── app.py # Archivo para Hugging Face (importa la app) ├── Tools_BooksLives.py # Utilidades IA (sentimientos, música, imágenes) ├── requirements.txt └── README.md

▶️ Cómo correr la app localmente pip install -r requirements.txt streamlit run BooksLives.py

🌐 Deploy en Hugging Face (ya configurado)

El archivo app.py contiene:

import BooksLives

Esto hace que Hugging Face abra automáticamente tu app.

🔑 Variables de entorno necesarias

Configura en Hugging Face → Settings → Variables:

OPENAI_API_KEY = "tu_api" REPLICATE_API_TOKEN = "tu_api"

Autor: Proyecto desarrollado por @javiersc19.