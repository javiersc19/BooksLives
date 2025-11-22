# Aquí vamos a definir las funciones que vamos a utilizar en la arquitectura de nuestra BooksLives

# Librerias a utilizar
from transformers import pipeline
import scipy
import numpy as np 
from openai import OpenAI
import json
import re
from io import BytesIO
import os
import requests
from typing import List, Optional
from pydantic import BaseModel, Field
import replicate

#------------------------------------------------------ Conexión a APIS ----------------------------------------------------------#
# Variables de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

#Conexion ChatGPT

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

# Conexión a Replicate
client_replicate = replicate.Client(REPLICATE_API_TOKEN)
#------------------------------------------------------- Ingeniería de Prompt ---------------------------------------------------#

# Prompt para la generación de instrucciones de música (MusicGen)
prompt_instruccion_genera_musica = '''You are a specialized AI music curator for the MusicGen model. Your task is to analyze a 
provided text passage, focusing on its emotional atmosphere, implied tempo, genre, setting, and overall mood. Your output MUST be 
a single, short (max 15 words) English-language prompt, optimized for MusicGen, that captures the essence of the text. Do not 
include any explanation, introductory phrases, or extra characters, only the prompt itself. Be highly descriptive with musical 
terms (e.g., 'ambient drone', 'cinematic brass').'''

# Prompt para la generación de instrucciones de imagenes (Replicate)
prompt_text_imagen='''Quiero que actúes como un generador de prompts visuales para modelos de imagen (como Flux, Stable Diffusion o SD3). Recibirás un fragmento de texto de un libro y tu tarea será transformarlo en un prompt claro, visual, cinematográfico y listo para usar en un modelo de generación de imágenes.

Sigue estas reglas:
1. No resumas el texto: conviértelo en una escena visual.
2. Describe:
   - ambiente y escenario
   - época o estilo (si aplica)
   - iluminación
   - emociones o atmósfera
   - detalles visuales clave (ropa, objetos, colores, gestos)
3. Mantén el prompt entre 2 y 4 líneas máximo.
4. Evita elementos que no aparezcan o no puedan inferirse del fragmento.
5. No hagas una interpretación literaria: conviértelo en una IMAGEN.
6. NO incluyas texto literal del libro en la imagen.
7. Usa un estilo descriptivo pero natural.
'''
#------------------------------------------------------ Definición de funciones------------------------------------------------#

# Función que genera el prompt para generar la música
def text_to_music(texto):   
    respuesta = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": prompt_instruccion_genera_musica},
            {"role": "user", "content": texto}
        ]
    )
    return respuesta.choices[0].message.content.strip()

# Función genera música
def musicgen_generation(prompt_from_text):
    # Carga del modelo small para mayor velocidad
    synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")

    music = synthesiser(
        prompt_from_text,
        forward_params={
            "do_sample": True,
            "max_new_tokens": 512
        }
    )
    
    # Escalar y convertir a int16 para evitar distorsión de volumen
    audio_data_float = music["audio"]
    audio_data_int = (audio_data_float * 32767).astype(np.int16)

    audio_bytes = BytesIO()
    scipy.io.wavfile.write(audio_bytes, music["sampling_rate"], audio_data_int)

    return audio_bytes.getvalue()

# Función que limpia texto
def clean_text(texto):
    patron = r"www\.lectulandia\.com\s-\sPágina\s\d+"
    texto_limpio=re.sub(patron, "", texto)
    texto_limpio= re.sub(r"\n\s*\n", "\n", texto_limpio).strip()
    return texto_limpio

# Función Analisis de emociones y topicos texto
def vector_sentimientos(texto):
    prompt = f"""
    Analiza el siguiente TEXTO y devuelve un objeto JSON con puntajes numéricos entre 0.0 y 1.0 (inclusive) para cada emoción. Los puntajes deben sumar aproximadamente 1.0 (±0.05). Redondea a tres decimales y muestralos en formato de porcentaje.
    
    Esquema requerido:
      "alegria": float %,
      "tristeza": float %,
      "miedo": float %,
      "enojo": float %,
      "sorpresa": float %,
      "neutralidad": float %
      
    Texto:
    "{texto}"
    """
    respuesta = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": "Eres un analizador de emociones preciso y estricto. Responde únicamente con JSON válido que siga exactamente el esquema indicado. No agregues explicaciones ni texto fuera del JSON."},
                  {"role": "user", "content": prompt}],
    )
    
    return json.loads(respuesta.choices[0].message.content)


#------------------------------------------------------ Definición de funciones nuevas-----------------------------------------#

# Función que me devuelve un Structured Output para generación de insights del texto
class BookInsights(BaseModel):
    """
    Output estructurado con insights clave en español extraídos de un fragmento de libro.
    """
    Título: str = Field(description="Título del libro al que pertenece el fragmento de texto, si no se reconoce colocar Título Desconocido")
    Autor: List[str] = Field(default_factory=list, description="Autor o autores del libro si es que se puede saber, sino colocar Autor Desconocido")
    Sentimiento: Optional[str] = Field(description="Sentimiento general: Alegría / Tristeza / Miedo / Enojo / Sorpresa")
    Resumen: str = Field(description="Idea principal del texto en español (párrafo de 3 líneas)")
    Tópicos: List[str] = Field(default_factory=list, description="Tópicos principales del texto, cado uno con un máximo de 3 palabras, en español")
    Obras_relacionadas: List[str] = Field(default_factory=list, description="Libros relacionados con el libro al que pertenece el texto, con un máximo de 3 libros, en español")
    Premios: List[str] = Field(default_factory=list, description="Premios que ha ganado el libro al que pertenece el texto, con un máximo de 3 libros, en español, sino colocar Desconocido.")
    Personajes_principales: List[str] = Field(default_factory=list, description="Nombres de los personajes principales mencionados en el texto.")
    Lugar_hechos: Optional[str] = Field(description="Descripción del lugar o escenario principal donde ocurre la acción.")
    Época: Optional[str] = Field(description="El período de tiempo o época en que se desarrolla la narrativa.")
    Temas_adicionales: List[str] = Field(default_factory=list, description="Otros temas importantes explorados en el texto, con un máximo de 3 palabras cada uno.")
    Narrativa: Optional[str] = Field(description="El tono general de la narración: Melancólico / Misterioso / Nostálgico / introspectivo).")

def get_book_insights(client, book_text: str, model_name: str = "gpt-5-mini") -> BookInsights:
    """
    Obtiene insights clave de un fragmento de texto de un libro utilizando un modelo de OpenAI.

    Args:
        client: Cliente de OpenAI inicializado.
        book_text: El texto del fragmento del libro.
        model_name: El nombre del modelo de OpenAI a utilizar.

    Returns:
        Un objeto diccionario derivado de BookInsights con los insights extraídos.
    """
    response = client.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "system", "content": "Eres un experto Analista de libros y novelas. Devuelve SOLO un JSON válido que siga exactamente el esquema de BookInsights. Salidas en español."},
            {"role": "user", "content": book_text},
        ],
        response_format=BookInsights,
    )
    insights = response.choices[0].message.parsed
    return insights.model_dump()


# Función que genera el prompt para generar la imagen
def text_to_imagen(texto):
    respuesta = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": prompt_text_imagen},
            {"role": "user", "content": texto}
        ]
    )
    return respuesta.choices[0].message.content.strip()

# Función que genera la imagen respectivo al prompt extraido
def crea_imagen(prompt_imagen):
    output = client_replicate.run(
        "black-forest-labs/flux-1.1-pro",
        input={"prompt": prompt_imagen}
      )
    return output   # URL de la imagen