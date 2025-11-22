
#Librerías
import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from io import BytesIO
from openai import OpenAI
import json
import hashlib
import os
from Tools_BooksLives import vector_sentimientos, text_to_music,musicgen_generation,clean_text,text_to_imagen,crea_imagen,get_book_insights
import re
import requests
from typing import List, Optional
from pydantic import BaseModel, Field
import replicate


#------------------------------------------------------ Configuración inicial ----------------------------------------------------------#

#Titulo
st.title("BooksLives")

# Variables de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

#Conexion ChatGPT

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

# Conexión a Replicate
client_replicate = replicate.Client(REPLICATE_API_TOKEN)

model_tts = 'gpt-4o-mini-tts'

#------------------- Configuración global de la página ----------------------------
st.set_page_config(page_title="BooksLives: El soudtrack de tu lectura", layout="wide", page_icon='📖')


st.markdown("""
    <style>
    /* 1. Eliminar espacio superior por defecto de Streamlit */
    .block-container {
        padding-top: 2rem; /* Reducido de 5rem (por defecto) a 2rem */
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    /* 2. Estilo compactado para los títulos */
    .header-title {
        text-align: center;
        font-size: 2em; /* Título más pequeño */
        color: #FF4B4B; 
        margin-top: 0px; /* Elimina margen superior del h1 */
        margin-bottom: 5px; /* Reduce el espacio debajo del h1 */
    }
    .header-subtitle {
        text-align: center;
        font-size: 1em; /* Subtítulo más discreto */
        color: #777777;
        margin-top: 0px; /* Elimina margen superior del h3 */
        margin-bottom: 20px; /* Espacio para el contenido principal */
    }
    </style>
    
    <h1 class="header-title">📖 BooksLives</h1>
    <h3 class="header-subtitle">El soundtrack de tu lectura</h3>
    """, unsafe_allow_html=True)


#---------------- Funciones con caché para evitar regeneración -------------------
# Función con caché para la generación de audiolibro
@st.cache_data(show_spinner="Generando audiolibro...")
def generate_audiobook(_client, model_tts, text):
    audio_bytes = None
    if not text:
        return None, "No se pudo extraer texto de esta página."
    try:
        if text.strip():
            speech = _client.audio.speech.create(model=model_tts, voice="onyx", input=text)
            audio_bytes = speech.read()
            return audio_bytes, "Audio generado con éxito."
        else:
            return None, "Texto de la página vacío o no válido."   
    except Exception as exc:
        return None, f"No se pudo generar la voz sintética: {exc}"
    
# Función con caché para la generación de música
@st.cache_data(show_spinner="Generando música...")
def get_book_music(prompt):
    return musicgen_generation(prompt)

# Función con caché para la generación de insights
@st.cache_data(show_spinner="Analizando el texto...")
def get_book_insights_cached(text):
    """Genera los insights (JSON) y los cachea."""
    # Aquí se llama a tu función original. Usamos el nombre 'get_book_insights_cached' 
    # para distinguir el wrapper cacheado.
    return get_book_insights(client,book_text=text)

# Función con caché para generar el prompt de generación de imagen
@st.cache_data(show_spinner="Analizando texto...")
def get_image_prompt_cached(text):
    return text_to_imagen(text) 

# Función con caché para generar la imagen
@st.cache_data(show_spinner="Creando imagen con IA...")
def get_generated_image_cached(prompt_imagen):
    output = crea_imagen(prompt_imagen) 
    if isinstance(output, str):
        return output
    elif isinstance(output, list) and output:
        return str(output[0])
    else:
        return str(output)

# Definimos el archivo de progreso de avance de lectura del archivo
PROGRESS_FILE = "pdf_progress.json"  


#------------------- Interfaz gráfica del lector de PDF ----------------------------
# Cargar progreso previo (si existe)
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)
else:
    progress = {}

# Barra de herramientas de la app
with st.sidebar:
    st.image("./BooksLives_Logo.png", width=200) # Ajusta el 'width' para controlar el tamaño.
    st.header("Empieza a leer")
    st.write("Selecciona algún libro de tu colección")
    uploaded_file = st.file_uploader("Sube un archivo PDF", type=["pdf"])

    st.markdown("---")


if uploaded_file:
    # Definimos un layout de dos columnas para la app
    col_lector, col_audio_controls = st.columns([8, 2])

    # Columna del lector de PDF
    with col_lector:
        pdf_bytes = uploaded_file.read()
        # Convertir PDF a imágenes (una por página)
        pages = convert_from_bytes(pdf_bytes)
        num_pages = len(pages)

        # Recuperar página guardada si existe
        pdf_id = hashlib.md5(pdf_bytes).hexdigest()

        if pdf_id in progress:
            st.session_state.page_number = progress[pdf_id]
        elif "page_number" not in st.session_state:
            st.session_state.page_number = 1

        # Mostrar número de página
        st.info(f"Página **{st.session_state.page_number}** de **{num_pages}**") # Estilo más limpio

        # Mostrar la página actual
        st.image(
            pages[st.session_state.page_number - 1], 
            use_container_width=True, # Usa el ancho completo de la columna para la imagen
            caption=f"Página {st.session_state.page_number} de {num_pages}"
        )
        # Navegación en el PDF
        col1, col2, col3, col4 = st.columns([1, 1, 2, 1]) 

        # Botón anterior
        with col1:
            if st.button("⬅️ Anterior", use_container_width=True):
                if st.session_state.page_number > 1:
                    st.session_state.page_number -= 1
                    progress[pdf_id] = st.session_state.page_number
                    with open(PROGRESS_FILE, "w") as f:
                        json.dump(progress, f)
                    st.rerun()         

        # Botón siguiente
        with col2:
            if st.button("Siguiente ➡️", use_container_width=True):
                if st.session_state.page_number < num_pages:
                    st.session_state.page_number += 1
                    progress[pdf_id] = st.session_state.page_number
                    with open(PROGRESS_FILE, "w") as f:
                        json.dump(progress, f)
                    st.rerun()

        # Campo para ir a página
        with col4:
            target_page = st.number_input(
                "Ir a página:", 
                min_value=1, 
                max_value=num_pages, 
                value=st.session_state.page_number, 
                step=1
            )
            if st.button("Ir", use_container_width=True):
                st.session_state.page_number = target_page
                progress[pdf_id] = target_page
                with open(PROGRESS_FILE, "w") as f:
                    json.dump(progress, f)
                st.rerun()

    #----------------- Extracción de texto, análisis y construcción de widgets de la aplicación ----------------#
    # Extraer texto de la página actual
    reader = PdfReader(BytesIO(pdf_bytes))
    text = reader.pages[st.session_state.page_number - 1].extract_text()

    with col_audio_controls:
        # -:--------------- Generación de audiolibro -----------------------------------------#

        audio_bytes, status_message = generate_audiobook(client, model_tts, text) #<...........Verdadera
        #with open('./audio_lectura_prueba.wav', 'rb') as f:#<...........Linea para desarrollo de pruebas
         #   audio_bytes = BytesIO(f.read())
        #status_message = "éxito"

        # Muestra el mensaje de estado (éxito o error/info)
        if "éxito" in status_message:
            st.success(status_message)
        elif "No se pudo" in status_message:
            st.error(status_message)
        else:
            st.info(status_message)

        # Muestra el reproductor solo si hay bytes de audio
        st.caption("Audiolibro")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3", start_time=0, sample_rate=None, end_time=None, loop=False, autoplay=False, width="stretch")
        
        st.markdown("---") # Separador visual

        #st.subheader("Música de Ambiente")

        # -:--------------- Generación de música basada en el libro -----------------------------------------#

        # Inicializar el estado de la música si no existe
        if 'book_music_bytes' not in st.session_state:
            st.session_state.book_music_bytes = None

        with st.sidebar:
            # Música para tu lectura
            prompt_musica = None
            try:
                # Obtener el prompt (la función text_to_music se ejecuta en cada rerun)
                prompt_musica = text_to_music(text) #<...........Verdadera
                #prompt_musica = 'Prompt_prueba' #<...........Linea para desarrollo de pruebas

            except Exception as exc:
                # Manejo de errores de text_to_music
                st.error(f"Error al procesar texto para música: {exc}")
                
            # El botón dispara la generación
            boton_genera_musica = st.button("Generar música para tu lectura")

            if boton_genera_musica:
                if prompt_musica:
                    with st.spinner("Generando audio... esto puede tardar unos momentos."):
                        try:
                            # 1. Llamar a la función cacheada para obtener los bytes del audio
                            book_music_bytes = get_book_music(prompt_musica) #<...........Verdadera
                            #with open('./music_prueba.wav', 'rb') as f:  #<...........Linea para desarrollo de pruebas
                            #    book_music_bytes = BytesIO(f.read())
                            
                            # 2. ALMACENAR LOS BYTES EN EL ESTADO DE SESIÓN
                            st.session_state.book_music_bytes = book_music_bytes
                            
                            st.success("¡Música generada!")
                            
                        except Exception as e:
                            # Limpiar el estado de sesión en caso de error
                            st.session_state.book_music_bytes = None 
                            st.error(f"Ocurrió un error durante la generación: {e}")
                else:
                    st.warning("No se pudo generar un prompt musical válido para esta página.")
                    st.session_state.book_music_bytes = None
        st.caption('Música para acompañar tu lectura')
        #Ejecución del reproductor de manera permanente (caché)
        if st.session_state.book_music_bytes:
            st.audio(
                st.session_state.book_music_bytes, 
                format='audio/wav', 
                loop=True
            )
        with st.sidebar:
            st.header("Insights del texto")
            st.write('Emociones que describen lo que lees')
            vecsen=vector_sentimientos(text)
            #vecsen='Emociones'
            st.write(vecsen)
            st.markdown("---")

        # -:--------------- Generación de Insights del libro -----------------------------------------#

        if 'page_insights' not in st.session_state:
            st.session_state.page_insights = None

        with st.sidebar:
            
            # El botón dispara la generación
            boton_genera_insights = st.button("Generar insights de tu lectura")

            if boton_genera_insights:
                with st.spinner("Generando insights... esto puede tardar unos momentos."):
                    try:
                        # 1. Llamar a la función cacheada
                        insights = get_book_insights_cached(text)
                        
                        # 2. Guardar el resultado en la sesión
                        st.session_state.page_insights = insights
                        
                        # 3. Mostrar la "Ventana Flotante" (st.toast) para confirmar el éxito
                        st.toast("¡Insights generados! Revisa los resultados a continuación.", icon="💡")
                        
                    except Exception as e:
                        st.session_state.page_insights = None # Limpiar en caso de error
                        st.error(f"Ocurrió un error durante la generación: {e}")
                        st.toast("Error al generar insights.", icon="❌")

        if st.session_state.page_insights:
            st.caption("Insights generados")
            
            # Mostrar el JSON de manera estructurada y limpia
            insights_data = st.session_state.page_insights

            if isinstance(insights_data, dict):
                for key, value in insights_data.items():
                    st.markdown(f"**{key.capitalize()}:** {value}")
            else:
                st.json(insights_data)
                
        else:
            st.info("Pulsa el botón para analizar el texto.")

        # ------------------ SECCIÓN DE GENERACIÓN DE IMAGEN ------------------
        # Inicializar el estado de la música si no existe
        if 'generated_image_data' not in st.session_state:
            st.session_state.generated_image_data = None
            
        with st.sidebar:
            st.markdown("---")
            st.header("Generar Imagen")

            boton_genera_imagen = st.button("Generar Imagen representativa")

            if boton_genera_imagen:
                with st.spinner("Preparando el prompt y generando la imagen... esto puede tardar un momento."):
                    try:
                        prompt_imagen = get_image_prompt_cached(text)
                        
                        if not prompt_imagen:
                            st.warning("No se pudo generar un prompt de imagen válido para esta página.")
                            st.session_state.generated_image_data = None
                        else:
                            imagen_data = get_generated_image_cached(prompt_imagen)

                            #image_url = imagen_data
                            #st.markdown(
                            #    f"Haz clic aquí para ver la imagen generada: **[Ver Imagen Generada]({image_url})**",
                            #    unsafe_allow_html=True
                            #)
                                
                            st.session_state.generated_image_data = imagen_data
                            
                            st.toast("¡Imagen generada! Revisa el panel principal.", icon="🖼️")
                            
                    except Exception as e:
                        st.session_state.generated_image_data = None # Limpiar en caso de error
                        st.error(f"Ocurrió un error durante la generación de la imagen: {e}")
                        st.toast("Error al generar la imagen.", icon="❌")
        st.markdown("---") 
        st.subheader("🖼️ Visualización con IA")
        if st.session_state.generated_image_data:
            image_url = st.session_state.generated_image_data
            
            st.success("¡Imagen generada!")
            
            # Usamos Markdown para crear un enlace clicleable
            st.markdown(
                f"Haz clic aquí para ver la imagen generada: **[Ver Imagen Generada]({image_url})**",
                unsafe_allow_html=True
            )
            
            # Opcional: Mostrar la URL completa en una expansión para copiar
            with st.expander("Mostrar URL completa"):
                st.code(image_url)

        else:
            st.info("Presiona 'Generar Imagen de la Página' en la barra lateral.")

        










