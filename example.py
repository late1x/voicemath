from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from openai import OpenAI
import base64
import os
import PIL
import logging

from voicemath.api import *
from voicemath.config import Configuration
from voicemath.helpers.exceptions import *

configs = Configuration()

client = OpenAI(
    api_key= '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'
)

# Configurar logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Example:
    def subsEnglishUsingOpenAI(self, text):
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": """I will give you a mathematical expression, and I would like you to generate an audio reading of this operation, where you will replace the symbols as follows: the "+" sign will be read as "plus," the "-" sign as "minus," the "*" sign as "times," the "÷" sign as "divided by," and the "=" sign will be read as "is equal to." When the expression includes the "^" sign, it should be read as "raised to the" followed by the ordinal form of the exponent and ending with the word "power." When the expression contains a fraction represented by the "/" symbol, before reading the denominator, say "the fraction with the top part" followed by the contents of the numerator, then say "and the bottom part" followed by the contents of the denominator. Don't generate the solution of ecuation."""
            },
            {
                "role": "user",
                "content": text
            }]
        )
        return completion.choices[0].message.content

    def subsSpanishUsingOpenAI(self, text):
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": """I will give you a mathematical expression, and I would like you to generate an audio reading of this operation, where you will replace the symbols as follows: the "+" sign will be read as "más," the "-" sign as "menos," the "*" sign as "por," the "÷" sign as "entre," and the "=" sign will be read as "es igual a." When the expression includes the "^" sign, it should be read as "elevado a la" followed by the ordinal form of the exponent and ending with the word "potencia." When the expression contains a fraction represented by the "/" symbol, before reading the denominator, say "la fracción con la parte superior" followed by the contents of the numerator, then say "y con la parte inferior de" followed by the contents of the denominator. Don't generate the solution of ecuation."""
            },
            {
                "role": "user",
                "content": text
            }]
        )
        return completion.choices[0].message.content

    def generateAudioWithOpenAI(self, text):
        response = client.audio.speech.create(
            model="tts-1",
            voice="shimmer",
            input=text
        )
        
        audio_bytes = response.content
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        return audio_base64

app = Flask(__name__)
CORS(app)

def resize_image(img, max_height=200):
    if img is None:
        raise ValueError("Imagen inválida")
        
    height, width = img.shape[:2]
    if height <= max_height:
        return img
        
    aspect_ratio = width / height
    new_height = max_height
    new_width = int(max_height * aspect_ratio)
    
    # Usar INTER_AREA para reducción, INTER_CUBIC para ampliación
    interpolation = cv2.INTER_AREA if height > max_height else cv2.INTER_CUBIC
    return cv2.resize(img, (new_width, new_height), interpolation=interpolation)


def preprocess_image(image_path):
    # Cargar imagen usando OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("No se pudo cargar la imagen")

    # Redimensionar imagen manteniendo la relación de aspecto
    img = resize_image(img, max_height=600)

    # Convertir imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mejorar contraste usando ecualización del histograma
    gray = cv2.equalizeHist(gray)

    # Aplicar desenfoque gaussiano con kernel adaptativo
    kernel_size = max(3, min(img.shape[0], img.shape[1]) // 100)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Detectar bordes usando Canny con umbral adaptativo
    median = np.median(blur)
    lower = int(max(0, (1.0 - 0.33) * median))
    upper = int(min(255, (1.0 + 0.33) * median))
    edges = cv2.Canny(blur, lower, upper)

    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Ordenar contornos por área y seleccionar el más grande
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        pizarra_contour = contours[0]

        # Aproximar contorno al polígono más cercano
        epsilon = 0.02 * cv2.arcLength(pizarra_contour, True)
        approx = cv2.approxPolyDP(pizarra_contour, epsilon, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)

            # Ordenar puntos
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            # Definir dimensiones del nuevo pizarrón
            max_width = 1223
            max_height = 487
            dst = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]], dtype="float32")

            # Aplicar transformación de perspectiva
            M = cv2.getPerspectiveTransform(rect, dst)
            warp = cv2.warpPerspective(img, M, (max_width, max_height))

            # Definir medidas del recorte
            height, width = warp.shape[:2]
            desired_width = 1198
            desired_height = 455

            # Coordenadas para recorte centrado
            x_start = (width - desired_width) // 2
            y_start = (height - desired_height) // 2

            # Realizar recorte
            crop = warp[y_start:y_start + desired_height, x_start:x_start + desired_width]

            # Aplicar filtros para limpiar el fondo
            warp_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Umbral adaptativo
            block_size = 11
            C = 2
            warp_thresh = cv2.adaptiveThreshold(
                warp_gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                C
            )
            
            # Aplicar operaciones morfológicas para limpiar ruido
            kernel = np.ones((3,3), np.uint8)
            warp_clean = cv2.morphologyEx(warp_thresh, cv2.MORPH_CLOSE, kernel)
            warp_clean = cv2.medianBlur(warp_clean, 5)

            # Guardar imagen final
            final_image_path = "operacion.jpg"
            cv2.imwrite(final_image_path, warp_clean)
            return final_image_path
        else:
            raise ValueError("No se detectó un contorno rectangular.")
    else:
        raise ValueError("No se encontraron contornos.")


@app.route("/predict", methods=["POST"])
def predict():
    # Validación del archivo
    if "file" not in request.files:
        return jsonify({
            "error": "No se encontró ningún archivo en la solicitud",
            "error_code": "NO_FILE"
        }), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({
            "error": "No se seleccionó ningún archivo",
            "error_code": "EMPTY_FILENAME"
        }), 400

    # Validación del tipo de archivo
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({
            "error": "Formato de archivo no soportado. Use PNG, JPG o JPEG",
            "error_code": "INVALID_FORMAT"
        }), 400

    try:
        # Procesamiento de imagen
        image = Image.open(file)
        image_path = "temp_image.png"
        image.save(image_path)

        preprocessed_image_path = preprocess_image(image_path)

        # Reconocimiento de expresión matemática
        hme_recognizer = HME_Recognizer()
        hme_recognizer.load_image(preprocessed_image_path, data_type='path')
        expression, img = hme_recognizer.recognize()
        
        if not expression or expression.isspace():
            return jsonify({
                "error": "No se pudo detectar una expresión matemática en la imagen",
                "error_code": "NO_EXPRESSION_DETECTED"
            }), 400

        example = Example()
        
        try:
            # Procesamiento de texto con OpenAI
            spanish_expression = example.subsSpanishUsingOpenAI(expression)
            english_expression = example.subsEnglishUsingOpenAI(expression)
        except Exception as e:
            return jsonify({
                "error": "Error al procesar el texto con OpenAI",
                "error_code": "OPENAI_TEXT_ERROR",
                "details": str(e)
            }), 500

        try:
            # Generación de audio
            spanish_audio_base64 = example.generateAudioWithOpenAI(spanish_expression)
            english_audio_base64 = example.generateAudioWithOpenAI(english_expression)
        except Exception as e:
            return jsonify({
                "error": "Error al generar el audio",
                "error_code": "AUDIO_GENERATION_ERROR",
                "details": str(e)
            }), 500

        # Limpieza de archivos temporales
        try:
            os.remove(image_path)
            os.remove(preprocessed_image_path)
        except Exception:
            # Log error but don't fail the request
            print(f"Warning: Could not delete temporary files")

        return jsonify({
            "math_expression": expression,
            "expression_in_spanish": spanish_expression,
            "expression_in_english": english_expression,
            "audio_base64_spanish": spanish_audio_base64,
            "audio_base64_english": english_audio_base64
        })

    except (GrammarError, SintaticError, LexicalError) as e:
        logger.error(f"Error en el procesamiento matemático: {str(e)}")
        return jsonify({
            "error": "Error en el procesamiento matemático",
            "error_code": "MATH_PROCESSING_ERROR",
            "details": str(e)
        }), 400

    except PIL.UnidentifiedImageError as e:
        logger.error(f"Error de imagen no válida: {str(e)}")
        return jsonify({
            "error": "El archivo no es una imagen válida",
            "error_code": "INVALID_IMAGE"
        }), 400

    except Exception as e:
        logger.error(f"Error interno del servidor: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Error interno del servidor",
            "error_code": "INTERNAL_SERVER_ERROR",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
