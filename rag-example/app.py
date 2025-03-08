from flask import Flask, request, jsonify
from flask_cors import CORS  # Importa CORS
from faces import FaceEmbedding
from rag import Rag
import base64
import tempfile
import os

app = Flask(__name__)
# Habilita CORS para todas las rutas
CORS(app, resources={r"/*": {"origins": "*"}})
"""
    RAG Database = Vector + sqlite with Archmey as ORM
"""
rag = Rag(0,1,512)

"""
    Face recognition + embedding extrac
"""
face = FaceEmbedding()

from flask import make_response

@app.before_request
def before_request():
    headers = {'Access-Control-Allow-Origin': '*',
               'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
               'Access-Control-Allow-Headers': 'Content-Type'}
    if request.method.lower() == 'options':
        return jsonify(headers), 200



def save_base64_as_jpg(base64_string):
    """
    Decodifica una imagen en Base64 y la guarda como un archivo JPG temporal.
    
    :param base64_string: Cadena en Base64 que representa la imagen.
    :return: Ruta del archivo temporal guardado.
    """
    try:
        image_data = base64.b64decode(base64_string)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", mode="wb") as temp_file:
            temp_file.write(image_data)
            return temp_file.name  # Retorna la ruta del archivo guardado
    except Exception as e:
        return None, str(e)

@app.route("/person/", methods=["POST", "OPTIONS"])
def create_person():
    """
    Cargar un muñeco en la base de datos con su imagen.
    
    Request:
    {
        "data": { ... },
        "imageBase64": "..."
    }
    
    Response:
    201 Created
    {
        "message": "Person created successfully"
    }
    400 Error
    {
        "error": "Error message"
    }
    """
    try:
        data = request.json
        if "data" not in data or "imageBase64" not in data:
            return jsonify({"error": "Missing data or imageBase64"}), 400

        # Guardar la imagen
        image_path = save_base64_as_jpg(data["imageBase64"])
        if not image_path:
            return jsonify({"error": "Invalid Base64 image"}), 400

        print(image_path)

        try:
            landmarks, frame = face.detect_from_image(image_path)
        except Exception as e:
            print("Error detecting face: ", str(e))
            return jsonify({"error": str(e)})


        try:
            face_normalized, cropped = face.crop(landmarks, frame)
        except Exception as e:
            print("Error croping face: ", str(e))
            return jsonify({"error": str(e)})
        
        emb = face.get_embedding(face_normalized)

        try:
            rag.insert_embedding(emb, data["data"])            
        except Exception as e:
            print("Error at inserion in RAG ", str(e))
            return jsonify({"error": str(e)})

        return jsonify({"message": "Person created successfully"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/person/find/", methods=["POST", "OPTIONS"])
def find_person():
    """
    Buscar un muñeco en la base de datos por su imagen.

    Request:
    {
        "imageBase64": "..."
    }

    Response:
    200 OK
    {
        "data": { ... }
    }
    400 Error
    {
        "error": "Error message"
    }
    """
    try:
        data = request.json
        if "imageBase64" not in data:
            return jsonify({"error": "Missing imageBase64"}), 400

        # Guardar la imagen temporalmente
        query_image_path = save_base64_as_jpg(data["imageBase64"])
        if not query_image_path:
            return jsonify({"error": "Invalid Base64 image"}), 400

        # Guardar la imagen
        image_path = save_base64_as_jpg(data["imageBase64"])
        if not image_path:
            return jsonify({"error": "Invalid Base64 image"}), 400


        try:
            landmarks, frame = face.detect_from_image(image_path)
        except Exception as e:
            print("Error detecting face: ", str(e))
            return jsonify({"error": str(e)})


        try:
            face_normalized, cropped = face.crop(landmarks, frame)
        except Exception as e:
            print("Error croping face: ", str(e))
            return jsonify({"error": str(e)})
        
        emb = face.get_embedding(face_normalized)

        try:
            result = rag.search(emb)
        except Exception as e:
            print("Error searching face: ", str(e))
            return jsonify({"error": str(e)})

        print("Result: ", result)

        return jsonify({"data": result["data"]}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
