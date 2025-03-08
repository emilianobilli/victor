import cv2
import time
import numpy as np
import onnxruntime as ort
import mediapipe as mp

class FaceEmbedding(object):
    def __init__(self):
        self.onnx_session = ort.InferenceSession("facenet.onnx", providers=["CPUExecutionProvider"])
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    @staticmethod
    def crop(landmarks, img):
        # Obtener dimensiones de la imagen original
        h, w, _ = img.shape

        # Extraer coordenadas de los puntos del rostro
        face_points = np.array([(int(landmark.x * w), int(landmark.y * h)) for landmark in landmarks], dtype=np.int32)

        # Calcular la bounding box
        min_x, min_y = np.min(face_points[:, 0]), np.min(face_points[:, 1])
        max_x, max_y = np.max(face_points[:, 0]), np.max(face_points[:, 1])

        # Expandir la bounding box ligeramente
        padding = 0 # Se aumentó el padding para evitar cortes en el rostro
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(w, max_x + padding)
        max_y = min(h, max_y + padding)

        # Recortar la región del rostro
        face_crop = img[min_y:max_y, min_x:max_x]

        if face_crop.size == 0:
            raise ValueError("Recorte de rostro inválido")

        # Determinar el tamaño de la imagen recortada
        crop_h, crop_w = face_crop.shape[:2]
        target_size = 160

        # Crear un lienzo negro de 160x160
        face_padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)

        # Redimensionar manteniendo la relación de aspecto
        aspect_ratio = crop_w / crop_h
        if aspect_ratio > 1:  
            new_w = target_size
            new_h = int(target_size / aspect_ratio)
        else:
            new_h = target_size
            new_w = int(target_size * aspect_ratio)

        face_resized = cv2.resize(face_crop, (new_w, new_h))

        # Calcular la posición de centrado
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2

        # Colocar la imagen redimensionada en el centro del lienzo
        face_padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = face_resized

        # Convertir a RGB (FaceNet usa RGB, OpenCV usa BGR)
        face_padded = cv2.cvtColor(face_padded, cv2.COLOR_BGR2RGB)

        # Normalizar a rango [-1,1]
        face_normalized = (face_padded.astype('float32') / 127.5) - 1.0

        # Expandir dimensiones a (1, 160, 160, 3) para el modelo
        face_normalized = np.expand_dims(face_normalized, axis=0)

        return face_normalized, face_padded

    @staticmethod
    def _open_image(image):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image}")

        return img

    @staticmethod
    def _detect_pose(landmarks, size):
        h, w = size
        KEY_POINTS = {
            "left_eye": 33,    # Ojo izquierdo
            "right_eye": 263,  # Ojo derecho
            "nose": 1,         # Nariz (punta)
        }
        points = {key: (int(landmarks[val].x * w), int(landmarks[val].y * h)) for key, val in KEY_POINTS.items()}

        nose_x = points["nose"][0]
        left_eye_x = points["left_eye"][0]
        right_eye_x = points["right_eye"][0]

        nose_to_left = abs(nose_x - left_eye_x)
        nose_to_right = abs(nose_x - right_eye_x)

        if abs(nose_to_left - nose_to_right) < 35:
            return 0
        else:
            return 1 if nose_to_left < nose_to_right else -1


    def detect_from_video(self, frontal_hold_time=2):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return None, None

        frontal_start_time = None
        image_captured = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                if self._detect_pose(landmarks, (h,w)) == 0:
                    yaw_status = "Frontal"
                    if frontal_start_time is None:
                        frontal_start_time = time.time()
                    elif time.time() - frontal_start_time >= frontal_hold_time and not image_captured:
                        # Capturar imagen y terminar la detección
                        cv2.imshow("Captured Face", frame)
                        cv2.waitKey(2000)  # Espera 2 segundos
                        cv2.destroyWindow("Captured Face")

                        cap.release()
                        cv2.destroyAllWindows()
                        return landmarks, frame  # Retorna la imagen capturada y los landmarks
                else:
                    yaw_status = "Mirando derecha" if 1 else "Mirando izquierda"
                    frontal_start_time = None  
                    image_captured = False  

                status_text = f"Yaw: {yaw_status}"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow("Head Pose Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return None, None  # Si no se capturó ninguna imagen

    def detect_from_image(self, image):

        img = self._open_image(image)

        """Detecta el rostro con MediaPipe, lo recorta, redimensiona a 160x160 y normaliza."""
        # Convertir a RGB (MediaPipe usa RGB, OpenCV usa BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detectar rostro con FaceMesh
        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            raise ValueError("No se detectó ningún rostro en la imagen")

        # Obtener landmarks del primer rostro detectado
        landmarks = results.multi_face_landmarks[0].landmark

        return landmarks, img

    @staticmethod
    def _normalize(embedding):
        norm_l2 = np.linalg.norm(embedding)
        return embedding/norm_l2

    def get_embedding(self, face_normalized):
        """Obtiene el embedding del rostro con FaceNet ONNX."""
        #face_normalized, face_padded = self._detect_crop(image_path)

        # Realizar inferencia con ONNX Runtime
        inputs = {self.onnx_session.get_inputs()[0].name: face_normalized}
        embedding = self.onnx_session.run(None, inputs)[0]
        embedding = self._normalize(embedding[0])
        
        return embedding.tolist()

if __name__ == "__main__":
    face = FaceEmbedding()
    landmarks, frame = face.detect_from_video()
    face_normalized, cropped = face.crop(landmarks, frame)
    emb = face.get_embedding(face_normalized)
    captured_path = "captured_face_test.jpg"
    cv2.imwrite(captured_path, cropped)
    captured_img = cv2.imread(captured_path)

