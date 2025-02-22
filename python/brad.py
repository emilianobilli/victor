import cv2
import numpy as np
import mediapipe as mp
import insightface

mp_face_mesh = mp.solutions.face_mesh

class Brad(object):
    def __init__(self):
        try:
            self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        except Exception as e:
            print("Error initializing MediaPipe FaceMesh:", e)
            self.face_mesh = None
        
        try:
            self.model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.model.prepare(ctx_id=0)
        except Exception as e:
            print("Error initializing InsightFace:", e)
            self.model = None

    def get_face_embeddings(self, image_path: str):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image from '{image_path}'")

            # Convert to RGB (MediaPipe uses RGB, OpenCV uses BGR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb) if self.face_mesh else None

            if results and results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Get image dimensions
                h, w, _ = image.shape

                # Extract face landmarks
                face_points = np.array([(int(landmark.x * w), int(landmark.y * h)) for landmark in landmarks], dtype=np.int32)

                # Find face bounding box
                min_x, min_y = np.min(face_points[:, 0]), np.min(face_points[:, 1])
                max_x, max_y = np.max(face_points[:, 0]), np.max(face_points[:, 1])

                # Crop face region
                face_crop = image[min_y:max_y, min_x:max_x]
                if face_crop.size == 0:
                    raise ValueError("Invalid face crop")

                # Resize to 112x112 for InsightFace (maintaining aspect ratio and padding)
                target_size = 112
                old_h, old_w = face_crop.shape[:2]
                aspect_ratio = old_w / old_h

                if aspect_ratio > 1:
                    new_w = target_size
                    new_h = int(target_size / aspect_ratio)
                else:
                    new_h = target_size
                    new_w = int(target_size * aspect_ratio)

                face_resized = cv2.resize(face_crop, (new_w, new_h))

                # Add padding if image is not exactly 112x112
                delta_w = target_size - new_w
                delta_h = target_size - new_h
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)

                face_padded = cv2.copyMakeBorder(face_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

                # Convert to RGB (InsightFace uses RGB, OpenCV uses BGR)
                face_padded = cv2.cvtColor(face_padded, cv2.COLOR_BGR2RGB)

                # Normalize image to range [-1,1] for InsightFace
                face_normalized = (face_padded.astype('float32') / 127.5) - 1.0

                if len(face_normalized.shape) == 3:
                    face_normalized = np.expand_dims(face_normalized, axis=0)

                face_normalized = np.transpose(face_normalized, (0, 3, 1, 2))  # (1,112,112,3) → (1,3,112,112)

                if not self.model:
                    raise RuntimeError("InsightFace model is not available")

                embedding = self.model.models["recognition"].forward(face_normalized)
                return embedding
            else:
                print("Warning: No faces detected in the image")
                return []
        except Exception as e:
            print(f"Error processing image '{image_path}':", e)
            return []

brad = Brad()
print(brad.get_face_embeddings("input.jpg"))