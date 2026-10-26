import cv2
import numpy as np
import os
from config import FACE_MATCH_THRESHOLD, MIN_FACE_MATCH_PERCENTAGE

class FaceNetAuthenticator:
    def __init__(self):
        self.use_pytorch_facenet = False
        # Initialize OpenCV face cascade for fallback extraction
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        try:
            import torch
            from facenet_pytorch import MTCNN, InceptionResnetV1
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.mtcnn = MTCNN(keep_all=False, select_largest=True, device=self.device)
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            self.use_pytorch_facenet = True
            print("Loaded PyTorch InceptionResnetV1 FaceNet Model.")
        except Exception as e:
            print(f"PyTorch FaceNet initialization fallback: {e}")

    def extract_embedding(self, image_input):
        """
        Extracts 128/512 dimension face feature embedding from image path or BGR image numpy array.
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                return None
        else:
            img = image_input

        if self.use_pytorch_facenet:
            try:
                import torch
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                from PIL import Image
                pil_img = Image.fromarray(rgb_img)
                face_tensor = self.mtcnn(pil_img)
                if face_tensor is not None:
                    face_tensor = face_tensor.unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        embedding = self.resnet(face_tensor).squeeze(0).cpu().numpy()
                    # L2 Normalize
                    embedding = embedding / np.linalg.norm(embedding)
                    return embedding.tolist()
            except Exception as ex:
                print(f"FaceNet PyTorch embedding failed, switching to OpenCV extraction: {ex}")

        # Fallback OpenCV Face Extraction & Histogram/LBP Vector
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        if len(faces) == 0:
            # Crop center if face detector didn't find clear face boundary
            h, w = gray.shape
            face_roi = gray[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)]
        else:
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]

        face_resized = cv2.resize(face_roi, (128, 128))
        # Compute normalized multi-scale spatial histogram embedding
        hist = cv2.calcHist([face_resized], [0], None, [128], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Add edge gradient structure features
        gx = cv2.Sobel(face_resized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(face_resized, cv2.CV_32F, 0, 1, ksize=3)
        mag, _ = cv2.cartToPolar(gx, gy)
        mag_hist = cv2.calcHist([mag.astype(np.uint8)], [0], None, [128], [0, 256])
        mag_hist = cv2.normalize(mag_hist, mag_hist).flatten()
        
        vector = np.concatenate([hist, mag_hist])
        vector = vector / np.linalg.norm(vector)
        return vector.tolist()

    def compare_faces(self, registered_embedding, captured_embedding_or_img):
        """
        Compares registered embedding with webcam captured frame or embedding.
        Returns: auth_passed (bool), distance (float), confidence_score (float)
        Strictly requires match confidence >= 70.0%; otherwise REJECTS authentication.
        """
        if registered_embedding is None:
            return False, 1.0, 0.0

        if not isinstance(captured_embedding_or_img, list):
            captured_embedding = self.extract_embedding(captured_embedding_or_img)
        else:
            captured_embedding = captured_embedding_or_img

        if captured_embedding is None:
            return False, 1.0, 0.0

        reg_vec = np.array(registered_embedding, dtype=np.float32)
        cap_vec = np.array(captured_embedding, dtype=np.float32)

        # Handle size mismatch if models differ
        min_dim = min(len(reg_vec), len(cap_vec))
        reg_vec = reg_vec[:min_dim]
        cap_vec = cap_vec[:min_dim]

        # Normalize
        reg_norm = reg_vec / (np.linalg.norm(reg_vec) + 1e-8)
        cap_norm = cap_vec / (np.linalg.norm(cap_vec) + 1e-8)

        # Cosine similarity
        cosine_sim = float(np.dot(reg_norm, cap_norm))
        cosine_dist = float(max(0.0, 1.0 - cosine_sim))

        # Calculate confidence score percentage (0 to 100%)
        confidence = float(max(0.0, min(100.0, (1.0 - (cosine_dist / 0.8)) * 100)))
        confidence = round(confidence, 2)

        # Strict 70% threshold check: MUST be >= 70.0% match score to pass; otherwise REJECT!
        auth_passed = (confidence >= MIN_FACE_MATCH_PERCENTAGE) and (cosine_dist < 0.45)

        return auth_passed, round(cosine_dist, 4), confidence

# Global Instance
facenet_authenticator = FaceNetAuthenticator()
