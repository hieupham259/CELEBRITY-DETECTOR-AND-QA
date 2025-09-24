import cv2
from io import BytesIO
import numpy as np
from PIL import Image

def process_image(image_file, format=None):
    in_memory_file = BytesIO()
    if format:
        image_file.save(in_memory_file, format=format)
    else:
        image_file.save(in_memory_file)

    image_bytes = in_memory_file.getvalue()
    nparr = np.frombuffer(image_bytes, np.uint8)

    img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return image_bytes, None

    largest_face = max(faces, key=lambda r: r[2] * r[3])

    (x, y, w, h) = largest_face

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    is_success, buffer = cv2.imencode(".jpg", img)

    return buffer.tobytes(), largest_face

if __name__ == "__main__":
    # Open an image from disk
    image_file = Image.open(r"E:\CELEBRITY-DETECTOR-AND-QA\images\rose.jpg")

    image_bytes, face_coords = process_image(image_file, format="JPEG")
    if image_bytes:
        print(f"Face detected.")
    else:
        print("No face detected.")

    from celebrity_detector import CelebrityDetector
    celeb_detector = CelebrityDetector()
    # result, name = celeb_detector.identify(image_bytes)
    # print("name: ", name)
    result, name = celeb_detector.identify_gemini(image_bytes)
    print("name: ", name)