from flask import Blueprint, render_template, request

from app.utils.image_handler import process_image
from app.utils.celebrity_detector import CelebrityDetector
from app.utils.qa_engine import QAEngine

import base64

main = Blueprint('main', __name__)

celebrity_detector = CelebrityDetector()
qa_engine = QAEngine()

@main.route("/", methods=["GET", "POST"])
def index():
    player_info = ""
    result_img_data = ""
    user_question = ""
    answer = ""

    if request.method == "POST":
        if "image" in request.files:
            image_file = request.files["image"]

            if image_file:
                img_bytes, face_box = process_image(image_file)

                player_info, player_name = celebrity_detector.identify(img_bytes)