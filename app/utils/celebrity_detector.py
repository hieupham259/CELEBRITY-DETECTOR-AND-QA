import os
import groq
import base64

from dotenv import load_dotenv
load_dotenv()


class CelebrityDetector:

    def __init__(self):
        self.model = "meta-llama/llama-4-maverick-17b-128e-instruct"
        self.client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))

    def identify(self , image_bytes):
        encoded_image = base64.b64encode(image_bytes).decode()

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """You are a celebrity recognition expert AI. 
                                Identify the person in the image. If known, respond in this format:

                                - **Full Name**:
                                - **Profession**:
                                - **Nationality**:
                                - **Famous For**:
                                - **Top Achievements**:
                            If unknown, return "Unknown"."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_completion_tokens=1024,
        )

        # print(completion.choices[0].message)
        # content = completion.choices[0].message.content if completion.choices[0].message else "Unknown"
        if completion.choices[0].message:
            result = completion.choices[0].message.content
            name = self.extract_name(result)
            return result, name
        
        return "Unknown", ""

    def extract_name(self, content):
        for line in content.splitlines():
            if line.lower().startswith("- **full name**:"):
                name_part = line.split(":", 1)[1].strip()
                return name_part

        return "Unknown"  