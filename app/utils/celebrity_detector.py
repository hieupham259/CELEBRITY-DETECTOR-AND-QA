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
    
    def identify_gemini(self, image_bytes):
        """
        Use Google Gemini 2.5 Flash model to identify celebrities in images
        """
        try:
            import google.generativeai as genai
            
            # Configure Gemini API
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            
            # Initialize Gemini 2.5 Flash model
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Convert image bytes to PIL Image for Gemini
            from PIL import Image
            from io import BytesIO
            
            image = Image.open(BytesIO(image_bytes))
            
            # Create prompt for celebrity identification
            prompt = """You are a celebrity recognition expert AI. 
            Analyze the image carefully and identify the person. 
            If you recognize the person as a celebrity, respond in this exact format:

            - **Full Name**: [Full name including any alternate names]
            - **Profession**: [Main profession/occupation]
            - **Nationality**: [Country of origin]
            - **Famous For**: [What they are most known for]
            - **Top Achievements**: [Major accomplishments or awards]
            
            If you cannot identify the person or they are not a known celebrity, simply return "Unknown"."""
            
            # Generate response
            response = model.generate_content([prompt, image])
            
            if response and response.text:
                result = response.text.strip()
                name = self.extract_name(result)
                return result, name
            else:
                return "Unknown", "Unknown"
                
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            return f"Error: {str(e)}", "Unknown"  