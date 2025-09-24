import os
import groq

from dotenv import load_dotenv
load_dotenv()

class QAEngine:

    def __init__(self):
        self.model  = "meta-llama/llama-4-maverick-17b-128e-instruct"
        self.client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))

    def ask_about_celebrity(self, name, question):
        prompt = f"""
                    You are a AI Assistant that knows a lot about celebrities. You have to answer questions about {name} concisely and accurately.
                    Question : {question}
                    """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=512
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return "Sorry I couldn't find the answer"

# qa = QAEngine()
# content = qa.ask_about_celebrity("Tom Cruise", "What are his top achievements?")
# print(content)