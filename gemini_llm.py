from google import genai

client = genai.Client(api_key="your gemini API Key")

response = client.models.generate_content(
    model="gemini-2.5-pro-exp-03-25", contents="Give me code to build LLM Multi AI Agent using Google Gemini API"
)
print(response.text)