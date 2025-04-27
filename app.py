import google.generativeai as genai
import openai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import litellm
from litellm import acompletion
import os 
# Create FastAPI app
from openai import OpenAI
app = FastAPI()

# Enable CORS (adjust allowed origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for our endpoint.
class AnalyzeRequest(BaseModel):
    code: str
    api_key: str
    model_name: str
    provider: str  # Allowed values: "google", "openai", "huggingface"

PROMPT_TEMPLATE = """
For the following code, please perform an internal, step-by-step chain-of-thought reasoning process to determine its time and space complexities. However, do not include your internal reasoning in the final output; only provide the final result in the format below.

Expected Output:
Time Complexity: 
Space Complexity: 

Code:
{code}
"""


@app.post("/analyze")
async def analyze_code(request: AnalyzeRequest):

    try:
        prompt = PROMPT_TEMPLATE.format(code=request.code)
        # print(request.code)
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
        if request.provider.lower() == "huggingface":
            try:
                loop = asyncio.get_event_loop()

                # For Hugging Face, we use the OpenAI client with a custom base URL.
                hf_client = OpenAI(
                    base_url="https://api-inference.huggingface.co/v1/",
                    api_key=request.api_key
                )
                # Call the chat completions endpoint.
                response = await loop.run_in_executor(
                None,
                lambda:hf_client.chat.completions.create(
                    model=request.model_name,
                    messages=messages,
                    max_tokens=1024,
                    stream=False
                )
                )
                result_text = response.choices[0].message.content.strip() 
                # print(request.code)
                return {"result": result_text}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        else:
            async def get_response():
                prefix_response=request.provider.lower()
                prefix_api=request.provider.upper()
                # print(prefix_response)
                os.environ[f"{prefix_api}_API_KEY"]=request.api_key
                model=request.model_name
                response = await acompletion(
                    model=f"{prefix_response}/{model}", 
                    messages=messages
                    )
                return response
            
            response = await  get_response()
            result_text = response.choices[0].message.content.strip()

            return {"result": result_text}
            # print(result_text)
    except Exception as e:
        # print("jo")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
