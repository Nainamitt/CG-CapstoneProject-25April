import os
import csv
import uuid
import logging
import requests
from typing import Optional, Dict, Tuple
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# ==============================
# Environment and Logging
# ==============================
load_dotenv()

# Use the deployment name you set in Azure AI Foundry (e.g., "gpt-4o")
AZURE_ENDPOINT = os.getenv("AZURE_AGENT_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_AGENT_KEY")
MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4") 

if not AZURE_ENDPOINT or not AZURE_KEY:
    raise ValueError("Missing environment variables: AZURE_AGENT_ENDPOINT or AZURE_AGENT_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Customer Support Agent", version="5.1")

# ==============================
# Data Models
# ==============================
class ChatRequest(BaseModel):
    user_id: str
    query: str

class ChatResponse(BaseModel):
    ai_response: str
    intent: str
    ticket_id: Optional[str] = None
    status: str

# ==============================
# Enterprise Core (Backend)
# ==============================
class EnterpriseCore:
    def __init__(self, path: str):
        self.path = path
        self.kb = self._load_kb(path)

    def _load_kb(self, path: str) -> Dict[str, str]:
        if not Path(path).exists():
            logger.warning(f"Knowledge base {path} not found. Running without fallback.")
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                return {
                    row["Intent"].strip().lower(): row["Response"].strip()
                    for row in csv.DictReader(f)
                }
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return {}

    async def create_ticket(self) -> str:
        # Simulates backend DB write
        return f"TKT-{uuid.uuid4().hex[:6].upper()}"

# ==============================
# Azure AI Agent Logic
# ==============================
class AIAgent:
    def __init__(self, core: EnterpriseCore):
        self.core = core

    def call_azure(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_KEY  # Standard Azure header
        }

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 800,
            "temperature": 0.3,
        }

        try:
            response = requests.post(AZURE_ENDPOINT, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Azure API Communication Error: {e}")
            raise HTTPException(status_code=502, detail="AI Service currently unavailable")

    def get_intent(self, query: str) -> str:
        # Added stricter instructions to ensure AI doesn't ramble
        system_instructions = (
            "Classify query into: Order_Status, Support_Ticket, or General_Query. "
            "Output ONLY the intent name."
        )
        intent = self.call_azure(system_instructions, query)
        # Clean the response in case AI adds punctuation
        return intent.replace(".", "").strip()

    async def process(self, user_id: str, query: str) -> Tuple[str, str, Optional[str]]:
        intent = self.get_intent(query)
        logger.info(f"Detected Intent: {intent} for User: {user_id}")

        # Try to get AI response, fallback to KB if it fails
        try:
            response = self.call_azure("You are a professional support assistant.", query)
        except Exception:
            response = self.core.kb.get(intent.lower(), "I'm having trouble connecting to my brain, but I've noted your request.")

        ticket_id = None
        if intent == "Support_Ticket":
            ticket_id = await self.core.create_ticket()
            response += f"\n\n[System Update] A ticket has been raised: {ticket_id}"

        return response, intent, ticket_id

# ==============================
# Initialization & Routes
# ==============================
enterprise_core = EnterpriseCore("knowledge_base.csv")
ai_agent = AIAgent(enterprise_core)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    response_text, intent, ticket_id = await ai_agent.process(request.user_id, request.query)
    
    return ChatResponse(
        ai_response=response_text,
        intent=intent,
        ticket_id=ticket_id,
        status="success"
    )

if __name__ == "__main__":
    import uvicorn
    # Added host and port for easier deployment visibility
    uvicorn.run(app, host="0.0.0.0", port=8000)