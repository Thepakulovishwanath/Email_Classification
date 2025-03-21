import json
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define Pydantic model for FastAPI input validation
class EmailData(BaseModel):
    subject: str
    latestMessage: str
    messageId: str
    threadId: str

    class Config:
        extra = "allow"  # Allow extra fields if present

# Initialize FastAPI app
app = FastAPI()

class EmailClassifier:
    def __init__(self):
        # Initialize the Groq model
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name="llama-3.2-90b-vision-preview"
        )
        
    def apply_character_limit(self, text: str, max_chars: int, preserve_first: int, preserve_last: int) -> str:
        """
        Apply character limit to text, preserving beginning and end parts.
        """
        if text is None:
            return ""
        if len(text) <= max_chars:
            return text
        first_part = text[:preserve_first]
        last_part = text[-preserve_last:] if preserve_last > 0 else ""
        separator = "\n...[content trimmed]...\n"
        return first_part + separator + last_part
    
    def construct_system_prompt(self) -> str:
        """
        Construct the system prompt for email processing.
        """
        return """You are an experienced Customer Support Manager at WorldZone, a Freight Forwarding Company.
Your task is to accurately classify and extract information from emails in the freight forwarding industry.

For each email, you will:
1. Classify the email into one of the predefined categories
2. Extract key entities and information
3. Clean the email by removing pleasantries and signatures
4. Provide a brief summary of the email

Analyze carefully and be precise with your classifications and extracted information.
"""

    def construct_prompt(self, email_data: Dict[str, Any]) -> str:
        """
        Construct the user prompt for email processing.
        """
        subject = email_data.get('subject', '(No subject)')
        content = email_data.get('latestMessage', '')
        from_addresses = email_data.get('from', [])
        to_addresses = email_data.get('to', [])
        date = email_data.get('date')

        from_str = ', '.join([addr.get('email', '') for addr in from_addresses]) if from_addresses else 'Unknown'
        to_str = ', '.join([addr.get('email', '') for addr in to_addresses]) if to_addresses else 'Unknown'
        date_str = date.strftime("%Y-%m-%d %H:%M") if date else 'Unknown date'

        if len(content) > 12000:
            content = self.apply_character_limit(content, 12000, 10000, 2000)

        prompt = f"""Analyze the following email and provide a structured response:

EMAIL INFORMATION:
Date: {date_str}
From: {from_str}
To: {to_str}
Subject: {subject}

EMAIL CONTENT:
{content}
"""

        prompt += """
CLASSIFICATION TASK:
Classify this email into one of these primary categories:
1. Booking & Quotation
2. Documentation
3. Shipment Status
4. Customer Service
5. Financial
6. Compliance
7. Social & Promotional
8. Other

Then classify into one of these secondary categories:
- Booking Quotation Request
- Customer Booking Confirmation
- Bill of Lading (B/L) & Documents
- Customs Clearance Notifications
- Delivery Order (DO) Issuance
- Customs Documentation Requests
- Cargo Arrival Notices
- Proof of Delivery (POD)
- Exception Handling & Delay Notifications
- Shipment Status Updates
- Customer General Query - Customer
- Customer Escalation - Customer
- Agent Freight Quotations
- Rate Negotiations
- Lead Follow-Ups
- Invoice & Payment Requests
- Credit Notes & Adjustments
- Outstanding Balance Reminders
- Insurance Claims & Notifications
- Agent Response to Quote - Direct
- Payment Confirmations
- Regulatory Compliance Updates
- Dangerous Goods (DG) Compliance
- Social Media or other Promotional Emails
- Generic Email - All other communication

ENTITY EXTRACTION TASK:
Extract the following information if present:
1. Contact Information:
   - Names
   - Email addresses
   - Phone numbers
   - Company names

2. For Booking & Quotation emails:
   - Origin
   - Destination
   - Incoterms
   - Commodity
   - Hazardous (Y/N)
   - Weight
   - Container type
   - Volume
   - Quantity
   - Additional services needed

3. For Shipment Status emails:
   - Tracking numbers
   - Current status
   - Estimated arrival
   - Any delays or issues

4. For Financial emails:
   - Invoice numbers
   - Amounts
   - Currency
   - Payment details

CLEANING TASK:
Provide a cleaned version of the email with pleasantries, signatures, and non-business content removed.

SUMMARY TASK:
Provide a brief summary of the email's main purpose and key points.

RESPONSE FORMAT:
Provide your response in valid JSON format as follows:
{
  "classification": {
    "primary_category": "...",
    "secondary_category": "...",
    "confidence": 0.95
  },
  "contacts": [
    {
      "name": "...",
      "email_address": "...",
      "phone_number": "...",
      "mobile_number": "...",
      "designation": "...",
      "company_name": "..."
    }
  ],
  "entities": {
    // Entity fields will depend on the email category
  },
  "cleaned_content": "...",
  "summary": "..."
}
IMPORTANT: Do not include any explanatory text outside of the JSON structure. If certain fields have no data, include them as empty arrays [] or objects {} rather than explaining why they're empty. Make sure the response is strictly valid JSON without any additional comments or notes.
"""
        return prompt
    
    def classify_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify an email using the Groq model and return the JSON response.
        """
        try:
            system_prompt = self.construct_system_prompt()
            user_prompt = self.construct_prompt(email_data)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON response", "raw_response": response.content}
        except Exception as e:
            return {"error": str(e)}

# FastAPI endpoint
@app.post("/classify")
async def classify_email_endpoint(email: EmailData):
    classifier = EmailClassifier()
    email_dict = email.dict()
    result = classifier.classify_email(email_dict)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

def main():
    classifier = EmailClassifier()
    print("Please paste the JSON email data below and press Enter once to submit:")
    
    # Get single input (pasted JSON)
    email_json = input()
    
    # Parse and process the input
    try:
        email_data = json.loads(email_json)
        result = classifier.classify_email(email_data)
        print("\nClassification Result:")
        print(json.dumps(result, indent=2))
    except json.JSONDecodeError:
        print("Error: Invalid JSON input. Please ensure the format is correct.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    
    # Run FastAPI server (comment out main() call to use only API)
    uvicorn.run(app, host="127.0.0.1", port=8080)
    # Uncomment below to run terminal version instead
    # main()