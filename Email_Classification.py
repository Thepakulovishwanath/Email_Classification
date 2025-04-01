# import json
# from typing import Dict, Any
# from datetime import datetime
# from dotenv import load_dotenv
# import os
# from langchain_groq import ChatGroq
# from langchain_core.messages import SystemMessage, HumanMessage
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import uvicorn

# # Load environment variables
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # Define Pydantic model for FastAPI input validation
# class EmailData(BaseModel):
#     subject: str
#     latestMessage: str
#     messageId: str
#     threadId: str

#     class Config:
#         extra = "allow"  # Allow extra fields if present

# # Initialize FastAPI app
# app = FastAPI()

# class EmailClassifier:
#     def __init__(self):
#         # Initialize the Groq model
#         self.llm = ChatGroq(
#             api_key=GROQ_API_KEY,
#             model_name="llama-3.2-90b-vision-preview"
#         )
        
#     def apply_character_limit(self, text: str, max_chars: int, preserve_first: int, preserve_last: int) -> str:
#         """
#         Apply character limit to text, preserving beginning and end parts.
#         """
#         if text is None:
#             return ""
#         if len(text) <= max_chars:
#             return text
#         first_part = text[:preserve_first]
#         last_part = text[-preserve_last:] if preserve_last > 0 else ""
#         separator = "\n...[content trimmed]...\n"
#         return first_part + separator + last_part
    
#     def construct_system_prompt(self) -> str:
#         """
#         Construct the system prompt for email processing.
#         """
#         return """You are an experienced Customer Support Manager at WorldZone, a Freight Forwarding Company.
# Your task is to accurately classify and extract information from emails in the freight forwarding industry.

# For each email, you will:
# 1. Classify the email into one of the predefined categories
# 2. Extract key entities and information
# 3. Clean the email by removing pleasantries and signatures
# 4. Provide a brief summary of the email

# Analyze carefully and be precise with your classifications and extracted information.
# """

#     def construct_prompt(self, email_data: Dict[str, Any]) -> str:
#         """
#         Construct the user prompt for email processing.
#         """
#         subject = email_data.get('subject', '(No subject)')
#         content = email_data.get('latestMessage', '')
#         from_addresses = email_data.get('from', [])
#         to_addresses = email_data.get('to', [])
#         date = email_data.get('date')

#         from_str = ', '.join([addr.get('email', '') for addr in from_addresses]) if from_addresses else 'Unknown'
#         to_str = ', '.join([addr.get('email', '') for addr in to_addresses]) if to_addresses else 'Unknown'
#         date_str = date.strftime("%Y-%m-%d %H:%M") if date else 'Unknown date'

#         if len(content) > 12000:
#             content = self.apply_character_limit(content, 12000, 10000, 2000)

#         prompt = f"""Analyze the following email and provide a structured response:

# EMAIL INFORMATION:
# Date: {date_str}
# From: {from_str}
# To: {to_str}
# Subject: {subject}

# EMAIL CONTENT:
# {content}
# """

#         prompt += """
# CLASSIFICATION TASK:
# Classify this email into one of these primary categories:
# 1. Booking & Quotation
# 2. Documentation
# 3. Shipment Status
# 4. Customer Service
# 5. Financial
# 6. Compliance
# 7. Social & Promotional
# 8. Other

# Then classify into one of these secondary categories:
# - Booking Quotation Request
# - Customer Booking Confirmation
# - Bill of Lading (B/L) & Documents
# - Customs Clearance Notifications
# - Delivery Order (DO) Issuance
# - Customs Documentation Requests
# - Cargo Arrival Notices
# - Proof of Delivery (POD)
# - Exception Handling & Delay Notifications
# - Shipment Status Updates
# - Customer General Query - Customer
# - Customer Escalation - Customer
# - Agent Freight Quotations
# - Rate Negotiations
# - Lead Follow-Ups
# - Invoice & Payment Requests
# - Credit Notes & Adjustments
# - Outstanding Balance Reminders
# - Insurance Claims & Notifications
# - Agent Response to Quote - Direct
# - Payment Confirmations
# - Regulatory Compliance Updates
# - Dangerous Goods (DG) Compliance
# - Social Media or other Promotional Emails
# - Generic Email - All other communication

# ENTITY EXTRACTION TASK:
# Extract the following information if present:
# 1. Contact Information:
#    - Names
#    - Email addresses
#    - Phone numbers
#    - Company names

# 2. For Booking & Quotation emails:
#    - Origin
#    - Destination
#    - Incoterms
#    - Commodity
#    - Hazardous (Y/N)
#    - Weight
#    - Container type
#    - Volume
#    - Quantity
#    - Additional services needed

# 3. For Shipment Status emails:
#    - Tracking numbers
#    - Current status
#    - Estimated arrival
#    - Any delays or issues

# 4. For Financial emails:
#    - Invoice numbers
#    - Amounts
#    - Currency
#    - Payment details

# CLEANING TASK:
# Provide a cleaned version of the email with pleasantries, signatures, and non-business content removed.

# SUMMARY TASK:
# Provide a brief summary of the email's main purpose and key points.

# RESPONSE FORMAT:
# Provide your response in valid JSON format as follows:
# {
#   "classification": {
#     "primary_category": "...",
#     "secondary_category": "...",
#     "confidence": 0.95
#   },
#   "contacts": [
#     {
#       "name": "...",
#       "email_address": "...",
#       "phone_number": "...",
#       "mobile_number": "...",
#       "designation": "...",
#       "company_name": "..."
#     }
#   ],
#   "entities": {
#     // Entity fields will depend on the email category
#   },
#   "cleaned_content": "...",
#   "summary": "..."
# }
# IMPORTANT: Do not include any explanatory text outside of the JSON structure. If certain fields have no data, include them as empty arrays [] or objects {} rather than explaining why they're empty. Make sure the response is strictly valid JSON without any additional comments or notes.
# """
#         return prompt
    
#     def classify_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Classify an email using the Groq model and return the JSON response.
#         """
#         try:
#             system_prompt = self.construct_system_prompt()
#             user_prompt = self.construct_prompt(email_data)
#             messages = [
#                 SystemMessage(content=system_prompt),
#                 HumanMessage(content=user_prompt)
#             ]
#             response = self.llm.invoke(messages)
#             return json.loads(response.content)
#         except json.JSONDecodeError:
#             return {"error": "Failed to parse JSON response", "raw_response": response.content}
#         except Exception as e:
#             return {"error": str(e)}

# # FastAPI endpoint
# @app.post("/classify")
# async def classify_email_endpoint(email: EmailData):
#     classifier = EmailClassifier()
#     email_dict = email.dict()
#     result = classifier.classify_email(email_dict)
#     if "error" in result:
#         raise HTTPException(status_code=400, detail=result["error"])
#     return result

# def main():
#     classifier = EmailClassifier()
#     print("Please paste the JSON email data below and press Enter once to submit:")
    
#     # Get single input (pasted JSON)
#     email_json = input()
    
#     # Parse and process the input
#     try:
#         email_data = json.loads(email_json)
#         result = classifier.classify_email(email_data)
#         print("\nClassification Result:")
#         print(json.dumps(result, indent=2))
#     except json.JSONDecodeError:
#         print("Error: Invalid JSON input. Please ensure the format is correct.")
#     except Exception as e:
#         print(f"Error: {str(e)}")

# if __name__ == "__main__":
    
#     # Run FastAPI server (comment out main() call to use only API)
#     uvicorn.run(app, host="127.0.0.1", port=8080)
#     # Uncomment below to run terminal version instead
#     # main()










from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from langsmith import traceable
import json
import os

# Load environment variables from .env file
load_dotenv()

# Set up LangSmith environment variables
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "Email_Classification_001")

# Other environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("model_name")

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
        # First LLM for email classification
        self.llm_classifier = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL
        )
        # Second LLM for JSON validation and transformation
        self.llm_validator = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL
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
        Construct the system prompt for email processing (first LLM).
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
        Construct the user prompt for email processing (first LLM).
        """
        subject = email_data.get('subject', '(No subject)')
        content = email_data.get('latestMessage', '')
        from_addresses = email_data.get('from', [])
        to_addresses = email_data.get('to', [])
        date = email_data.get('date')

        from_str = ', '.join([addr.get('email', '') for addr in from_addresses]) if from_addresses else 'Unknown'
        to_str = ', '.join([addr.get('email', '') for addr in to_addresses]) if to_addresses else 'Unknown'
        date_str = date.strftime("%Y-%m-d %H:%M") if date else 'Unknown date'

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
    
    @traceable(run_type="chain", name="Validate and Transform Output")
    def validate_and_transform_output(self, raw_output: str) -> Dict[str, Any]:
        """
        Use the second LLM to verify if the output is JSON and transform it if needed.
        """
        system_prompt = """
    You are a JSON validation and transformation assistant.
    Your task is to:
    1. Check if the input is valid JSON.
    2. If it is valid JSON, return it as-is.
    3. If it is not valid JSON, analyze the raw text and convert it into a valid JSON structure matching this format:
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
    Return the result in valid JSON format. If conversion fails, return an error in JSON:
    {
    "error": "Unable to transform output into valid JSON"
    }
    IMPORTANT: Do not include any explanatory text outside of the JSON structure. If certain fields have no data, include them as empty arrays [] or objects {} rather than explaining why they're empty. Make sure the response is strictly valid JSON without any additional comments or notes or any "{{\\n, \\n\\n, \\, ```}".
    """
        user_prompt = f"Input: {raw_output}"
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # LLM call will be automatically traced by LangSmith, and this method is also traced
        response = self.llm_validator.invoke(messages)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"error": "Second LLM failed to produce valid JSON", "raw_response": response.content}

    @traceable(run_type="chain", name="Classify Email")
    def classify_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify an email using the first LLM and validate/transform with the second LLM.
        """
        try:
            # First LLM generates the classification
            system_prompt = self.construct_system_prompt()
            user_prompt = self.construct_prompt(email_data)
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # LLM call will be automatically traced by LangSmith, and this method is also traced
            response = self.llm_classifier.invoke(messages)
            raw_output = response.content

            # Second LLM validates and transforms if needed
            final_result = self.validate_and_transform_output(raw_output)
            return final_result
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

# def main():
#     classifier = EmailClassifier()
#     print("Please paste the JSON email data below and press Enter once to submit:")
    
#     # Get single input (pasted JSON)
#     # email_json = input()

#     # Define source and destination folders
#     source_folder = './json_files'           # Folder with original JSON files
#     destination_folder = './processed_json'  # Folder to save the output

#     # Create the destination folder if it doesn't exist
#     os.makedirs(destination_folder, exist_ok=True)
#     # Loop through each JSON file in the source folder
#     for filename in os.listdir(source_folder):
#         if filename.endswith('.json'):
#             source_path = os.path.join(source_folder, filename)
#             print(source_folder)
            
#             # Read JSON file
#             with open(source_path, 'r') as file:
#                 try:
#                     email_json = json.load(file)
#                 except json.JSONDecodeError as e:
#                     print(f"Failed to read {filename}: {e}")
#                     continue
        
#     # Parse and process the input
#     try:
#         email_data = json.loads(email_json)
#         result = classifier.classify_email(email_data)
#         print("\nClassification Result:")


#         # Save JSON to destination folder
#         destination_path = os.path.join(destination_folder, filename)
#         with open(destination_path, 'w') as out_file:
#             json.dump(result, out_file, indent=4)

#         print(f"Processed and saved: {filename}")
#         print(json.dumps(result, indent=2))
#     except json.JSONDecodeError:
#         print("Error: Invalid JSON input. Please ensure the format is correct.")
#     except Exception as e:
#         print(f"Error: {str(e)}")

def main():
    classifier = EmailClassifier()
    print("Processing JSON files from './json_files' directory...")
    
    # Define source and destination folders
    source_folder = './json_files'           # Folder with original JSON files
    destination_folder = './processed_json'  # Folder to save the output

    # Check if source folder exists
    if not os.path.exists(source_folder):
        print(f"Error: Source folder '{source_folder}' not found. Please create it and add JSON files.")
        return

    # Create the destination folder if it doesn’t exist
    os.makedirs(destination_folder, exist_ok=True)

    # Loop through each JSON file in the source folder
    found_files = False
    for filename in os.listdir(source_folder):
        if filename.endswith('.json'):
            found_files = True
            source_path = os.path.join(source_folder, filename)
            print(f"\nProcessing: {filename}")
            
            # Read JSON file
            with open(source_path, 'r') as file:
                try:
                    email_json = json.load(file)  # Loads JSON into a dict
                except json.JSONDecodeError as e:
                    print(f"Failed to read {filename}: {e}")
                    continue
            
            # Process the email data
            try:
                result = classifier.classify_email(email_json)  # Pass dict directly
                print("Classification Result:")

                # Save JSON to destination folder
                destination_path = os.path.join(destination_folder, filename)
                with open(destination_path, 'w') as out_file:
                    json.dump(result, out_file, indent=4)

                print(f"Processed and saved: {filename}")
                print(json.dumps(result, indent=2))
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    if not found_files:
        print("No JSON files found in './json_files'. Please add some JSON files to process.")

# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    # Run FastAPI server (comment out main() call to use only API)
    # uvicorn.run(app, host="127.0.0.1", port=8080)
    # Uncomment below to run terminal version instead
    main()