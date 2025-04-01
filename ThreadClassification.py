
import os
import json
import re
import time
import hashlib
import uuid
from datetime import datetime
from collections import defaultdict
import requests
import pandas as pd
from typing import Dict, List, Any, Set, Tuple, Optional
from langsmith import traceable


# Configuration
INPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "ucf.ae")  # Directory with email JSON files
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "transaction_analysis_enhanced")
THREAD_ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "thread_analysis")  # Directory for individual thread analysis

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(THREAD_ANALYSIS_DIR, exist_ok=True)

# Output file paths
TRANSACTIONS_JSON_PATH = os.path.join(OUTPUT_DIR, "transactions.json")
EMAIL_MAPPINGS_JSON_PATH = os.path.join(OUTPUT_DIR, "email_mappings.json")
TIMELINE_JSON_PATH = os.path.join(OUTPUT_DIR, "timeline.json")
SUMMARY_JSON_PATH = os.path.join(OUTPUT_DIR, "summary.json")
EXCEL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "transaction_analysis.xlsx")
UUID_MAPPING_PATH = os.path.join(OUTPUT_DIR, "uuid_mapping.json")


class EnhancedTransactionAnalyzer:
    def __init__(self, input_dir, output_dir, thread_analysis_dir, api_key, api_url, model="llama-3.3-70b-versatile"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.thread_analysis_dir = thread_analysis_dir
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.emails = []
        self.emails_by_id = {}  # Email ID -> Email data
        self.raw_transactions = {}  # Raw transaction ID -> Transaction data (before consolidation)
        self.consolidated_transactions = {}  # Consolidated transaction ID -> Transaction data
        self.email_to_transaction_map = {}  # Email ID -> List of Transaction IDs
        self.transaction_timeline = []  # Chronological list of transaction events
        self.uuid_mapping = {}  # Transaction ID -> UUID mapping

    def load_email_files(self) -> List[Dict]:
        """Load all JSON email files from the specified directory"""
        emails = []

        print(f"Loading email files from {self.input_dir}...")

        for filename in os.listdir(self.input_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.input_dir, filename), 'r', encoding='utf-8') as f:
                        email_data = json.load(f)
                        # Add the filename and a unique ID to the data for reference
                        email_data['_filename'] = filename
                        email_data['_id'] = hashlib.md5(filename.encode()).hexdigest()
                        emails.append(email_data)
                        self.emails_by_id[email_data['_id']] = email_data
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        print(f"Loaded {len(emails)} email files")

        # Sort emails by date for chronological processing
        emails.sort(key=lambda e: e.get('email_date', ''))

        self.emails = emails
        return emails

    def test_api_connection(self):
        """Test the API connection to ensure the key is valid"""
        print("Testing API connection...")

        # Simple test prompt
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, this is a test."}
            ],
            "temperature": 0.2,
            "max_tokens": 10
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers)

            # Print detailed response for debugging
            print(f"API Response Status Code: {response.status_code}")

            if response.status_code == 200:
                print("API connection successful!")
                return True
            else:
                print(f"API Error: {response.text}")
                return False

        except Exception as e:
            print(f"Error testing API connection: {e}")
            return False

    @traceable(run_type="chain", name="analyze_email_for_transactions")
    def analyze_email_for_transactions(self, email: Dict) -> List[Dict]:
        """
        Use LLM to analyze an email and identify which transaction(s) it belongs to
        Returns a list of transaction details extracted from the email
        """
        # Extract email content
        subject = email.get('email_subject', 'No Subject')
        content = email.get('latest_message', '')
        email_id = email.get('_id', '')
        email_date = email.get('email_date', '')

        if not content:
            return []

        # Prepare the prompt for the LLM
        prompt = f"""
        Analyze this freight forwarding email and identify which transaction(s) it belongs to.

        SUBJECT: {subject}

        CONTENT:
        {content[:4000]}  # Limit content length to avoid token limits

        TASK:
        1. Identify any specific transaction identifiers (order numbers, quote references, booking numbers, etc.)
        2. Determine if this email is about a new transaction or an existing one
        3. Extract key details about the transaction (type, status, origin, destination, cargo, etc.)
        4. If the email mentions multiple transactions, identify each one separately

        FORMAT YOUR RESPONSE AS JSON:
        {{
            "transactions": [
                {{
                    "transaction_type": "order" or "quote" or "booking" or "inquiry",
                    "raw_transaction_id": "extracted identifier or generated if none exists",
                    "is_new_transaction": true/false,
                    "status": "inquiry", "quote_requested", "quote_provided", "order_placed", "in_progress", "completed", etc.,
                    "confidence": 0-100 (how confident you are this is a distinct transaction),
                    "reference_numbers": ["list", "of", "reference", "numbers"],
                    "route": {{
                        "origin": "Location or N/A",
                        "destination": "Location or N/A"
                    }},
                    "cargo_details": {{
                        "cargo_type": "Description or N/A",
                        "container_type": "Type or N/A",
                        "quantity": "Amount or N/A"
                    }},
                    "key_points": [
                        "Important point about this transaction"
                    ],
                    "summary": "Brief summary of what this email says about this transaction"
                }}
            ]
        }}

        IMPORTANT GUIDELINES:
        - If no clear transaction is identified, return an empty transactions array
        - If multiple distinct transactions are mentioned, include each as a separate object in the transactions array
        - For raw_transaction_id, extract any explicit reference numbers. If none exist, generate a descriptive ID based on the cargo, route, and date
        - The confidence score should reflect how certain you are that this is a distinct transaction
        - Include ALL reference numbers you can find (booking numbers, job numbers, container numbers, etc.) in the reference_numbers array
        """

        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system",
                 "content": "You are a freight forwarding expert who analyzes emails to identify distinct transactions (orders, quotes, bookings)."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1500,
            "response_format": {"type": "json_object"}
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Make the request
        try:
            response = requests.post(self.api_url, json=payload, headers=headers)

            # Check for errors
            if response.status_code != 200:
                print(f"API Error ({response.status_code}): {response.text}")
                return []

            result = response.json()

            # Extract the analysis
            if "choices" in result and len(result["choices"]) > 0:
                analysis_text = result["choices"][0]["message"]["content"]
                try:
                    analysis = json.loads(analysis_text)

                    # Add email_id and date to each transaction for reference
                    for transaction in analysis.get("transactions", []):
                        transaction["email_ids"] = [email_id]
                        transaction["email_dates"] = [email_date]

                        # Add dates structure
                        transaction["dates"] = {
                            "first_contact": email_date,
                            "last_update": email_date
                        }

                        # Add status history
                        transaction["status_history"] = [
                            {
                                "status": transaction.get("status", "unknown"),
                                "date": email_date,
                                "email_id": email_id
                            }
                        ]

                    return analysis.get("transactions", [])
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON response: {e}")
                    print(f"Raw response: {analysis_text}")
                    return []
            else:
                print(f"Unexpected API response format: {result}")
                return []

        except Exception as e:
            print(f"Error analyzing email: {e}")
            return []

    def process_emails(self):
        """Process all emails and identify raw transactions"""
        print("Processing emails and identifying raw transactions...")

        for i, email in enumerate(self.emails):
            email_id = email.get('_id', '')
            print(f"Processing email {i + 1}/{len(self.emails)}: {email.get('_filename', '')}")

            # Analyze the email to identify transactions
            transactions = self.analyze_email_for_transactions(email)

            # Track which transactions this email belongs to
            self.email_to_transaction_map[email_id] = []

            for transaction_data in transactions:
                raw_transaction_id = transaction_data.get("raw_transaction_id", "")

                # Clean and standardize the transaction ID
                if not raw_transaction_id:
                    # Generate a transaction ID if none was extracted
                    route = transaction_data.get("route", {})
                    origin = route.get("origin", "unknown")
                    destination = route.get("destination", "unknown")
                    cargo_details = transaction_data.get("cargo_details", {})
                    cargo = cargo_details.get("cargo_type", "unknown")
                    date_str = email.get("email_date", "").split("T")[0] if "T" in email.get("email_date",
                                                                                             "") else "unknown"

                    raw_transaction_id = f"{transaction_data.get('transaction_type', 'tx')}_{origin}_{destination}_{cargo}_{date_str}"
                    raw_transaction_id = re.sub(r'[^\w\-]', '_', raw_transaction_id)
                    transaction_data["raw_transaction_id"] = raw_transaction_id

                # Add this email to the transaction's email list
                if raw_transaction_id in self.raw_transactions:
                    # Update existing transaction
                    self.raw_transactions[raw_transaction_id]["email_ids"].append(email_id)
                    self.raw_transactions[raw_transaction_id]["email_dates"].append(email.get('email_date', ''))

                    # Update dates
                    self.raw_transactions[raw_transaction_id]["dates"]["last_update"] = email.get('email_date', '')

                    # Update status if this is a newer status
                    current_status = self.raw_transactions[raw_transaction_id].get("status", "")
                    new_status = transaction_data.get("status", "")

                    # Simple status progression logic
                    status_progression = [
                        "inquiry", "quote_requested", "quote_provided",
                        "order_placed", "in_progress", "completed"
                    ]

                    if new_status in status_progression and current_status in status_progression:
                        if status_progression.index(new_status) > status_progression.index(current_status):
                            self.raw_transactions[raw_transaction_id]["status"] = new_status

                            # Add to status history
                            self.raw_transactions[raw_transaction_id]["status_history"].append({
                                "status": new_status,
                                "date": email.get('email_date', ''),
                                "email_id": email_id
                            })

                    # Append key points and update summary
                    self.raw_transactions[raw_transaction_id]["key_points"].extend(
                        transaction_data.get("key_points", []))

                    # Remove duplicate key points
                    self.raw_transactions[raw_transaction_id]["key_points"] = list(dict.fromkeys(
                        self.raw_transactions[raw_transaction_id]["key_points"]
                    ))

                    # Update reference numbers
                    for ref in transaction_data.get("reference_numbers", []):
                        if ref not in self.raw_transactions[raw_transaction_id]["reference_numbers"]:
                            self.raw_transactions[raw_transaction_id]["reference_numbers"].append(ref)

                    # Update route and cargo details if they were N/A before
                    route = transaction_data.get("route", {})
                    cargo_details = transaction_data.get("cargo_details", {})

                    if self.raw_transactions[raw_transaction_id]["route"]["origin"] == "N/A" and route.get("origin",
                                                                                                           "N/A") != "N/A":
                        self.raw_transactions[raw_transaction_id]["route"]["origin"] = route.get("origin")

                    if self.raw_transactions[raw_transaction_id]["route"]["destination"] == "N/A" and route.get(
                            "destination", "N/A") != "N/A":
                        self.raw_transactions[raw_transaction_id]["route"]["destination"] = route.get("destination")

                    if self.raw_transactions[raw_transaction_id]["cargo_details"][
                        "cargo_type"] == "N/A" and cargo_details.get("cargo_type", "N/A") != "N/A":
                        self.raw_transactions[raw_transaction_id]["cargo_details"]["cargo_type"] = cargo_details.get(
                            "cargo_type")

                    if self.raw_transactions[raw_transaction_id]["cargo_details"][
                        "container_type"] == "N/A" and cargo_details.get("container_type", "N/A") != "N/A":
                        self.raw_transactions[raw_transaction_id]["cargo_details"][
                            "container_type"] = cargo_details.get("container_type")

                    if self.raw_transactions[raw_transaction_id]["cargo_details"][
                        "quantity"] == "N/A" and cargo_details.get("quantity", "N/A") != "N/A":
                        self.raw_transactions[raw_transaction_id]["cargo_details"]["quantity"] = cargo_details.get(
                            "quantity")
                else:
                    # Create new transaction
                    self.raw_transactions[raw_transaction_id] = transaction_data

                # Add this transaction to the email's transaction list
                self.email_to_transaction_map[email_id].append(raw_transaction_id)

            # Wait a bit to avoid rate limiting
            time.sleep(1)

    def consolidate_transactions(self):
        """Consolidate related transactions based on booking numbers and other identifiers"""
        print("Consolidating related transactions...")

        # Group by booking numbers
        booking_groups = defaultdict(list)
        job_groups = defaultdict(list)
        reference_groups = defaultdict(list)
        route_groups = defaultdict(list)

        # First, identify potential groups
        for tx_id, transaction in self.raw_transactions.items():
            # Check for booking numbers
            booking_match = None
            for ref in transaction.get("reference_numbers", []):
                if booking_match := re.search(r'(?:BKNG|booking)[^\d]*(\d{3,4})', ref, re.IGNORECASE):
                    booking_num = booking_match.group(1)
                    booking_groups[f"BKNG-{booking_num}"].append(tx_id)
                    break

            # If no booking number, check for job numbers
            if not booking_match:
                job_match = None
                for ref in transaction.get("reference_numbers", []):
                    if job_match := re.search(r'(?:JOB)[^\d]*(\d{3,4})', ref, re.IGNORECASE):
                        job_num = job_match.group(1)
                        job_groups[f"JOB-{job_num}"].append(tx_id)
                        break

            # Check for reference numbers
            for ref in transaction.get("reference_numbers", []):
                if re.match(r'(?:AKKIST|ORG|EBKG|WZKSAQ)\d+', ref, re.IGNORECASE):
                    reference_groups[ref].append(tx_id)

            # Group by route if no other identifiers
            if not booking_match and not job_match and len(reference_groups) == 0:
                origin = transaction.get("route", {}).get("origin", "N/A")
                destination = transaction.get("route", {}).get("destination", "N/A")
                container_type = transaction.get("cargo_details", {}).get("container_type", "N/A")

                if origin != "N/A" and destination != "N/A":
                    route_key = f"{origin}-{destination}-{container_type}"
                    route_groups[route_key].append(tx_id)

        # Now consolidate the transactions
        consolidated = {}

        # Process booking groups first (highest priority)
        for master_id, related_ids in booking_groups.items():
            if len(related_ids) > 0:
                # Generate a UUID for this transaction
                transaction_uuid = str(uuid.uuid4())
                self.uuid_mapping[master_id] = transaction_uuid

                consolidated_tx = self._merge_related_transactions(master_id, related_ids)
                consolidated_tx["uuid"] = transaction_uuid
                consolidated[master_id] = consolidated_tx

                # Mark these as processed
                for tx_id in related_ids:
                    self.raw_transactions[tx_id]["processed"] = True

        # Process job groups next
        for master_id, related_ids in job_groups.items():
            # Only process unprocessed transactions
            unprocessed_ids = [tx_id for tx_id in related_ids if
                               not self.raw_transactions[tx_id].get("processed", False)]
            if len(unprocessed_ids) > 0:
                # Generate a UUID for this transaction
                transaction_uuid = str(uuid.uuid4())
                self.uuid_mapping[master_id] = transaction_uuid

                consolidated_tx = self._merge_related_transactions(master_id, unprocessed_ids)
                consolidated_tx["uuid"] = transaction_uuid
                consolidated[master_id] = consolidated_tx

                # Mark these as processed
                for tx_id in unprocessed_ids:
                    self.raw_transactions[tx_id]["processed"] = True

        # Process reference groups next
        for ref, related_ids in reference_groups.items():
            # Only process unprocessed transactions
            unprocessed_ids = [tx_id for tx_id in related_ids if
                               not self.raw_transactions[tx_id].get("processed", False)]
            if len(unprocessed_ids) > 0:
                master_id = f"REF-{ref}"

                # Generate a UUID for this transaction
                transaction_uuid = str(uuid.uuid4())
                self.uuid_mapping[master_id] = transaction_uuid

                consolidated_tx = self._merge_related_transactions(master_id, unprocessed_ids)
                consolidated_tx["uuid"] = transaction_uuid
                consolidated[master_id] = consolidated_tx

                # Mark these as processed
                for tx_id in unprocessed_ids:
                    self.raw_transactions[tx_id]["processed"] = True

        # Process route groups last
        for route_key, related_ids in route_groups.items():
            # Only process unprocessed transactions
            unprocessed_ids = [tx_id for tx_id in related_ids if
                               not self.raw_transactions[tx_id].get("processed", False)]
            if len(unprocessed_ids) > 0:
                master_id = f"ROUTE-{route_key}"

                # Generate a UUID for this transaction
                transaction_uuid = str(uuid.uuid4())
                self.uuid_mapping[master_id] = transaction_uuid

                consolidated_tx = self._merge_related_transactions(master_id, unprocessed_ids)
                consolidated_tx["uuid"] = transaction_uuid
                consolidated[master_id] = consolidated_tx

                # Mark these as processed
                for tx_id in unprocessed_ids:
                    self.raw_transactions[tx_id]["processed"] = True

        # Add any remaining unprocessed transactions
        for tx_id, transaction in self.raw_transactions.items():
            if not transaction.get("processed", False):
                # Generate a UUID for this transaction
                transaction_uuid = str(uuid.uuid4())
                self.uuid_mapping[tx_id] = transaction_uuid

                consolidated_tx = transaction.copy()
                consolidated_tx["transaction_id"] = tx_id
                consolidated_tx["uuid"] = transaction_uuid
                consolidated[tx_id] = consolidated_tx

        self.consolidated_transactions = consolidated
        print(
            f"Consolidated {len(self.raw_transactions)} raw transactions into {len(consolidated)} unique transactions")

        # Save UUID mapping
        with open(UUID_MAPPING_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.uuid_mapping, f, indent=2, ensure_ascii=False)

        return consolidated

    def _merge_related_transactions(self, master_id: str, related_ids: List[str]) -> Dict:
        """Merge related transactions into a single consolidated transaction"""
        if not related_ids:
            return {}

        # Start with the first transaction as a base
        base_tx = self.raw_transactions[related_ids[0]].copy()
        base_tx["transaction_id"] = master_id
        base_tx["related_raw_transactions"] = related_ids

        # Initialize merged collections
        all_email_ids = set(base_tx.get("email_ids", []))
        all_email_dates = set(base_tx.get("email_dates", []))
        all_reference_numbers = set(base_tx.get("reference_numbers", []))
        all_key_points = set(base_tx.get("key_points", []))
        all_status_history = base_tx.get("status_history", [])

        # Merge data from all related transactions
        for tx_id in related_ids[1:]:
            tx = self.raw_transactions[tx_id]

            # Merge email IDs and dates
            all_email_ids.update(tx.get("email_ids", []))
            all_email_dates.update(tx.get("email_dates", []))

            # Merge reference numbers
            all_reference_numbers.update(tx.get("reference_numbers", []))

            # Merge key points
            all_key_points.update(tx.get("key_points", []))

            # Merge status history
            all_status_history.extend(tx.get("status_history", []))

            # Update route if it's more specific
            if base_tx["route"]["origin"] == "N/A" and tx["route"]["origin"] != "N/A":
                base_tx["route"]["origin"] = tx["route"]["origin"]

            if base_tx["route"]["destination"] == "N/A" and tx["route"]["destination"] != "N/A":
                base_tx["route"]["destination"] = tx["route"]["destination"]

            # Update cargo details if they're more specific
            if base_tx["cargo_details"]["cargo_type"] == "N/A" and tx["cargo_details"]["cargo_type"] != "N/A":
                base_tx["cargo_details"]["cargo_type"] = tx["cargo_details"]["cargo_type"]

            if base_tx["cargo_details"]["container_type"] == "N/A" and tx["cargo_details"]["container_type"] != "N/A":
                base_tx["cargo_details"]["container_type"] = tx["cargo_details"]["container_type"]

            if base_tx["cargo_details"]["quantity"] == "N/A" and tx["cargo_details"]["quantity"] != "N/A":
                base_tx["cargo_details"]["quantity"] = tx["cargo_details"]["quantity"]

            # Update dates
            if tx.get("dates", {}).get("first_contact", "") < base_tx["dates"]["first_contact"]:
                base_tx["dates"]["first_contact"] = tx["dates"]["first_contact"]

            if tx.get("dates", {}).get("last_update", "") > base_tx["dates"]["last_update"]:
                base_tx["dates"]["last_update"] = tx["dates"]["last_update"]

        # Sort and deduplicate status history
        all_status_history.sort(key=lambda x: x["date"])
        unique_status_history = []
        seen_statuses = set()

        for status_entry in all_status_history:
            status_key = f"{status_entry['status']}_{status_entry['date']}"
            if status_key not in seen_statuses:
                seen_statuses.add(status_key)
                unique_status_history.append(status_entry)

        # Update the merged transaction
        base_tx["email_ids"] = list(all_email_ids)
        base_tx["email_dates"] = list(all_email_dates)
        base_tx["reference_numbers"] = list(all_reference_numbers)
        base_tx["key_points"] = list(all_key_points)
        base_tx["status_history"] = unique_status_history

        # Determine current status (latest status in history)
        if unique_status_history:
            base_tx["status"] = unique_status_history[-1]["status"]

        # Generate a comprehensive summary
        base_tx["summary"] = self._generate_consolidated_summary(base_tx)

        return base_tx

    def _generate_consolidated_summary(self, transaction: Dict) -> str:
        """Generate a comprehensive summary for a consolidated transaction"""
        # Get basic transaction info
        tx_type = transaction.get("transaction_type", "transaction")
        origin = transaction.get("route", {}).get("origin", "unknown location")
        destination = transaction.get("route", {}).get("destination", "unknown destination")
        container_type = transaction.get("cargo_details", {}).get("container_type", "unknown container type")
        cargo_type = transaction.get("cargo_details", {}).get("cargo_type", "goods")

        # Get status info
        current_status = transaction.get("status", "unknown status")
        status_history = transaction.get("status_history", [])

        # Create a summary based on transaction type and status
        if tx_type == "booking" and current_status == "completed":
            summary = f"Completed booking for {container_type} of {cargo_type} from {origin} to {destination}."
        elif tx_type == "booking" and current_status == "in_progress":
            summary = f"In-progress booking for {container_type} of {cargo_type} from {origin} to {destination}."
        elif tx_type == "inquiry" and current_status == "quote_requested":
            summary = f"Quote requested for shipping {container_type} of {cargo_type} from {origin} to {destination}."
        elif tx_type == "inquiry" and current_status == "inquiry":
            summary = f"Initial inquiry about shipping {container_type} of {cargo_type} from {origin} to {destination}."
        elif tx_type == "order" and current_status == "order_placed":
            summary = f"Order placed for {container_type} of {cargo_type} from {origin} to {destination}."
        else:
            summary = f"{tx_type.capitalize()} for {container_type} of {cargo_type} from {origin} to {destination}, currently in {current_status} status."

        # Add status progression if available
        if len(status_history) > 1:
            first_status = status_history[0]["status"]
            last_status = status_history[-1]["status"]
            first_date = status_history[0]["date"].split("T")[0] if "T" in status_history[0]["date"] else \
            status_history[0]["date"]
            last_date = status_history[-1]["date"].split("T")[0] if "T" in status_history[-1]["date"] else \
            status_history[-1]["date"]

            summary += f" Started as {first_status} on {first_date} and progressed to {last_status} by {last_date}."

        return summary

    def map_emails_to_transactions(self):
        """Create a mapping between emails and their related transactions"""
        print("Creating email-to-transaction mappings...")

        email_mappings = {}

        for email_id, email in self.emails_by_id.items():
            # Find which consolidated transactions this email belongs to
            related_consolidated_txs = []
            related_uuids = []

            for tx_id, tx in self.consolidated_transactions.items():
                if email_id in tx.get("email_ids", []):
                    related_consolidated_txs.append(tx_id)
                    related_uuids.append(tx.get("uuid", ""))

            # Determine primary transaction (the one most relevant to this email)
            primary_tx = None
            primary_uuid = None
            if related_consolidated_txs:
                # For simplicity, use the first one as primary
                # In a more sophisticated implementation, we could use NLP to determine relevance
                primary_tx = related_consolidated_txs[0]
                primary_uuid = self.uuid_mapping.get(primary_tx, "")

            email_mappings[email_id] = {
                "email_id": email_id,
                "filename": email.get("_filename", ""),
                "date": email.get("email_date", ""),
                "subject": email.get("email_subject", ""),
                "primary_transaction": primary_tx,
                "primary_uuid": primary_uuid,
                "related_transactions": related_consolidated_txs,
                "related_uuids": related_uuids
            }

        return email_mappings

    def generate_transaction_timeline(self):
        """Generate a chronological timeline of all transaction events"""
        print("Generating transaction timeline...")

        timeline = []

        for tx_id, transaction in self.consolidated_transactions.items():
            # Get the UUID for this transaction
            tx_uuid = transaction.get("uuid", "")

            # Add status history events
            for status_change in transaction.get("status_history", []):
                timeline.append({
                    "date": status_change["date"],
                    "transaction_id": tx_id,
                    "transaction_uuid": tx_uuid,
                    "event_type": f"status_change_to_{status_change['status']}",
                    "email_id": status_change["email_id"],
                    "details": f"Status changed to {status_change['status']}"
                })

            # Add first contact event
            first_contact_date = transaction.get("dates", {}).get("first_contact", "")
            if first_contact_date:
                timeline.append({
                    "date": first_contact_date,
                    "transaction_id": tx_id,
                    "transaction_uuid": tx_uuid,
                    "event_type": "first_contact",
                    "email_id": transaction.get("email_ids", [""])[0] if transaction.get("email_ids") else "",
                    "details": f"First contact regarding {transaction.get('transaction_type', 'transaction')}"
                })

        # Sort by date
        timeline.sort(key=lambda x: x["date"])

        self.transaction_timeline = timeline
        return timeline

    def generate_transaction_summary(self) -> Dict:
        """Generate a summary of all transactions"""
        print("Generating transaction summary...")

        summary = {
            "total_transactions": len(self.consolidated_transactions),
            "transaction_types": defaultdict(int),
            "transaction_statuses": defaultdict(int),
            "origins": defaultdict(int),
            "destinations": defaultdict(int),
            "container_types": defaultdict(int),
            "date_range": {
                "start": None,
                "end": None
            },
            "analysis_date": datetime.now().isoformat()
        }

        all_dates = []

        for transaction_id, transaction in self.consolidated_transactions.items():
            # Count transaction types
            transaction_type = transaction.get("transaction_type", "unknown")
            summary["transaction_types"][transaction_type] += 1

            # Count transaction statuses
            status = transaction.get("status", "unknown")
            summary["transaction_statuses"][status] += 1

            # Count origins and destinations
            origin = transaction.get("route", {}).get("origin", "unknown")
            destination = transaction.get("route", {}).get("destination", "unknown")
            container_type = transaction.get("cargo_details", {}).get("container_type", "unknown")

            if origin and origin != "N/A" and origin != "unknown":
                summary["origins"][origin] += 1

            if destination and destination != "N/A" and destination != "unknown":
                summary["destinations"][destination] += 1

            if container_type and container_type != "N/A" and container_type != "unknown":
                summary["container_types"][container_type] += 1

            # Collect dates for date range
            for date_str in transaction.get("email_dates", []):
                if date_str:
                    all_dates.append(date_str)

        # Set date range
        if all_dates:
            summary["date_range"]["start"] = min(all_dates)
            summary["date_range"]["end"] = max(all_dates)

        return summary

    def analyze_individual_thread(self, transaction_id: str, transaction: Dict):
        """Perform detailed analysis on an individual transaction thread"""
        print(f"Analyzing thread: {transaction_id}")

        # Get the UUID for this transaction
        tx_uuid = transaction.get("uuid", "")

        # Sanitize transaction_id for file path (replace invalid characters)
        safe_transaction_id = re.sub(r'[\\/*?:"<>|]', '_', transaction_id)

        # Create a directory for this thread
        thread_dir = os.path.join(self.thread_analysis_dir, f"{safe_transaction_id}_{tx_uuid[:8]}")
        os.makedirs(thread_dir, exist_ok=True)

        # Get all emails for this thread
        email_ids = transaction.get("email_ids", [])
        emails = [self.emails_by_id[email_id] for email_id in email_ids if email_id in self.emails_by_id]

        # Sort emails by date
        emails.sort(key=lambda e: e.get('email_date', ''))

        # Create a thread summary
        thread_summary = {
            "transaction_id": transaction_id,
            "transaction_uuid": tx_uuid,
            "type": transaction.get("transaction_type", ""),
            "status": transaction.get("status", ""),
            "email_count": len(emails),
            "date_range": {
                "start": transaction.get("dates", {}).get("first_contact", ""),
                "end": transaction.get("dates", {}).get("last_update", "")
            },
            "route": transaction.get("route", {}),
            "cargo_details": transaction.get("cargo_details", {}),
            "reference_numbers": transaction.get("reference_numbers", []),
            "summary": transaction.get("summary", ""),
            "key_points": transaction.get("key_points", []),
            "status_history": transaction.get("status_history", []),
            "emails": []
        }

        # Add email details
        for email in emails:
            thread_summary["emails"].append({
                "email_id": email.get("_id", ""),
                "filename": email.get("_filename", ""),
                "date": email.get("email_date", ""),
                "subject": email.get("email_subject", ""),
                "from": email.get("from_address", ""),
                "to": email.get("to_address", ""),
                "content_snippet": email.get("latest_message", "")[:200] + "..." if len(
                    email.get("latest_message", "")) > 200 else email.get("latest_message", "")
            })

        # Save thread summary
        thread_summary_path = os.path.join(thread_dir, "thread_summary.json")
        with open(thread_summary_path, 'w', encoding='utf-8') as f:
            json.dump(thread_summary, f, indent=2, ensure_ascii=False)

        # Create a thread timeline
        thread_timeline = []

        # Add status history events
        for status_change in transaction.get("status_history", []):
            thread_timeline.append({
                "date": status_change["date"],
                "event_type": f"status_change_to_{status_change['status']}",
                "email_id": status_change["email_id"],
                "details": f"Status changed to {status_change['status']}"
            })

        # Add email events
        for email in emails:
            thread_timeline.append({
                "date": email.get("email_date", ""),
                "event_type": "email",
                "email_id": email.get("_id", ""),
                "details": f"Email: {email.get('email_subject', '')}"
            })

        # Sort timeline by date
        thread_timeline.sort(key=lambda x: x["date"])

        # Save thread timeline
        thread_timeline_path = os.path.join(thread_dir, "thread_timeline.json")
        with open(thread_timeline_path, 'w', encoding='utf-8') as f:
            json.dump({"timeline": thread_timeline}, f, indent=2, ensure_ascii=False)

        # Create a thread Excel report
        thread_excel_path = os.path.join(thread_dir, "thread_analysis.xlsx")
        writer = pd.ExcelWriter(thread_excel_path, engine='xlsxwriter')

        # Summary sheet
        summary_data = {
            "Metric": [
                "Transaction ID",
                "Transaction UUID",
                "Type",
                "Status",
                "Email Count",
                "First Contact",
                "Last Update",
                "Origin",
                "Destination",
                "Cargo Type",
                "Container Type",
                "Quantity",
                "Reference Numbers",
                "Summary"
            ],
            "Value": [
                transaction_id,
                tx_uuid,
                transaction.get("transaction_type", ""),
                transaction.get("status", ""),
                len(emails),
                transaction.get("dates", {}).get("first_contact", ""),
                transaction.get("dates", {}).get("last_update", ""),
                transaction.get("route", {}).get("origin", ""),
                transaction.get("route", {}).get("destination", ""),
                transaction.get("cargo_details", {}).get("cargo_type", ""),
                transaction.get("cargo_details", {}).get("container_type", ""),
                transaction.get("cargo_details", {}).get("quantity", ""),
                ", ".join(transaction.get("reference_numbers", [])),
                transaction.get("summary", "")
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Emails sheet
        email_data = []
        for email in emails:
            email_data.append({
                "Date": email.get("email_date", ""),
                "Subject": email.get("email_subject", ""),
                "From": email.get("from_address", ""),
                "To": email.get("to_address", ""),
                "Filename": email.get("_filename", "")
            })

        email_df = pd.DataFrame(email_data)
        email_df.to_excel(writer, sheet_name='Emails', index=False)

        # Timeline sheet
        timeline_data = []
        for event in thread_timeline:
            timeline_data.append({
                "Date": event.get("date", ""),
                "Event Type": event.get("event_type", ""),
                "Details": event.get("details", "")
            })

        timeline_df = pd.DataFrame(timeline_data)
        timeline_df.to_excel(writer, sheet_name='Timeline', index=False)

        # Key Points sheet
        key_points_data = []
        for point in transaction.get("key_points", []):
            key_points_data.append({
                "Key Point": point
            })

        key_points_df = pd.DataFrame(key_points_data)
        key_points_df.to_excel(writer, sheet_name='Key Points', index=False)

        # Save Excel file
        writer.close()

        return thread_dir

    def save_results(self):
        """Save the analysis results to JSON and Excel files"""
        print("Saving analysis results...")

        # Save consolidated transactions JSON
        with open(TRANSACTIONS_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump({"transactions": self.consolidated_transactions}, f, indent=2, ensure_ascii=False)

        # Save email mappings JSON
        email_mappings = self.map_emails_to_transactions()
        with open(EMAIL_MAPPINGS_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump({"email_mappings": email_mappings}, f, indent=2, ensure_ascii=False)

        # Save timeline JSON
        timeline = self.generate_transaction_timeline()
        with open(TIMELINE_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump({"timeline": timeline}, f, indent=2, ensure_ascii=False)

        # Generate and save summary
        summary = self.generate_transaction_summary()
        with open(SUMMARY_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Create Excel file with multiple sheets
        writer = pd.ExcelWriter(EXCEL_OUTPUT_PATH, engine='xlsxwriter')

        # Transactions sheet
        transactions_data = []
        for transaction_id, transaction in self.consolidated_transactions.items():
            transactions_data.append({
                "Transaction ID": transaction_id,
                "UUID": transaction.get("uuid", ""),
                "Type": transaction.get("transaction_type", ""),
                "Status": transaction.get("status", ""),
                "Origin": transaction.get("route", {}).get("origin", ""),
                "Destination": transaction.get("route", {}).get("destination", ""),
                "Cargo Type": transaction.get("cargo_details", {}).get("cargo_type", ""),
                "Container Type": transaction.get("cargo_details", {}).get("container_type", ""),
                "Quantity": transaction.get("cargo_details", {}).get("quantity", ""),
                "First Contact": transaction.get("dates", {}).get("first_contact", ""),
                "Last Update": transaction.get("dates", {}).get("last_update", ""),
                "Email Count": len(transaction.get("email_ids", [])),
                "Reference Numbers": ", ".join(transaction.get("reference_numbers", [])),
                "Summary": transaction.get("summary", "")
            })

        transactions_df = pd.DataFrame(transactions_data)
        transactions_df.to_excel(writer, sheet_name='Transactions', index=False)

        # Timeline sheet
        timeline_data = []
        for event in timeline:
            timeline_data.append({
                "Date": event.get("date", ""),
                "Transaction ID": event.get("transaction_id", ""),
                "Transaction UUID": event.get("transaction_uuid", ""),
                "Event Type": event.get("event_type", ""),
                "Details": event.get("details", "")
            })

        timeline_df = pd.DataFrame(timeline_data)
        timeline_df.to_excel(writer, sheet_name='Timeline', index=False)

        # Email Mappings sheet
        email_data = []
        for email_id, mapping in email_mappings.items():
            email_data.append({
                "Email ID": email_id,
                "Filename": mapping.get("filename", ""),
                "Date": mapping.get("date", ""),
                "Subject": mapping.get("subject", ""),
                "Primary Transaction": mapping.get("primary_transaction", ""),
                "Primary UUID": mapping.get("primary_uuid", ""),
                "Related Transactions": ", ".join(mapping.get("related_transactions", []))
            })

        email_df = pd.DataFrame(email_data)
        email_df.to_excel(writer, sheet_name='Email Mappings', index=False)

        # Summary sheet
        summary_data = {
            "Metric": [
                "Total Transactions",
                "Date Range Start",
                "Date Range End",
                "Analysis Date"
            ],
            "Value": [
                summary["total_transactions"],
                summary["date_range"]["start"],
                summary["date_range"]["end"],
                summary["analysis_date"]
            ]
        }

        # Add transaction types
        for tx_type, count in summary["transaction_types"].items():
            summary_data["Metric"].append(f"Transaction Type: {tx_type}")
            summary_data["Value"].append(count)

        # Add transaction statuses
        for status, count in summary["transaction_statuses"].items():
            summary_data["Metric"].append(f"Status: {status}")
            summary_data["Value"].append(count)

        # Add origins
        for origin, count in summary["origins"].items():
            summary_data["Metric"].append(f"Origin: {origin}")
            summary_data["Value"].append(count)

        # Add destinations
        for dest, count in summary["destinations"].items():
            summary_data["Metric"].append(f"Destination: {dest}")
            summary_data["Value"].append(count)

        # Add container types
        for container, count in summary["container_types"].items():
            summary_data["Metric"].append(f"Container: {container}")
            summary_data["Value"].append(count)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Transaction Details sheet with key points
        details_data = []
        for transaction_id, transaction in self.consolidated_transactions.items():
            key_points = transaction.get("key_points", [])
            key_points_str = "\n".join([f"• {point}" for point in key_points])

            details_data.append({
                "Transaction ID": transaction_id,
                "UUID": transaction.get("uuid", ""),
                "Type": transaction.get("transaction_type", ""),
                "Status": transaction.get("status", ""),
                "Key Points": key_points_str,
                "Summary": transaction.get("summary", "")
            })

        details_df = pd.DataFrame(details_data)
        details_df.to_excel(writer, sheet_name='Transaction Details', index=False)

        # Save Excel file
        writer.close()

        # Analyze individual threads
        thread_dirs = []
        for transaction_id, transaction in self.consolidated_transactions.items():
            thread_dir = self.analyze_individual_thread(transaction_id, transaction)
            thread_dirs.append(thread_dir)

        print(f"Results saved to {self.output_dir}")
        print(f"Transactions JSON: {TRANSACTIONS_JSON_PATH}")
        print(f"Email Mappings JSON: {EMAIL_MAPPINGS_JSON_PATH}")
        print(f"Timeline JSON: {TIMELINE_JSON_PATH}")
        print(f"Summary JSON: {SUMMARY_JSON_PATH}")
        print(f"Excel report: {EXCEL_OUTPUT_PATH}")
        print(f"Individual thread analyses saved to {self.thread_analysis_dir}")
        print(f"UUID mapping saved to {UUID_MAPPING_PATH}")

    def print_transaction_summaries(self):
        """Print a summary of each consolidated transaction to the console"""
        print("\n" + "=" * 80)
        print(f"CONSOLIDATED TRANSACTION SUMMARIES ({len(self.consolidated_transactions)} transactions)")
        print("=" * 80)

        for transaction_id, transaction in self.consolidated_transactions.items():
            print(f"\nTransaction ID: {transaction_id}")
            print(f"UUID: {transaction.get('uuid', 'No UUID')}")
            print(f"Type: {transaction.get('transaction_type', 'Unknown')}")
            print(f"Status: {transaction.get('status', 'Unknown')}")
            print(f"Emails: {len(transaction.get('email_ids', []))}")
            print(
                f"Date Range: {transaction.get('dates', {}).get('first_contact', 'Unknown')} to {transaction.get('dates', {}).get('last_update', 'Unknown')}")
            print(
                f"Route: {transaction.get('route', {}).get('origin', 'Unknown')} → {transaction.get('route', {}).get('destination', 'Unknown')}")
            print(f"Cargo: {transaction.get('cargo_details', {}).get('cargo_type', 'Unknown')}")
            print(f"Container: {transaction.get('cargo_details', {}).get('container_type', 'Unknown')}")
            print(f"Reference Numbers: {', '.join(transaction.get('reference_numbers', []))}")
            print(f"Summary: {transaction.get('summary', 'No summary available')}")

            print("\nKey Points:")
            for point in transaction.get("key_points", []):
                print(f"  • {point}")

            print("\nStatus History:")
            for status in transaction.get("status_history", []):
                print(f"  • {status['date']}: {status['status']}")

            print("-" * 80)

    def run(self):
        """Run the complete analysis process"""
        print("Enhanced Transaction Thread Analyzer")
        print("===================================")

        # Test API connection first
        if not self.test_api_connection():
            print("API connection failed. Please check your API key and try again.")
            return

        # Load email files
        self.load_email_files()

        if not self.emails:
            print("No emails found. Please check the input directory.")
            return

        # Process emails to identify raw transactions
        self.process_emails()

        # Consolidate related transactions
        self.consolidate_transactions()

        # Print transaction summaries
        self.print_transaction_summaries()

        # Save results
        self.save_results()

        # Print summary
        summary = self.generate_transaction_summary()
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Total Unique Transactions: {summary['total_transactions']}")
        print("\nTransaction Types:")
        for tx_type, count in summary["transaction_types"].items():
            print(f"  - {tx_type}: {count}")

        print("\nTransaction Statuses:")
        for status, count in summary["transaction_statuses"].items():
            print(f"  - {status}: {count}")

        print("\nTop Origins:")
        for origin, count in sorted(summary["origins"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {origin}: {count}")

        print("\nTop Destinations:")
        for dest, count in sorted(summary["destinations"].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {dest}: {count}")

        print("\nAnalysis Complete!")


def main():
    """Main function"""
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("python-dotenv not installed. Environment variables will not be loaded from .env file.")
        print("Install with: pip install python-dotenv")

    # API configuration
    api_url = "https://api.groq.com/openai/v1/chat/completions"

    # Get API key from environment variable or user input
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        api_key = input("Please enter your Groq API key: ")

    # Create and run the analyzer
    analyzer = EnhancedTransactionAnalyzer(INPUT_DIR, OUTPUT_DIR, THREAD_ANALYSIS_DIR, api_key, api_url)
    analyzer.run()


if __name__ == "__main__":
    main()

 