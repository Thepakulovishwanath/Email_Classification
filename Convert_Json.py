from Json_Converter import JSONConverter

raw_text = """{
  "error": "Failed to parse JSON response",
  "raw_response": "{\n  \"classification\": {\n    \"primary_category\": \"Customer Service\",\n    \"secondary_category\": \"Customer General Query - Customer\",\n    \"confidence\": 0.8\n  },\n  \"contacts\": [],\n  \"entities\": {},\n  \"cleaned_content\": \"A system generated offer on PDF format has been sent with this email, for your working. We look forward to your kind confirmation. Mode of service: UPS WWEF FOR PALLETISED STACKABLE CARGO ONLY Effective September 9, the following changes will apply. These changes are not yet reflected in our current Service & Tariff Guide. \u2022 All UPS export and import brokerage service fees will increase by 5.1%. \u2022 The Disbursement Fee will increase by 5% and the fee applied to the advanced amount will increase by 0.5 percent points. \u2022 The Paper Commercial Invoice Services Surcharge will increase to the local currency equivalent of EUR 9.\",\n  \"summary\": \"The email confirms the receipt of an inquiry and provides a system-generated offer in PDF format. The email also announces upcoming changes to UPS export and import brokerage service fees, effective September 9.\"\n}\n\nNote: Since the email does not contain any specific information about a shipment, booking, or financial transaction, the entities section is empty. The classification is based on the general tone and content of the email, which seems to be a response to a customer inquiry."
}

 """
json_data = JSONConverter.convert(raw_text)
print(json_data) 