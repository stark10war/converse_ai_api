from llama_cpp import Llama
from guidance import models, gen, select
import guidance
import re
import json
from huggingface_hub import hf_hub_download




### Summarization template
def customer_query_prompt(text):
  prompt = f"""
Customer message: Hello my name is ajay sagar and I want to change my phone number and address. please give me more time for payment of my next emi.
Customer Queries:
- Customer wants to change phone number.
- Customer wants to change address.
- Customer wants more time for next EMI payment.
Finished.

Customer message:I want my Paytm postpaid account statement and want noc for the loan
Customer Queries:
- Customer wants Paytm postpaid account statment.
- Customer wants NOC for the loan
Finished.

Customer message: Hi, I am trying to reach you guys from 2 days i am not able to reach. please Clear my late fees. I need to preclose the loan need pre closure letter and payment link. Regards, Lokesh.S
Customer Queries:
- Customer is requesting to clear his late fee.
- Customer wants to pre-close the loan.
- Customer needs the pre-closure letter.
Finished.

Customer message: Dear Sir/mam,Kindly inhance my Paytm postpaid limit. Regard Dhiraj Kumar Mishra
Customer Queries:
- Customer is requesting credit limit increase for his Paytm postpaid account.
Finished.

Customer message: {text}
Customer Queries:
"""
  return prompt


## Entities extraction templates
def entities_prompt(text):
    prompt =  '''
List of Entitites : [Name, Phone Number, Email Mail Address, Account Number]
Instruction : Retrieved Entities should always be present in the List of Entities.

Customer Text: Hello my name is ajay sagar and I want to change my phone number and address. please give me more time for payment of my next emi.
Retrieved Entities:
{
"Name" : "Ajay Sagar",
"Phone Number" : "None",
"Email Mail Address" : "None",
"Account Number" : "None"
}
Finished Retrieving Entities

Customer Text: One of my postpaid overdue amount is active, my postpaid account is closed but the loan is active, I have already paid all those amounts respectively, this is impacting my credit score, please help with this as soon as possible. Anshuman Chaudhary 7481082179. REach out at anshu.man @yahuuu.com
Retrieved Entities:
{
"Name" : "Anshuman Chaudhary",
"Phone Number" : "7481082179",
"Email Mail Address" : "anshu.man@yahuuu.com",
"Account Number" : "None"
}
Finished Retrieving Entities

Customer Text: Dear sir, I am Chandrakant y patil account number -PYTMPPABFL67654534345 need account statement Chandrakant y patil in urgent basis
Retrieved Entities:
{
"Name" : "Chandrakant y patil",
"Phone Number" : "None",
"Email Mail Address" : "None",
"Account Number" : "PYTMPPABFL67654534345"
}
Finished Retrieving Entities'''

    add_text =  '''
Customer Text:{}
Retrieved Entities:
'''.format(text)

    promptnew = prompt + add_text
 

    return promptnew



## Auto response generic template
def auto_reply_generic_prompt(text):
    input_prompt = f'''
Just acknowledge the email with a generic response on behalf of the customer support team that we have got your request regarding the issue mentioned by customer and someone will respond shortly.
Keep the response short within 2-3 lines.

below is the response format :

Dear [CUSTOMER NAME],


[MESSAGE]


Regards,
Customer Support Team


Customer message: {text}
Response:
'''
    return input_prompt



### CRM response prompt template for only CRM intents 
def auto_reply_only_crm_prompt(text, input_data, intents_crm):

   prompt_template = f"""
Just acknowledge the customer message with a generic response on behalf of the customer support team. Thank the customer for reaching out for the customer query and use the below data to answer the query of the customer.
keep your response short and only answer "Customer queries" mentioned below.

Customer message: {text}
Customer data: {input_data}
Customer queries: {intents_crm}

below is the response format :

Dear [CUSTOMER NAME],


[MESSAGE]


Regards,
Customer Support Team


Response:


"""
   return prompt_template




# CRM response prompt template V1 for mix intents

def auto_reply_crm_mix_prompt_v1(text, input_data, intents_crm , intents_others):
   
   prompt_template = f"""
CUSTOMER MESSAGE: {text}
CUSTOMER DATA: {input_data}

for below "CUSTOMER DATA QUERIES" write a response using provided "CUSTOMER DATA".
CUSTOMER DATA QUERIES: {intents_crm}
Write in paragraph not in points.


for below "OTHER QUERIES" Just acknowledge the email with a generic response on behalf of the customer support team that we have got your request regarding the issue mentioned by customer and someone will respond shortly.
OTHER QUERIES : {intents_others}
Write in paragraph not in points.

Folow is the response Template :

Dear [CUSTOMER NAME],


Thanks for reaching out regarding [QUERIES]

["CUSTOMER DATA QUERIES" response ]

["OTHER QUERIES" response]



Regards,
Customer Support Team


Final Response:
"""
   return prompt_template




### Ajay Prompt template for CRM response V2

def auto_reply_crm_mix_prompt_v2(text, input_data, intents_crm, intents_others):
   prompt_template = f"""
Respond to the given customer message using 'Specific Queries' and 'Customer data'. Thank the customer for reaching out. Respond in 2-3 lines.

Customer message: {text}
Customer data: {input_data}
Specific Queries: {intents_crm}

Also, please confirm the receipt of given 'Generic Queries'. Avoid false information.
Use 'We have received your request to..... Our customer support team will reach you out shortly'.
Generic Queries: {intents_others}

Follow the response format provided below:

Dear [CUSTOMER NAME],


[MESSAGE]


Regards,
Customer Support Team

Response:
"""
   return prompt_template


