
### Summarization template0
def customer_query_prompt(text):
  prompt = f"""
Customer message: Hello my name is ajay sagar and I want to change my phone number and address. please give me more time for payment of my next emi.
Customer Queries:
- Customer wants to change phone number.
- Customer wants to change address.
- Customer wants more time for next EMI payment.
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
   
# CRM response prompt template V1 for mix intents
   prompt_template = f"""

CUSTOMER DATA: {input_data}

for below "CUSTOMER DATA QUERIES" write a response using provided "CUSTOMER DATA".
CUSTOMER DATA QUERIES: {intents_crm}
Write in paragraph not in points.


for below "OTHER QUERIES" Just acknowledge the email with  on behalf of the customer support team that we have got your request regarding the issue mentioned by customer and our team will reach out to you shortly.
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








####### INTENTS ################


##### Template for prompt creation - Intents

intent_queries_template = """Understand the customer message and break down into individual queries which can be addressed.

Customer Text: Hello my name is ajay sagar and I want to change my phone number and address. My account Loan Account number is AB0003455678682 and my email is ravi.mangal@gmail.com. Also Please tell me my due date and EM due amount.
Main Customer Intents:
1. Customer wants to change phone number.
2. Customer wants to change address.
3. Customer wants to know his due date.
4. Customer wants to know his due amount.
Finished.
   
Customer Text: Hi, I am trying to reach you guys from 2 days i am not able to reach. please Clear my late fees. I need to preclose the loan need pre closure letter and payment link. Regards, Lokesh.S
Main Customer Intents:
1. Customer is requesting to clear his late fee.
2. Customer wants to pre-close the loan.
3. Customer needs the pre-closure letter.
4. Customer needs the payment link.
Finished.

Customer Text: Dear Sir/Madam, Please find the below account details and provide the LOAN AGAINST PROPERTY account statement on the 1-4-2022 To 31-03-2023 to the current date. Account no : AB02546789C2 Email Id:- kisansadgir2@gmail.com Mobile Number: 8805298300 Please provide as soon as possible. It will help me alot. Sent from my iPhone
Main Customer Intents:
1. Customer is requesting for a loan account statement.

Customer Text: {message_text}
Main Customer Intents:"""




def generate_intent_match_template(LIST_OF_INTENTS):
   intents_string = ",\n".join(LIST_OF_INTENTS)

   intent_match_template1 = f"""LIST OF INTENTS :[
{intents_string}
]

Each of the given 'Queries' needs to be mapped with one of the above given intents in LIST OF INTENTS.

Queries:
1. Customer wants to pre-close the loan.
2. Customer has provided his mobile number.
3. Customer needs the pre-closure letter.

Mapped Intents:
1. Customer wants to pre-close the loan. : Closure Request
2. Customer has provided his mobile number. : Others
3. Customer needs the pre-closure letter. : Foreclosure Letter or NOC Certificate Issuance
Finished.


Queries:
1.  Customer has already paid the EMI amount through Razorpay link.
2. Customer has mentioned his account details.
3. Customer wants to change his mobile number.

Mapped Intents:
1. Customer has already paid the EMI amount through Razorpay link. : Others
2. Customer has mentioned his account details. : Others
3. Customer wants to change his mobile number. : Phone Number Change Request
Finished.

Queries:
1. Customer wants to change ECS account number and bank details.
2. Customer wants to provide his contact details for further communication.

Mapped Intents:
1. Customer wants to change ECS account number and bank details. : ECS Auto Debit Activation or Deactivation Request
2. Customer wants to provide his contact details for further communication. : Others
Finished.


Queries :{{text}}

Mapped Intents:"""

   return intent_match_template1




############# AUTO Response #################################################

generic_reply_template = '''
Just acknowledge the email with a generic response on behalf of the customer support team that we have got your request regarding the issue mentioned by customer and someone will respond shortly.
Keep the response short within 2-3 lines.

below is the response format :

Dear [CUSTOMER NAME if name is available else use "Dear Customer"],


[MESSAGE]


Regards,
Customer Support Team


Customer message: {text}
Response:
'''




crm_reply_template = f'''
Respond to customer email for the below "Customer Intents". Remember to thank the customer for his email. Use only relevant information from "Customer Data" given below to answer only the customer intents.
keep your answer short.

Customer Intents: 
{{queries_crm}}

Customer Data:
{{input_data}}


below is the response format :

Dear [CUSTOMER NAME if name is available else use "Dear Customer"],

Thanks for reaching out to us regarding ["Customer Intents"]. [MESSAGE]


Regards,
Customer Support Team


Response:
'''


generic_reply_template_others = '''
Just acknowledge the email with a generic response on behalf of the customer support team that we have got your request regarding the issue mentioned by customer and someone will respond shortly.
Keep the response short within 2-3 lines.

below is the response format :

Dear [CUSTOMER NAME] if name is available else use "Dear Customer",


[MESSAGE]


Regards,
Customer Support Team


Customer Queries: 
{queries_others}
Response:
'''



#### Combined Teamplate###############################

combine_response_template = """Below are two email responses to be sent to same customer. rephrase and Combine the below two mail responses in a single response to the customer. make it more professional.

First Mail:
{reply_crm}

Second Mail:
{reply_other_intents}

Response:
"""



### Action Items template
def action_item_prompt(text):
  prompt = f'''
Understand the key Customer mail and generate high level to do tasks for the customer support executive
don't give detailed steps just give short key tasks.

Example:

Customer mail : Dear Sir/Madam I have closed my Loan with Aditya Birla Finance Ltd With Reference to this mail, I hereby request you to provide me with a Statement of Account from the starting of LOan to Loan Closer Date. Further Please Provide me Interest Certificate of AY2023-24 AY2022-23 T

Tasks:
- Send the Statement of Account from the starting date of the loan to the closure date 
- Send Interest Certificate for AY 2023-24 and AY 2022-23.


Customer mail: {text}

Tasks:

'''

  return prompt






def filter_intents(llm_response, LIST_OF_INTENTS):

    intents = []
    queries = []
    for line in llm_response.split("\n"):
        intent = line.split(":")[-1]
        query  = line.split(":")[0]
        intents.append(intent)
        queries.append(query)

    all_intents = []
    final_intents = []
    generated_intents = []
    query_intent_map = {}
    for (intent, query) in zip(intents,queries):
        intent_list = intent.split(",")
        for item in intent_list:
            item = item.strip()
            if item not in ["", "Others"]:
                all_intents.append(item)
                if item in LIST_OF_INTENTS:
                    final_intents.append(item)
                    query_intent_map[item] = remove_bullet_numbers(query)
                    
                else:
                    generated_intents.append(item)
                    query_intent_map[item] = remove_bullet_numbers(query)
    
    if len(all_intents)==0:
        all_intents.append("Others")
        final_intents.append("Others")
        query_intent_map["Others"] = ""

    return final_intents, generated_intents, all_intents, query_intent_map


import re

def remove_bullet_numbers(text):
    # Define a regular expression pattern to match bullet numbers
    pattern = r'^\d+\.\s+'
    
    # Use re.sub() to remove the bullet numbers
    result = re.sub(pattern, '', text)
    
    return result


