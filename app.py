import os
import time
from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate 
from langchain.agents import load_tools, initialize_agent , AgentType
from langchain_experimental.agents import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from langchain.document_loaders import CSVLoader
from langchain_experimental.tools import PythonREPLTool
from langchain.chains import RetrievalQA

from langchain import hub
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from functools import lru_cache


os.environ['XDG_CACHE_HOME'] = './models/'

def make_prompt(text):
  prompt = f"""
Customer Text: Hello my name is ajay sagar and I want to change my phone number and address. please give me more time for payment of my next emi.
Customer Queries:
- Customer wants to change phone number.
- Customer wants to change address.
- Customer wants more time for next EMI payment.
Please provide quick resolution.

Customer Text:I want my Paytm postpaid account statement and want noc for the loan
Customer Queries:
- Customer wants Paytm postpaid account statment.
- Customer wants NOC for the loan
Please provide quick resolution.

Customer Text: Hi, I am trying to reach you guys from 2 days i am not able to reach. please Clear my late fees. I need to preclose the loan need pre closure letter and payment link. Regards, Lokesh.S
Customer Queries:
- Customer is requesting to clear his late fee.
- Customer wants to pre-close the loan.
- Customer needs the pre-closure letter.
Please provide quick resolution.

Customer Text: Dear Sir/mam,Kindly inhance my Paytm postpaid limit. Regard Dhiraj Kumar Mishra
Customer Queries:
- Customer is requesting credit limit increase for his Paytm postpaid account.
Please provide quick resolution.

Customer Text: {text}
Customer Queries:
"""
  return prompt


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








# model_id = "TheBloke/dolphin-2.6-mistral-7B-GGUF"
# model_file = "dolphin-2.6-mistral-7b.Q4_K_M.gguf"


# model_id = "TheBloke/dolphin-2_6-phi-2-GGUF"
# model_file = "dolphin-2_6-phi-2.Q8_0.gguf"


model_id = "stabilityai/stablelm-2-zephyr-1_6b"
model_file = "stablelm-2-zephyr-1_6b-Q4_1.gguf"


# model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
# model_file = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"



config = {'temperature': 0.1, 'context_length': 2048, 'gpu_layers':8,'max_new_tokens' : 512, "batch_size": 512}

def build_llm(model_id, config):


  llm = CTransformers(model= model_id,
                          model_file= model_file,
                          config=config,

                          threads=os.cpu_count() -2 ,
                          streaming=True,
                          callbacks = [StreamingStdOutCallbackHandler()]
                        )
  
  return llm

# caching LLM
@lru_cache(maxsize=100)
def get_cached_llm():
        chat = build_llm(model_id, config)
        return chat





prompt_template = entities_prompt('''hi there,

I want to change my address and phone number. My account number LAN id is AB32472AC . my current new phone number is 8998999898.

Also I want to get the premium receipts of my policies. please mail the at stanwar34@gmail.com

thanks!!

regards, Shashank Tanawar''')

llm = get_cached_llm()

response = llm(prompt_template, stop = ["quick resolution", "Quick Resolution"])

llm.invoke(prompt_template)



# print(prompt_template)



# for text in llm.stream(prompt_template, stop=["Finished", "finished"]):
#    print(text)



from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# streaming=True controls how openai api responds to langchain
# llm=ChatOpenAI(openai_api_key="your api key",streaming=True)
prompt=ChatPromptTemplate.from_messages([("human","{content}")])

PromptTemplate


chain=LLMChain(llm=chat,prompt=prompt_template)

response=chain.stream(input={"content":"tell me a joke"})


# this returns generator
print(response)

for res in response:
    print(res)