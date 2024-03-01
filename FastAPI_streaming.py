import os
import asyncio
from typing import Any
from typing import AsyncIterable, Awaitable
import sys
import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from queue import Queue
from pydantic import BaseModel
from langchain.llms import CTransformers
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.schema import LLMResult
from functools import lru_cache



os.environ['XDG_CACHE_HOME'] = './models/'

config = {'temperature': 0.1, 'context_length': 2048, 'gpu_layers':8,'max_new_tokens' : 512, "batch_size" : 512 }

model_id = "TheBloke/dolphin-2.6-mistral-7B-GGUF"
model_file = "dolphin-2.6-mistral-7b.Q4_K_S.gguf"
# model_id = "TheBloke/phi-2-GGUF"
# model_file = "phi-2.Q8_0.gguf"

#model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
#model_file = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"



def build_llm(model_id, model_file, config):


  llm = CTransformers(model= model_id,
                          model_file= model_file,
                          config=config,
                          threads=os.cpu_count() -1 ,
                          streaming=True,
                          callbacks = []
                        )
  
  return llm

# caching LLM
@lru_cache(maxsize=100)
def get_cached_llm():
        chat = build_llm(model_id, model_file, config)
        return chat


llm1  = get_cached_llm()

# llm2 = CTransformers(model= model_id,
#                         model_file= "dolphin-2.6-mistral-7b.Q4_K_S.gguf",
#                         config=config,
#                         threads=os.cpu_count()-1 ,
#                         callbacks = []
#                        )



class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    content: str = ""    
    def __init__(self) -> None:
        super().__init__()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
#        print(token)
        self.content += token
        # if we passed the final answer, we put tokens in queue
        sys.stdout.write(token)
        sys.stdout.flush()

        self.queue.put_nowait(token)
    
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print("ended: final Token  -  ")


async def run_call(query: str, stream_it: AsyncCallbackHandler, llm_model, stop_words):
    # assign callback handler
    llm_model.callbacks = [stream_it]
    # now query
    print("before LLM")
    await asyncio.to_thread(llm_model, query, stop_words)
    print("after LLM")

# request input format
class Query(BaseModel):
    text: str


async def wrap_done(fn: Awaitable, event: asyncio.Event):
        """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
        try:
            await fn
        except Exception as e:
            # TODO: handle exception
            print(f"Caught exception: {e}")
        finally:
            # Signal the aiter to stop.
            event.set()



async def create_gen(query: str, stream_it: AsyncCallbackHandler, llm_model, stop_words):
    task = asyncio.create_task(wrap_done(run_call(query, stream_it, llm_model, stop_words), stream_it.done))
    async for token in stream_it.aiter():
        yield token
        # await asyncio.sleep(0.2)
    print("before task")
    await task
    print("after task")



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


def auto_reply_prompt(text):
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









app = FastAPI()
# request input format
class Query(BaseModel):
    text: str



@app.get("/get_summary")
async def split_customer_text(query: Query = Body(...)):
    # few-shot example
    print(query)
    stream_it = AsyncCallbackHandler()
    customer_mail = query.text
    input_prompt = customer_query_prompt(customer_mail)
    gen = create_gen(input_prompt, stream_it, llm_model=llm1, stop_words = ["Finished.", "Finished", "finished", "Customer message:", "Queries:"])

    return StreamingResponse(gen, media_type="text/event-stream")

@app.get("/entities")
async def split_customer_text(query: Query = Body(...)):
    # few-shot example
    print(query)
    stream_it = AsyncCallbackHandler()
    customer_mail = query.text
    input_prompt = entities_prompt(customer_mail)
    gen = create_gen(input_prompt, stream_it, llm_model=llm1, stop_words = ["Finished", "finished"])

    return StreamingResponse(gen, media_type="text/event-stream")


@app.get("/auto_reply")
async def split_customer_text(query: Query = Body(...)):
    # few-shot example
    print(query)
    stream_it = AsyncCallbackHandler()
    customer_mail = query.text
    input_prompt = auto_reply_prompt(customer_mail)


    gen = create_gen(input_prompt, stream_it, llm_model=llm1, stop_words = ["Response:", "Customer message", "Note:"])

    return StreamingResponse(gen, media_type="text/event-stream")



@app.get("/generic")
async def split_customer_text(query: Query = Body(...)):
    # few-shot example
    print(query)
    stream_it = AsyncCallbackHandler()
    customer_mail = query.text

    input_prompt = "{}".format(customer_mail)

    gen = create_gen(input_prompt, stream_it, llm_model=llm1, stop_words = ["Finished", "finished"])

    return StreamingResponse(gen, media_type="text/event-stream")



if __name__ == "__main__":
    uvicorn.run("FastAPI_streaming:app", host="localhost", port=8000, reload=False,
                workers=1)


  # Load a model, start the server, and run this example in your terminal
# Choose between streaming and non-streaming mode by setting the "stream" field


