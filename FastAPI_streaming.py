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
import time
import pandas as pd
from prompt_templates import customer_query_prompt, entities_prompt, auto_reply_generic_prompt, auto_reply_only_crm_prompt, auto_reply_crm_mix_prompt_v1, auto_reply_crm_mix_prompt_v2

## Llama CPP imports

from llama_cpp import Llama
from guidance import models, gen, select
import guidance
import re
import json
from huggingface_hub import hf_hub_download

os.environ['XDG_CACHE_HOME'] = './models/'

# model_id = "TheBloke/dolphin-2.6-mistral-7B-GGUF"
# model_file = "dolphin-2.6-mistral-7b.Q4_K_S.gguf"
# model_id = "TheBloke/phi-2-GGUF"
# model_file = "phi-2.Q8_0.gguf"

model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
model_file = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"




config = {'temperature': 0.1, 'context_length': 2048, 'gpu_layers':8,'max_new_tokens' : 512, "batch_size" : 512 }


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





def load_crm_data():
    crm_data = pd.read_excel("CRM Dummy Data.xlsx")
    crm_data["Due Date"] = crm_data["Due Date"].astype(str)
    crm_data["Open Date"] = crm_data["Open Date"].astype(str)
    return crm_data

def check_registered(from_email, crm_data):
    if from_email.strip().lower() in crm_data["Email ID"].str.strip().str.lower().to_list():
        return "YES"
    else:
        return "NO"


def get_data_from_email(from_email, crm_data):
    filtered_data = crm_data[crm_data["Email ID"].str.strip().str.lower() == from_email.strip().lower()]
    if filtered_data.shape[0]>0:
        json_data = filtered_data.to_dict(orient='records')[0]
    else:
        json_data= {}
    return json_data


def get_data_from_account(account_number, crm_data):
    filtered_data = crm_data[crm_data["Account number"].str.strip().str.lower() == account_number.strip().lower()]
    if filtered_data.shape[0]>0:
        json_data = filtered_data.to_dict(orient='records')[0]
    else:
        json_data= {}
    return json_data

crm_data = load_crm_data()
intent_category_mapping = pd.read_excel("intent_Categorization.xlsx")

CRM_INTENTS = intent_category_mapping[intent_category_mapping["CRM Intents"] == "YES"]["Intent"].to_list()
KNOWLEDGE_BASE_INTENTS = intent_category_mapping[intent_category_mapping["Knowledge Intents"] == "YES"]["Intent"].to_list()


app = FastAPI()

# request input format
class Query(BaseModel):
    text: str

@app.get("/get_summary")
async def get_summary(query: Query = Body(...)):
    # few-shot example
    print(query)
    stream_it = AsyncCallbackHandler()
    customer_mail = query.text
    input_prompt = customer_query_prompt(customer_mail)
    gen = create_gen(input_prompt, stream_it, llm_model=llm1, stop_words = ["Finished.", "Finished", "finished", "Customer message:", "Queries:"])

    return StreamingResponse(gen, media_type="text/event-stream")

@app.get("/entities")
async def get_entities(query: Query = Body(...)):
    # few-shot example
    print(query)
    stream_it = AsyncCallbackHandler()
    customer_mail = query.text
    input_prompt = entities_prompt(customer_mail)
    gen = create_gen(input_prompt, stream_it, llm_model=llm1, stop_words = ["Finished", "finished"])

    return StreamingResponse(gen, media_type="text/event-stream")


class AutoReply(BaseModel):
    text: str
    meta : dict

@app.get("/auto_reply")
async def get_auto_reply(query: AutoReply = Body(...)):
    # few-shot example
    print(query)
    customer_mail = query.text
    meta = query.meta

    if meta["auto_response_type"] == "HyperPersonal - CRM + KB":
        print("generate CRM Response ")
        print("generate KB Response ")
        print("Generic response for other query if there")
        input_prompt = auto_reply_crm_mix_prompt_v2(customer_mail,input_data=meta["crm_data"],intents_crm=meta["intents_crm"],intents_others=meta["intents_others"])


    elif meta["auto_response_type"] == "HyperPersonal - CRM":
        print( "Generate CRM Response")
        print("Generic response for other query if there")
        input_prompt = auto_reply_only_crm_prompt(customer_mail,input_data=meta["crm_data"],intents_crm=meta["intents_crm"])

    elif meta["auto_response_type"] == "HyperPersonal - KB":
        print("Generate KB Response")
        print("Generic response for other query if there")
        input_prompt = auto_reply_generic_prompt(customer_mail)
    
    else:
        print("Generate Generic Response for all queries")
        input_prompt =  auto_reply_generic_prompt(customer_mail)



    stream_it = AsyncCallbackHandler()
    gen = create_gen(input_prompt, stream_it, llm_model=llm1, stop_words = ["Response:", "Customer message", "Note:"])

    return StreamingResponse(gen, media_type="text/event-stream")





@app.get("/intent")
async def get_intent(query: Query = Body(...)):
    print(query)

    intents = ["Due Date Request", "Due Amount Request", "Address Change Request" ]

    return {"intent_tags": intents}




class DetailsFormat(BaseModel):
    text: str
    from_mail : str
    intent_response : list
    entities_response: dict

@app.get("/get_details")
async def get_intent(query: DetailsFormat = Body(...)):
    print(query)

    customer_mail = query.text
    predicted_intents = query.intent_response
    CRM_INTENTS1 = [i.lower().strip() for i in CRM_INTENTS ]
    KNOWLEDGE_BASE_INTENTS1 = [i.lower().strip() for i in KNOWLEDGE_BASE_INTENTS ]
    intents_crm = [i for i in predicted_intents if i.lower().strip() in CRM_INTENTS1]
    intent_kb = [i for i in predicted_intents if  i.lower().strip() in KNOWLEDGE_BASE_INTENTS1]
    intents_others = [i for i in predicted_intents if i.lower().strip() not in CRM_INTENTS1 + KNOWLEDGE_BASE_INTENTS1]

    from_mail = query.from_mail

    case_category =    list(intent_category_mapping[intent_category_mapping["Intent"].isin(predicted_intents)]["Case Category Department"].unique())
    case_assign = intent_category_mapping[intent_category_mapping["Intent"].isin(predicted_intents)][["Intent", "Assigned To"]]
    case_assign = dict(zip(case_assign['Intent'], case_assign['Assigned To']))

    timestamp_id = str(int(time.time()))
    registered = check_registered(from_mail, crm_data)


    if (registered == "YES") and (len(intents_crm)>0):
        query_crm_db = "YES"
        input_data = get_data_from_email(from_mail, crm_data)
    else:
        query_crm_db = "NO"
        input_data  = {}
    

    if len(intent_kb)>0:
        query_knowledge_base = "YES"
    else:
        query_knowledge_base = "NO"
    

    if query_crm_db == "YES" and query_knowledge_base == "YES":
        auto_response_type = "HyperPersonal - CRM + KB"
        auto_response_flg = "YES"

    elif query_crm_db== "YES" and query_knowledge_base =="NO":
        auto_response_type = "HyperPersonal - CRM"
        auto_response_flg = "YES"
    
    elif query_crm_db== "NO" and query_knowledge_base =="YES":
        auto_response_type = "HyperPersonal - KB"
        auto_response_flg = "YES"
    
    else:
        auto_response_type = "HyperPersonal- Generic"
        auto_response_flg = "YES"




    meta = {
     "case_id": timestamp_id,
    "status":"pending",
    "case_category": case_category,
    "assigned_to" : case_assign,
    "query_crm_db" : query_crm_db,
    "query_knowledge_base" : query_knowledge_base,
    "auto_response_type": auto_response_type,
    "auto_response_flg": auto_response_flg,
    "registered": registered,
    "crm_data": input_data,
    "intents_crm" : intents_crm,
    "intent_kb" : intent_kb,
    "intents_others" : intents_others
    } 

    return {"meta": meta}




@app.get("/generic")
async def generic(query: Query = Body(...)):
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


