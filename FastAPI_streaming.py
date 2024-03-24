import os
import asyncio
from typing import Any
from typing import AsyncIterable, Awaitable
import sys
import uvicorn
from fastapi import FastAPI, Body , HTTPException
from fastapi.responses import StreamingResponse
from queue import Queue
from pydantic import BaseModel
from langchain.llms import CTransformers
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
from langchain.schema import LLMResult
from functools import lru_cache
import time
import pandas as pd
from prompt_templates import customer_query_prompt, entities_prompt, auto_reply_generic_prompt, auto_reply_only_crm_prompt, auto_reply_crm_mix_prompt_v1, auto_reply_crm_mix_prompt_v2,action_item_prompt
from prompt_templates import generate_intent_match_template, intent_queries_template, filter_intents, remove_bullet_numbers,generic_reply_template , crm_reply_template, combine_response_template

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# simple sequential chain
from langchain.chains import SimpleSequentialChain

from utils import get_embeddings_model, get_vector_db , get_rag_pipeline

import torch
import io
torch.cuda.is_available()

print(torch.cuda.device_count())


os.environ['XDG_CACHE_HOME'] = './models/'

model_id = "TheBloke/dolphin-2.6-mistral-7B-GGUF"
model_file = "dolphin-2.6-mistral-7b.Q4_K_M.gguf"

# model_id = "TheBloke/phi-2-GGUF"
# model_file = "phi-2.Q8_0.gguf"

# model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
# model_file = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"



config = {'temperature': 0.0, 'context_length': 2048, 'gpu_layers':6, "top_k" : 1, "seed" :1,
          'max_new_tokens' : 512, "batch_size": 256, "stream" : True,  "stop" :["Queries :", "queries:", "Finished", "finished"]}
# config = {'temperature': 0.1, 'context_length': 2048, 'gpu_layers':8,'max_new_tokens' : 512, "batch_size" : 512 }


def build_llm(model_id, model_file, config):


  llm = CTransformers(model= model_id,
                          model_file= model_file,
                          config=config,
                          threads=os.cpu_count() -2 ,
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

print(llm1.__dict__["client"].__dict__["_config"].__dict__)
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



embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
pdf_file_path = "FAQ_KnowledgeBase_SAMPLE.pdf"


embed_model = get_embeddings_model(embed_model_id)

vectorstore = get_vector_db(pdf_file_path, embed_model)

rag_pipeline = get_rag_pipeline(llm1, vectorstore)







async def run_call(query: str, stream_it: AsyncCallbackHandler, llm_model, stop_words):
    # assign callback handler
    llm_model.callbacks = [stream_it]
    # now query
    print("before LLM")
    await asyncio.to_thread(llm_model, query, stop_words)
    print("after LLM")


async def run_call_CHAIN(inputs, llm_chain):
    # assign callback handler
    # now query
    print("before LLM CHAIN")
    await asyncio.to_thread(llm_chain, inputs)
    print("after LLM CHAIN")


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

async def create_gen_CHAIN(inputs, stream_it: AsyncCallbackHandler, llm_chain):
    task = asyncio.create_task(wrap_done(run_call_CHAIN(inputs, llm_chain), stream_it.done))
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

global intent_category_mapping



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



@app.get("/action_items")
async def get_action_items(query: Query = Body(...)):
    # few-shot example
    print(query)
    stream_it = AsyncCallbackHandler()
    customer_mail = query.text
    input_prompt = action_item_prompt(customer_mail)
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
        input_prompt = auto_reply_crm_mix_prompt_v1(customer_mail,input_data=meta["crm_data"],intents_crm=meta["intents_crm"],intents_others=meta["intents_others"])


    elif meta["auto_response_type"] == "HyperPersonal - CRM":
        print( "Generate CRM Response")
        print("Generic response for other query if there")
        input_prompt = auto_reply_crm_mix_prompt_v1(customer_mail,input_data=meta["crm_data"],intents_crm=meta["intents_crm"], intents_others=meta["intents_others"])

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



@app.get("/auto_reply_generic")
async def get_intent(query: AutoReply = Body(...)):
    print(query)

    customer_mail = query.text
    meta = query.meta


    
    stream_it = AsyncCallbackHandler()
    llm1.callbacks =  [stream_it]

    ##### Generic Reply prompt
    generic_reply_prompt = PromptTemplate(
        input_variables=["text"],
        template= generic_reply_template)

    generic_reply_chain = LLMChain(llm=llm1, prompt=generic_reply_prompt)

    
    inputs = {"text": customer_mail}

    gen = create_gen_CHAIN(inputs, stream_it, llm_chain=generic_reply_chain)

    return StreamingResponse(gen, media_type="text/event-stream")    



@app.get("/auto_reply_others_generic")
async def get_intent(query: AutoReply = Body(...)):
    print(query)

    customer_mail = query.text
    meta = query.meta


    other_map = meta["intents_others"]
    ### input Queries for auto reply
    queries_others = list(other_map.values())
    queries_others = "\n".join(queries_others)


    ##### Generic Reply prompt
    generic_reply_prompt = PromptTemplate(
        input_variables=["text"],
        template= generic_reply_template)

    generic_reply_chain = LLMChain(llm=llm1, prompt=generic_reply_prompt)

    
    stream_it = AsyncCallbackHandler()
    llm1.callbacks =  [stream_it]
    
    
    inputs = {"text": queries_others}

    gen = create_gen_CHAIN(inputs, stream_it, llm_chain=generic_reply_chain)

    return StreamingResponse(gen, media_type="text/event-stream")    



@app.get("/auto_reply_crm")
async def get_intent(query: AutoReply = Body(...)):
    print(query)

    customer_mail = query.text
    meta = query.meta


    crm_map = meta["intents_crm"]
    ### input Queries for auto reply
    queries_crm = list(crm_map.values())
    queries_crm = "\n".join(queries_crm)
    queries_crm

    input_data = meta["crm_data"]

    ### CRM reply prompt
    prompt_crm = PromptTemplate(
        input_variables=["queries_crm","input_data"],
        template= crm_reply_template)

    crm_reply_chain = LLMChain(llm=llm1, prompt=prompt_crm)

    
    stream_it = AsyncCallbackHandler()
    llm1.callbacks =  [stream_it]

    
    inputs = {"queries_crm": queries_crm, "input_data": input_data}

    gen = create_gen_CHAIN(inputs, stream_it, llm_chain=crm_reply_chain)

    return StreamingResponse(gen, media_type="text/event-stream")    




class AutoReplyCombined(BaseModel):
    reply_crm: str
    reply_generic : str

@app.get("/auto_reply_combined")
async def get_intent(query: AutoReplyCombined = Body(...)):
    print(query)

    reply_crm = query.reply_crm
    reply_generic = query.reply_generic


    
    stream_it = AsyncCallbackHandler()
    llm1.callbacks =  [stream_it]

    ### Combined Reply Prompt
    prompt_combine_mail = PromptTemplate(
        input_variables=["reply_crm", "reply_other_intents"],
        template= combine_response_template)

    combine_mail_chain = LLMChain(llm=llm1, prompt=prompt_combine_mail)

    inputs = {"reply_crm" : reply_crm, "reply_other_intents" : reply_generic}

    gen = create_gen_CHAIN(inputs, stream_it, llm_chain=combine_mail_chain)

    return StreamingResponse(gen, media_type="text/event-stream")    



@app.get("/intent")
async def get_intent(query: Query = Body(...)):
    print(query)
    customer_mail = query.text
    
    stream_it = AsyncCallbackHandler()
    llm1.callbacks =  [stream_it]

    LIST_OF_INTENTS = intent_category_mapping["Intent"].to_list()
    intent_match_template_dynamic = generate_intent_match_template(LIST_OF_INTENTS)

    first_prompt = PromptTemplate(
        input_variables=["message_text"],
        template= intent_queries_template)

    chain_one = LLMChain(llm=llm1, prompt=first_prompt)

    second_prompt = PromptTemplate.from_template(intent_match_template_dynamic)

    chain_two = LLMChain(llm=llm1, prompt=second_prompt)

    intent_classification_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                                verbose=True
                                                )
    
    # inputs = {"input": {"message_text": customer_mail}}
    inputs = {"message_text": customer_mail}


    # gen = create_gen_CHAIN(inputs, stream_it, llm_chain=intent_classification_chain)

    # return StreamingResponse(gen, media_type="text/event-stream")    


    response = intent_classification_chain.run(input=inputs)

    final_intents, generated_intents, all_intents, query_intent_map = filter_intents(response, LIST_OF_INTENTS)

    return {"intent_tags": final_intents, "gen_intents": generated_intents, "all_predicted_intents": all_intents, "intent_map": query_intent_map}



class DetailsFormat(BaseModel):
    text: str
    from_mail : str
    intent_response : dict
    entities_response: dict

@app.get("/get_details")
async def get_intent(query: DetailsFormat = Body(...)):
    print(query)
    CRM_INTENTS = intent_category_mapping[intent_category_mapping["CRM Intents"] == "YES"]["Intent"].to_list()
    KNOWLEDGE_BASE_INTENTS = intent_category_mapping[intent_category_mapping["Knowledge Intents"] == "YES"]["Intent"].to_list()
    customer_mail = query.text
    intent_response = query.intent_response
    query_intent_map = intent_response["intent_map"]
    predicted_intents = list(query_intent_map.keys())
    CRM_INTENTS1 = [i.lower().strip() for i in CRM_INTENTS ]
    KNOWLEDGE_BASE_INTENTS1 = [i.lower().strip() for i in KNOWLEDGE_BASE_INTENTS ]

    crm_map = {}
    kb_map ={}
    other_map = {}

    for intent in query_intent_map:
        if intent.strip().lower() in CRM_INTENTS1:
            crm_map[intent] = query_intent_map[intent]
        elif intent.strip().lower() in KNOWLEDGE_BASE_INTENTS1:
            kb_map[intent] = query_intent_map[intent]
        else:
            other_map[intent] = query_intent_map[intent]

    intents_crm = list(crm_map.keys())
    intent_kb = list(kb_map.keys())
    intents_others = list(other_map.keys())

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
    "intents_crm" : crm_map,
    "intent_kb" : kb_map,
    "intents_others" : other_map
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



@app.get("/rag")
async def rag_qa(query: AutoReply = Body(...)):

    text = query.text
    meta = query.meta
    kb_map = meta["intent_kb"]
    ### input Queries for auto reply
    queries_kb = list(kb_map.values())[0]

    stream_it = AsyncCallbackHandler()
    llm1.callbacks =  [stream_it]
    rag_pipeline = get_rag_pipeline(llm1, vectorstore)
    # response = rag_pipeline(queries_kb)
    # print('----------->>>>',response)

    # inputs = {"input": {"question": text}}
    # inputs = {"question": customer_mail}

    inputs = {"query": queries_kb}


    gen = create_gen_CHAIN(inputs, stream_it, llm_chain=rag_pipeline)

    return StreamingResponse(gen, media_type="text/event-stream")    

    # return {'response': response}





@app.get("/excel/")
async def get_excel():
    try:
        global intent_category_mapping
        intent_category_mapping = pd.read_excel("intent_categorization_v2.xlsx")
        # Convert DataFrame to Excel format
        excel_bytes = io.BytesIO()
        intent_category_mapping.to_excel(excel_bytes, index=False)
        excel_bytes.seek(0)
        
        # Return the Excel file as a streaming response
        return StreamingResponse(io.BytesIO(excel_bytes.getvalue()), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": "attachment; filename=export.xlsx"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")

@app.post("/add_row/")
async def add_row(data: dict):
    try:
        global intent_category_mapping
        # Convert data to DataFrame
        new_row = pd.DataFrame([data])
        print(new_row)
        
        # Concatenate the new row with the existing DataFrame
        intent_category_mapping = pd.concat([intent_category_mapping, new_row], ignore_index=True)
        
        # Save DataFrame to Excel file
        intent_category_mapping.to_excel("intent_categorization_v2.xlsx", index=False)  # Replace "path_to_your_excel_file.xlsx" with the path to your Excel file
        
        return {"message": "Row added successfully"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")



@app.post("/delete_row/")
async def delete_row(row_data: dict):
    intent_to_delete   = row_data.get("intent_to_delete")
    print(intent_to_delete)

    try:
        global intent_category_mapping
        # Delete row from the DataFrame based on the provided row index
        intent_category_mapping = intent_category_mapping[intent_category_mapping["Intent"] != intent_to_delete]
        
        # Save the updated DataFrame to the Excel file
        intent_category_mapping.to_excel("intent_categorization_v2.xlsx", index=False)
        
        return {"message": "Row deleted successfully"}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")
    





if __name__ == "__main__":
    uvicorn.run("FastAPI_streaming:app", host="localhost", port=8000, reload=False,
                workers=1)


  # Load a model, start the server, and run this example in your terminal
# Choose between streaming and non-streaming mode by setting the "stream" field

