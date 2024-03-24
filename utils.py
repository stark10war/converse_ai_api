from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.llms import CTransformers
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def get_embeddings_model(embed_model_id):
    device = cuda.current_device()
    embed_model = HuggingFaceEmbeddings(
        model_name= embed_model_id,
        model_kwargs= {'device': device},
        encode_kwargs= {'device': device}
    )

    return embed_model

def get_vector_db(pdf_file_path, embed_model):
    loader = PDFMinerLoader(pdf_file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=150)
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embed_model)

    return vectorstore

def get_llm(config, model_name_or_path, model_basename):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    llm = CTransformers(
    model=model_name_or_path,
    model_file = model_basename,
    config = config,
    f16_kv=True,
    callback_manager=callback_manager,
    verbose=False,
)
    return llm



from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


rag_reply_template = """
Generate an email response for the Customer Queries mentioned below. Use the relevant infromation provided in the below context.
Don't use the context as it is, Draft the mail in a well structured format with proper spaces and new lines whereever required. Don't provide any "click here", links, urls. Please Ignore them.

Context:
{context}

Customer Queries: {question}


The email response should be in below format :
Dear Customer,


[MESSAGE]


Regards,
Customer Support Team


Response:
"""





rag_reply_prompt = PromptTemplate(input_variables=["context", "question"], template=rag_reply_template)




def get_rag_pipeline(llm, vectorstore, prompt = rag_reply_prompt):
    rag_pipeline = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever= vectorstore.as_retriever(search_type="similarity_score_threshold", 
                                 search_kwargs={"score_threshold": 0.01, 
                                                "k": 2}),
        return_source_documents=True,
        chain_type_kwargs={
        "verbose": True,
            "prompt": prompt  },
    
    )

    return rag_pipeline
    

    