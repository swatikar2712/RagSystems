from vector_searching import VectorDBSearching
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import transformers
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoConfig, AutoTokenizer, TextIteratorStreamer
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
import threading

class RagPipeline(VectorDBSearching):

    def __init__(self):

        """
        The RagPipeline class is responsible for building a Retrieval-Augmented Generation (RAG) pipeline.
        It inherits from VectorDBSearching, which provides document preprocessing and vector search capabilities.
        This class loads a quantized LLM (Mistral-7B) using HuggingFace Transformers, embeds the documents using 
        sentence-transformers, and generates responses based on retrieved context using LangChain.
        """

        super().__init__()
        self.db = FAISS.from_documents(self.components[1], 
                          HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
        self.model_name='mistralai/Mistral-7B-Instruct-v0.1'
        self.model_config = AutoConfig.from_pretrained(self.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.use_4bit = True
        self.bnb_4bit_compute_dtype = "float16"
        self.bnb_4bit_quant_type = "nf4"
        self.use_nested_quant = False
        self.bnb_4bit_compute_dtype = "float16"
        self.compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
        self.bnb_config = BitsAndBytesConfig(
                        load_in_4bit=self.use_4bit,
                        bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                        bnb_4bit_compute_dtype=self.compute_dtype,
                        bnb_4bit_use_double_quant=self.use_nested_quant,
                    )
        self.model_llm = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            quantization_config=self.bnb_config,
                        )
        
    def Pipeline(self, query):
        """
        Executes the RAG pipeline by retrieving context from the FAISS DB and generating 
        an answer using the quantized Mistral model.
        
        Args:
            query (str): User query
        
        Returns:
            list: [retrieved context, generated response]
        """


        self.prompt_template = """
                <s>[INST]
                You are a helpful assistant. Use the information from the context below to answer the user's question.
                the context will be provided properly and there is always an answer if asked from the document. Try to answer
                the question from the provided context only.

                Context:
                {context}

                Question: {question}
                [/INST]
                """
        
   
        self.retriever = self.db.as_retriever()
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        docs = self.retriever.get_relevant_documents(query)
        context = format_docs(docs)
        prompt = self.prompt_template.format(context=context, question=query)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model_llm.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        generate_kwargs = dict(
                            inputs=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            streamer=streamer,
                            max_new_tokens=300,
                            )
        thread = threading.Thread(target=self.model_llm.generate, kwargs=generate_kwargs)
        thread.start()
        return (context, streamer)

 
    
__all__ = ["RagPipeline"]
