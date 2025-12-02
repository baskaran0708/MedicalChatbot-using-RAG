from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from duckduckgo_search import DDGS
import os
import logging
from datetime import datetime


app = Flask(__name__)

# Create logs directory if it doesn't exist (BEFORE logging config)
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("Medical Chatbot Application Started")
logger.info("="*80)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
OPENROUTER_API_KEY=os.environ.get('OPENROUTER_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "medical-bot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# OpenRouter uses OpenAI-compatible API
from langchain_openai import ChatOpenAI

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

chatModel = ChatOpenAI(
    model="mistralai/mistral-7b-instruct:free",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.5,
    max_tokens=512,
    default_headers={
        "HTTP-Referer": "http://localhost:8080",
        "X-Title": "Medical Chatbot"
    }
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('index.html')

@app.route("/chat")
def chat_page():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    logger.info(f"[{request_id}] NEW REQUEST: '{msg}'")
    logger.info(f"[{request_id}] Client IP: {request.remote_addr}")
    
    # Handle casual greetings without RAG
    greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
    if msg.lower().strip() in greetings:
        response_text = "Hello! I'm your medical assistant. How can I help you today? You can ask me questions about medical conditions, treatments, and health topics."
        logger.info(f"[{request_id}] SOURCE: Greeting Handler")
        logger.info(f"[{request_id}] RESPONSE: {response_text}")
        logger.info(f"[{request_id}] REQUEST COMPLETED\n")
        return response_text
    
    try:
        # Try RAG first
        logger.info(f"[{request_id}] Attempting RAG retrieval...")
        response = rag_chain.invoke({"input": msg})
        answer = str(response["answer"])
        logger.debug(f"[{request_id}] RAG raw response: {answer[:200]}...")
        
        # Clean up model artifacts
        answer = answer.replace("[/INST]", "").replace("[INST]", "")
        answer = answer.replace("[B_INST]", "").replace("[/B_INST]", "")
        answer = answer.replace("[OUT]", "").replace("[/OUT]", "")
        answer = answer.replace("[ASS]", "").replace("[/ASS]", "")
        answer = answer.replace("<s>", "").replace("</s>", "")
        answer = answer.strip()
        
        # If answer is too short or empty, search the web
        if not answer or len(answer) < 10 or "don't have" in answer.lower() or "don't know" in answer.lower():
            logger.warning(f"[{request_id}] RAG response insufficient (length: {len(answer)})")
            logger.info(f"[{request_id}] Falling back to WEB SEARCH...")
            
            try:
                # Search DuckDuckGo for medical information with better query
                search_query = f"{msg} medical definition symptoms causes treatment"
                logger.info(f"[{request_id}] Search query: '{search_query}'")
                
                with DDGS() as ddgs:
                    results = list(ddgs.text(search_query, max_results=5))
                
                logger.info(f"[{request_id}] Found {len(results)} web search results")
                    
                if results:
                    # Log search results
                    for i, r in enumerate(results):
                        logger.debug(f"[{request_id}] Result {i+1}: {r['title']}")
                    
                    # Combine search results with more context
                    search_context = "\n\n".join([f"Source {i+1} - {r['title']}:\n{r['body']}" for i, r in enumerate(results)])
                    
                    # Ask the model to summarize the search results with better prompt
                    search_prompt = f"""You are a medical information assistant. Answer this question based on the web search results below.

Question: {msg}

Web Search Results:
{search_context}

Instructions:
1. Provide a clear, accurate answer based on the search results
2. Include key medical facts, symptoms, causes, or treatments if relevant
3. Keep the answer concise but informative (2-4 sentences)
4. If the search results don't contain relevant medical information, say "I found limited information about this topic"

Answer:"""
                    
                    logger.info(f"[{request_id}] Invoking AI to summarize web results...")
                    web_response = chatModel.invoke(search_prompt)
                    answer = str(web_response.content)
                    logger.debug(f"[{request_id}] Web AI response: {answer[:200]}...")
                    
                    # Clean up again
                    answer = answer.replace("[/INST]", "").replace("[INST]", "")
                    answer = answer.replace("[B_INST]", "").replace("[/B_INST]", "")
                    answer = answer.replace("<s>", "").replace("</s>", "")
                    answer = answer.strip()
                    
                    # Only add source note if we got a good answer
                    if answer and len(answer) > 20 and "limited information" not in answer.lower():
                        answer += "\n\n(Source: Web search)"
                        logger.info(f"[{request_id}] SOURCE: Web Search (SUCCESS)")
                        logger.info(f"[{request_id}] RESPONSE: {answer}")
                    else:
                        answer = "I couldn't find reliable medical information about that topic. Please consult a healthcare professional for accurate advice."
                        logger.warning(f"[{request_id}] SOURCE: Web Search (INSUFFICIENT)")
                        logger.info(f"[{request_id}] RESPONSE: {answer}")
                else:
                    answer = "I couldn't find specific information about that. Please consult a healthcare professional for accurate medical advice."
                    logger.warning(f"[{request_id}] SOURCE: Web Search (NO RESULTS)")
                    logger.info(f"[{request_id}] RESPONSE: {answer}")
            except Exception as search_error:
                logger.error(f"[{request_id}] Web search ERROR: {str(search_error)}")
                logger.error(f"[{request_id}] Error traceback:", exc_info=True)
                answer = "I don't have specific information about that in my knowledge base. Please consult a healthcare professional for accurate information."
                logger.info(f"[{request_id}] SOURCE: Fallback (Web search failed)")
                logger.info(f"[{request_id}] RESPONSE: {answer}")
        else:
            logger.info(f"[{request_id}] SOURCE: RAG (PDF Knowledge Base)")
            logger.info(f"[{request_id}] RESPONSE: {answer}")
        
        logger.info(f"[{request_id}] REQUEST COMPLETED\n")
        return answer
    except Exception as e:
        logger.error(f"[{request_id}] CRITICAL ERROR: {str(e)}")
        logger.error(f"[{request_id}] Error traceback:", exc_info=True)
        logger.info(f"[{request_id}] REQUEST FAILED\n")
        return f"Error: {str(e)}"



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
