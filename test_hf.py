from dotenv import load_dotenv
import os

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

print(f"API Token exists: {HUGGINGFACEHUB_API_TOKEN is not None}")

try:
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    
    repo_id = "HuggingFaceH4/zephyr-7b-beta"
    print(f"\nTesting model: {repo_id}")
    
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        max_new_tokens=512
    )
    chatModel = ChatHuggingFace(llm=llm)
    
    print("Model initialized successfully!")
    
    # Test a simple query
    from langchain_core.messages import HumanMessage
    messages = [HumanMessage(content="What is diabetes?")]
    response = chatModel.invoke(messages)
    print(f"\nResponse: {response.content}")
    
except Exception as e:
    import traceback
    print(f"\nError occurred: {e}")
    print(f"\nFull traceback:\n{traceback.format_exc()}")
