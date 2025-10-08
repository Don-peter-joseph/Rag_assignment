# Rag_assignment
 simple AI pipeline using LangChain, LangGraph, and LangSmith to demonstrate an understanding of embeddings, vector databases, RAG


 # Instructions
 - Step 1 : Clone the repository to local.
 - Step 2 : Install the dependencies in requirement.txt  -> uv add -r requirements.txt
 - Step 3 : cd Rag_assignment
 - Step 4 : create .env file with below keys.
             huggingface_token = ""
             LANGSMITH_TRACING='true'
             LANGSMITH_ENDPOINT='https://api.smith.langchain.com'
             LANGSMITH_API_KEY=''
             LANGSMITH_PROJECT=''
             GROQ_API_KEY=''
             OPENWEATHERMAP_API_KEY=''
 - Step 5 : run the command -> streamlit run src/streamplit_app.py.

 You will be redirected streamlit ui from where you can interact with the chatbot.
