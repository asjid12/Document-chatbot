# main.py
import os
import tempfile
import shutil
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI # Import Qwen wrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Use Qwen embeddings
import uuid
import gradio as gr
from extractors import extract_text_from_file

# --- Load environment variables from .env file ---
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Document Chatbot API (Qwen)")

# --- Configuration ---
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Load Google API Key ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY or not GOOGLE_API_KEY.startswith("AIza"):
    raise ValueError("GOOGLE_API_KEY environment variable not set correctly.")

# --- Models (Pydantic models for API) ---
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    session_id: str

class UploadResponse(BaseModel):
    message: str
    session_id: str
    initial_task: str

# --- In-Memory Storage (Replace with DB for production) ---
session_stores = {}  # {session_id: {"task": str, "vectorstore": FAISS}}
chat_histories = {}  # {session_id: [langchain_core.messages.BaseMessage, ...]}

# --- LLM and Embeddings Setup (Qwen) ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.2)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# --- Prompts ---
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context and the initial task to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "Initial Task: {initial_task}\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# --- Helper Functions (Used by both API and Gradio) ---
def process_uploaded_files(task: str, files: List[UploadFile]) -> str:
    """Handles file saving, text extraction, chunking, embedding, and storing."""
    session_id = str(uuid.uuid4())
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(dir=UPLOAD_DIR)
        all_extracted_text = ""

        for file in files:
            if file.filename == '':
                continue
            valid_extensions = {'.pdf', '.txt', '.docx', '.html'}
            _, ext = os.path.splitext(file.filename)
            if ext.lower() not in valid_extensions:
                raise ValueError(f"Invalid file type: {ext}")

            # Save file temporarily
            file_location = os.path.join(temp_dir, file.filename)
            with open(file_location, "wb+") as file_object:
                shutil.copyfileobj(file.file, file_object)

            # Extract text
            extracted_text = extract_text_from_file(file_location)
            all_extracted_text += f"\n--- Content from {file.filename} ---\n{extracted_text}\n"

        # 2. Chunk text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = [Document(page_content=all_extracted_text)]
        splits = text_splitter.split_documents(docs) # List of Document objects

        # 3. Create embeddings and vector store using Qwen embeddings
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

        # 4. Store in session
        session_stores[session_id] = {
            "task": task,
            "vectorstore": vectorstore # Store the FAISS index
        }
        chat_histories[session_id] = [HumanMessage(content=task)] # Initialize history with task

        return session_id

    except Exception as e:
        # Clean up temp files on error
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise e # Re-raise to be handled by the caller

def chat_with_session(session_id: str, user_message: str):
    """
    Core chat logic using session data with Qwen (LangChain wrapper).
    """
    if session_id not in session_stores or session_id not in chat_histories:
        return "Error: Session not found. Please upload files first.", None

    try:
        initial_task = session_stores[session_id]["task"]
        vectorstore = session_stores[session_id]["vectorstore"]
        chat_history = chat_histories[session_id]

        # Add user message to history
        chat_history.append(HumanMessage(content=user_message))

        # Create retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        # --- Contextualize the Question ---
        # Using LangChain chain
        try:
            contextualize_q_chain = contextualize_q_prompt | llm
            contextualized_input = {
                "input": user_message,
                "chat_history": chat_history[:-1] # Exclude the just-added user message
            }
            contextualized_question_response = contextualize_q_chain.invoke(contextualized_input)
            # Access content correctly
            if hasattr(contextualized_question_response, 'content'):
                 contextualized_question = contextualized_question_response.content
            else:
                 contextualized_question = str(contextualized_question_response) # Fallback
            contextualized_question = contextualized_question.strip()

        except Exception as ctx_e:
            # Simple fallback if chain fails
            print(f"Warning: Contextualization chain failed, using fallback: {ctx_e}")
            formatted_history_for_context = ""
            for msg in chat_history[:-1]:
                role = "user" if isinstance(msg, HumanMessage) else "assistant" # Lowercase for prompt
                formatted_history_for_context += f"{role}: {msg.content}\n"
            contextualize_prompt_text = (
                f"{contextualize_q_system_prompt}\n\n"
                f"Chat History:\n{formatted_history_for_context}\n"
                f"Human: {user_message}\n"
                f"Standalone Question:"
            )
            context_response = llm.invoke(contextualize_prompt_text)
            if hasattr(context_response, 'content'):
                 contextualized_question = context_response.content
            else:
                 contextualized_question = str(context_response)
            contextualized_question = contextualized_question.strip()


        # --- Perform Retrieval ---
        docs = retriever.invoke(contextualized_question)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        # --- Prepare input for QA chain ---
        qa_chain_input = {
            "input": user_message,
            "chat_history": chat_history[:-1], # Exclude the just-added user message
            "context": context_text,
            "initial_task": initial_task
        }

        # --- Create and invoke the final QA chain ---
        try:
            qa_chain = qa_prompt | llm # Chain the prompt template with the Qwen model
            ai_msg = qa_chain.invoke(qa_chain_input)
            # Access content correctly
            if hasattr(ai_msg, 'content'):
                response_text = ai_msg.content
            else:
                response_text = str(ai_msg) # Fallback
            response_text = response_text.strip()

        except Exception as qa_e:
             # Simple fallback if chain fails
            print(f"Warning: QA chain failed, using fallback: {qa_e}")
            formatted_history_for_qa = ""
            for msg in chat_history[:-1]:
                role = "user" if isinstance(msg, HumanMessage) else "assistant" # Lowercase for prompt
                formatted_history_for_qa += f"{role}: {msg.content}\n"
            full_prompt_text = (
                f"{qa_system_prompt.format(initial_task=initial_task, context=context_text)}\n\n"
                f"Chat History:\n{formatted_history_for_qa}\n"
                f"Human: {user_message}\n"
                f"Assistant:"
            )
            qa_response = llm.invoke(full_prompt_text)
            if hasattr(qa_response, 'content'):
                response_text = qa_response.content
            else:
                response_text = str(qa_response)
            response_text = response_text.strip()


        # Add AI response to history
        chat_history.append(AIMessage(content=response_text))

        return response_text, session_id

    except Exception as e:
        error_msg = f"Error during chat with Qwen: {str(e)}"
        print(f"Chat Error: {error_msg}") # Log the error for debugging
        return error_msg, session_id # Return error message


# --- FastAPI API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def main():
    """Simple HTML form for uploading files and task."""
    content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chatbot Upload (Qwen API Test)</title>
    </head>
    <body>
        <h1>Upload Files and Task (Qwen API Test)</h1>
        <form action="/upload/" enctype="multipart/form-data" method="post">
        <label for="task">Task Prompt:</label><br>
        <textarea id="task" name="task" rows="4" cols="50" required></textarea><br><br>
        <label for="files">Choose files (PDF, TXT, DOCX, HTML):</label><br>
        <input type="file" id="files" name="files" multiple accept=".pdf,.txt,.docx,.html" required><br><br>
        <input type="submit" value="Upload and Process">
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=content)

@app.post("/upload/", response_model=UploadResponse)
async def upload_and_process_files(task: str = Form(...), files: List[UploadFile] = File(...)):
    """
    1. Accepts task prompt and files.
    2. Saves files temporarily.
    3. Extracts text content.
    4. Chunks text.
    5. Creates embeddings and stores in FAISS vectorstore.
    6. Initializes session data.
    7. Returns session ID.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    try:
        session_id = process_uploaded_files(task, files)
        return UploadResponse(
            message="Files uploaded and processed successfully.",
            session_id=session_id,
            initial_task=task
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    4. Chatbot Interaction:
        - Accepts user questions.
        - Uses session ID to retrieve context (task, vectorstore).
        - Performs retrieval using embeddings.
        - Combines task, retrieved context, chat history, and question.
        - Sends to Qwen.
        - Returns response.
    """
    response_text, sid = chat_with_session(request.session_id, request.message)
    if "Error:" in response_text and sid is None:
         raise HTTPException(status_code=404, detail=response_text)
    elif "Error:" in response_text:
         raise HTTPException(status_code=500, detail=response_text)
    # Ensure the session ID returned matches the one sent (for continuity)
    return ChatResponse(response=response_text, session_id=request.session_id)



def gradio_upload_and_process_improved(task: str, file_list: List[str]):
    """Wrapper for Gradio upload - Improved UI feedback."""
    if not task:
        gr.Warning("Please provide a task prompt.")
        return "Error: No task provided.", "", gr.update(visible=False), gr.update(visible=False)

    if not file_list:
        gr.Warning("Please upload at least one file.")
        return "Error: No files uploaded.", "", gr.update(visible=False), gr.update(visible=False)

    try:
        # --- Simulate UploadFile objects from file paths ---
        simulated_files = []
        file_names = []
        for file_path in file_list:
            if not os.path.isfile(file_path):
                 raise FileNotFoundError(f"File not found: {file_path}")
            file_names.append(os.path.basename(file_path))
            class SimFile:
                def __init__(self, path):
                    self.filename = os.path.basename(path)
                    self.file = open(path, 'rb')
            simulated_files.append(SimFile(file_path))

        # --- Process Files ---
        session_id = process_uploaded_files(task, simulated_files)
        # --- Close file handles ---
        for sim_file in simulated_files:
            sim_file.file.close()

        success_msg = f"✅ Files processed successfully!\n📁 Files: {', '.join(file_names)}\n🆔 Session ID: {session_id}"
        gr.Info("Files uploaded and processed. You can now start chatting!")
        return success_msg, session_id, gr.update(visible=True), gr.update(visible=True) # Show chat and clear sections

    except FileNotFoundError as fnf_err:
        for sim_file in simulated_files:
            try: sim_file.file.close()
            except: pass
        error_msg = f"File Error: {fnf_err}"
        gr.Error(error_msg)
        return error_msg, "", gr.update(visible=False), gr.update(visible=False)
    except Exception as e:
        # Ensure files are closed even if error occurs
        for sim_file in simulated_files:
            try: sim_file.file.close()
            except: pass
        error_msg = f"❌ Error processing files: {str(e)}"
        gr.Error(error_msg) # Show error in Gradio UI
        return error_msg, "", gr.update(visible=False), gr.update(visible=False)

def gradio_chat_improved(session_id: str, user_message: str, chat_history: List[List[str]]):
    """Wrapper for Gradio chat - Improved UI feedback."""
    if not session_id:
        gr.Warning("No session ID found. Please upload files first.")
        return "", chat_history, "No session active. Please upload files and process them first."

    if not user_message.strip():
        gr.Warning("Please enter a question.")
        return "", chat_history, "Please enter a question."

    try:
        # Call the core chat logic
        response_text, _ = chat_with_session(session_id, user_message.strip())
        # Update Gradio chat history
        updated_history = chat_history + [[user_message.strip(), response_text]]
        # Clear input box, update chat, clear status
        return "", updated_history, ""
    except Exception as e:
        error_msg = f"Chat Error: {str(e)}"
        gr.Error(error_msg)
        # Keep history, clear input, show error status
        return "", chat_history, error_msg

def reset_session():
    """Function to reset the session state and UI elements."""
    # This function helps clear the state and UI components
    # Gradio's State component handles the session_id
    # Returning None/empty updates the components
    gr.Info("Session cleared. Please upload new files.")
    return "", "", gr.update(visible=False), gr.update(visible=False), [], "" # session_id, status, chat_row, clear_row, chatbot, chat_status

# --- Improved Gradio Blocks UI ---
with gr.Blocks(title="Document Chatbot (Qwen)", theme=gr.themes.Soft()) as demo_improved: # Use a softer theme
    gr.Markdown("# 🤖 Document Chatbot ")
    gr.Markdown("### _Analyze documents based on your task using Google Qwen AI_")
    # Hidden markdown for task details from PDF
    gr.Markdown("""
    <details>
    <summary><b>📋 Task Requirements</b></summary>

    **Goal:** Create a chatbot that takes a user-defined task prompt and one or more resource files (PDF, TXT, DOCX, HTML). It uses the content and task to generate relevant responses using an LLM (now Qwen).

    **Steps:**
    1.  **User Input & Interface:** Input task description, upload files (.pdf, .txt, .docx, .html).
    2.  **Content Extraction:** Extract text from PDF, TXT, DOCX, HTML.
    3.  **Data Processing:** Chunk text, optionally create embeddings.
    4.  **Chatbot Interaction:** Ask questions related to the task and content.
    5.  **Chat History:** Use task, content snippets, question to generate response.

    **Tech Stack:** Python, FastAPI, LangChain, Qwen (Google Generative AI).
    </details>
    """, visible=False) # Start collapsed

    # State to hold the session ID
    session_state = gr.State("") # Initialize with empty string

    # --- Upload Section ---
    with gr.Accordion("📁 Upload Files & Define Task", open=True): # Use Accordion for better organization
        with gr.Row():
             with gr.Column(scale=2):
                 task_input = gr.Textbox(
                     label="🎯 Task Prompt",
                     placeholder="E.g., 'Summarize the key findings', 'Extract action items'...",
                     lines=3,
                     max_lines=5,
                     info="Describe what you want to do with the uploaded documents."
                 )
             with gr.Column(scale=1):
                 file_input = gr.File(
    file_types=[".pdf", ".txt", ".docx", ".html"],
    label="📎 Upload Files",
    file_count="multiple"
)
        with gr.Row():
            upload_btn = gr.Button("🚀 Process Files", variant="primary") # Primary button style
            clear_session_btn = gr.Button("🧹 Clear Session") # Button to clear/reset

        status_output = gr.Textbox(label="🔄 Status", interactive=False, lines=3)


    # --- Chat Section (Initially hidden) ---
    with gr.Row(visible=False) as chat_row: # Controlled by upload success
        with gr.Column():
            gr.Markdown("### 💬 Chat")
            chatbot = gr.Chatbot(
                label="Conversation History",
                bubble_full_width=False, # Make bubbles smaller
                height=400 # Set a fixed height for the chat area
            )
            msg_input = gr.Textbox(
                label="❓ Your Question",
                placeholder="Ask a question about the documents and your task...",
                lines=2
            )
            with gr.Row():
                send_btn = gr.Button("Send 📤", variant="secondary") # Secondary button
                clear_chat_btn = gr.Button("🗑️ Clear Chat") # Button to clear chat history locally

            chat_status_output = gr.Textbox(label="ℹ️ Chat Status", interactive=False, visible=False) # For chat errors/status

    # --- Linking components ---
    # Upload Button
    upload_event = upload_btn.click(
        fn=gradio_upload_and_process_improved,
        inputs=[task_input, file_input],
        outputs=[status_output, session_state, chat_row, chat_status_output] # Show chat on success
    )
    # Clear Session Button
    clear_session_btn.click(
        fn=reset_session,
        inputs=[], # No inputs needed for reset function
        outputs=[session_state, status_output, chat_row, chat_status_output, chatbot, chat_status_output],
        queue=False # Run immediately
    )

    # Chat Interaction
    chat_event = send_btn.click(
        fn=gradio_chat_improved,
        inputs=[session_state, msg_input, chatbot], # Pass session, message, and current history
        outputs=[msg_input, chatbot, chat_status_output], # Clear input, update chat, update status
        queue=True # Queue chat requests
    )
    # Submit message by pressing Enter
    msg_input.submit(
        fn=gradio_chat_improved,
        inputs=[session_state, msg_input, chatbot],
        outputs=[msg_input, chatbot, chat_status_output],
        queue=True
    )

    # Clear Chat History Button (local clear)
    clear_chat_btn.click(fn=lambda: ([], ""), outputs=[chatbot, msg_input], queue=False)



# --- Main Execution ---
if __name__ == "__main__":
    demo_improved.launch(inbrowser=True) # Launch Gradio UI in the browser
