# web_chatbot_gradio_ready.py
import gradio as gr
from rag_system import RAGSystem

# -------------------------------
# Initialize RAG system
# -------------------------------
rag = RAGSystem(folder_path="documents")  # Change folder path if needed

# -------------------------------
# Chat history storage
# -------------------------------
# Gradio handles session state for multiple users automatically
def respond(user_input, chat_history):
    if user_input.lower() in ["exit", "quit"]:
        chat_history.append({"role": "system", "content": "Exiting chatbot. Refresh the page to start over."})
        return chat_history, ""

    # Combine last few turns for context (last 3 Q&A pairs)
    recent_history = "\n".join(
        [f"Q: {msg['content']}" if msg['role']=='user' else f"A: {msg['content']}" for msg in chat_history[-6:]]
    )
    context_query = recent_history + f"\nQ: {user_input}"

    # Get answer from RAG system
    answer = rag.answer_query(context_query, top_k=3)

    # Append new Q&A to chat history in message format
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": answer})

    return chat_history, ""

# -------------------------------
# Gradio interface
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 📚 RAG PDF Chatbot")
    gr.Markdown("Ask questions about your documents. Answers include document names and page numbers.")
    
    chatbot = gr.Chatbot(label="Chat History", type="messages")
    msg = gr.Textbox(label="Your question", placeholder="Type your question here...")
    clear = gr.Button("Clear Chat")

    # Submit user input
    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: [], None, chatbot)

# -------------------------------
# Launch web server
# -------------------------------
# server_name="0.0.0.0" allows access from outside the host machine
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
