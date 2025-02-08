
"""
AI Chatbot Project - From Concept to Deployment
"""

# %% [1] Install Required Libraries
!pip install torch transformers gradio sentencepiece accelerate

# %% [2] Import Libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import random
import time

# %% [3] Initialize Model and Tokenizer
class ChatBot:
    def __init__(self):
        # Load pretrained GPT-2 model
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        self.chat_history = []
        
        # Add special tokens for better conversation
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def generate_response(self, user_input):
        """Generate AI response using GPT-2"""
        try:
            # Encode user input with chat history
            input_text = "\n".join(self.chat_history[-4:] + [f"User: {user_input}"])
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            
            # Generate response with adjusted parameters
            outputs = self.model.generate(
                inputs,
                max_length=200,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode and format response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            ai_response = response[len(input_text):].split("User:")[0].strip()
            
            # Update chat history
            self.chat_history.append(f"User: {user_input}")
            self.chat_history.append(f"AI: {ai_response}")
            
            return ai_response
            
        except Exception as e:
            return f"Error: {str(e)}"

# %% [4] Create Gradio Interface
def create_interface():
    bot = ChatBot()
    
    with gr.Blocks(css=".gradio-container {background: #f0f2f6}") as demo:
        gr.Markdown("# School Project AI Chatbot 🤖")
        
        with gr.Row():
            chatbot = gr.Chatbot(label="Conversation History")
            user_input = gr.Textbox(label="Your Message")
            
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear Chat")
        
        def respond(message, chat_history):
            bot_response = bot.generate_response(message)
            chat_history.append((message, bot_response))
            return "", chat_history
        
        submit_btn.click(
            respond,
            [user_input, chatbot],
            [user_input, chatbot]
        )
        
        user_input.submit(
            respond,
            [user_input, chatbot],
            [user_input, chatbot]
        )
        
        clear_btn.click(lambda: [], None, chatbot)
    
    return demo

# %% [5] Deployment Setup
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)  # Set share=False for local deployment
