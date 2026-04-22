import os
import gradio as gr
from groq import Groq

# ─── CONFIG ─────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL        = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = """You are a helpful, friendly, and knowledgeable AI assistant.
Answer clearly and concisely. If you don't know something, say so honestly."""
# ────────────────────────────────────────────────────────────

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY secret is missing! Add it in Space Settings → Secrets.")

client = Groq(api_key=GROQ_API_KEY)


def chat(user_message: str, history: list):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for human_msg, ai_msg in history:
        messages.append({"role": "user",      "content": human_msg})
        messages.append({"role": "assistant", "content": ai_msg})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def respond(user_message: str, history: list):
    if not user_message.strip():
        return history, ""
    bot_reply = chat(user_message, history)
    history.append((user_message, bot_reply))
    return history, ""


# ─── GRADIO UI ───────────────────────────────────────────────
CSS = """
    #chatbot { height: 500px; overflow-y: auto; }
    footer { display: none !important; }
    .title-box { text-align: center; padding: 10px 0 5px 0; }
"""

with gr.Blocks(
    title="MIAN Chatbot",
    theme=gr.themes.Soft(primary_hue="violet"),
    css=CSS
) as demo:

    gr.HTML("""
        <div class="title-box">
            <h1 style="margin:0; font-size:2rem;">⚡ MIAN Chatbot</h1>
            <p style="color:#888; margin:4px 0 0 0;">Powered by MUST · Built with Gradio</p>
        </div>
    """)

    chatbot = gr.Chatbot(elem_id="chatbot", type="tuples")

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Type your message here...",
            show_label=False,
            scale=9,
            container=False,
        )
        send_btn = gr.Button("Send ➤", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
        gr.HTML('<p style="color:#aaa; font-size:0.8rem; margin:auto 0;">Press Enter or click Send</p>')

    history_state = gr.State([])

    send_btn.click(
        fn=respond,
        inputs=[msg_box, history_state],
        outputs=[chatbot, msg_box]
    ).then(lambda h: h, inputs=[chatbot], outputs=[history_state])

    msg_box.submit(
        fn=respond,
        inputs=[msg_box, history_state],
        outputs=[chatbot, msg_box]
    ).then(lambda h: h, inputs=[chatbot], outputs=[history_state])

    clear_btn.click(
        fn=lambda: ([], [], ""),
        outputs=[chatbot, history_state, msg_box]
    )

demo.launch()
