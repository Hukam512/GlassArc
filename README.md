# GlassArc – AI Assistant with Live Web Search

GlassArc is a **single‑file, self‑contained AI assistant** that combines a local large language model (LLM) with a **zero‑API‑key web search engine**.  
It can run in your terminal or as a lightweight web application – no Gradio, no restart loops, no format errors.

---

## ✨ Features

- **Local AI Chat** – Uses any GGUF model (Qwen2.5‑3B by default) via `llama-cpp-python`.  
- **Live Web Search** – Scrapes DuckDuckGo + Google in parallel, with rotating user‑agents and polite delays.  
- **Huge Prompt Support** – Context window of **8192 tokens** (~20 000 characters). Inputs that exceed the limit are automatically split and summarised.  
- **CRC Integrity** – Stores CRC32 checksums for all generated text (same algorithm used in ZIP files).  
- **Full Trace Logging** – Every step (search, extraction, model call) is logged to `glassarc_trace.log`.  
- **Slow‑Operation Tracking** – Operations that take longer than 60 seconds are written to `slow_ops.log`.  
- **Two Interfaces**  
  - **Terminal** – type `/web <query>` for live search, or just chat with the AI.  
  - **Web UI** – a clean, single‑page Flask app that opens in your browser (no Gradio).  
- **Auto‑Model Selection** – Automatically picks the best available model (Qwen 3B or TinyLlama 1.1B) based on free RAM.  
- **No External APIs** – The web search uses public DuckDuckGo and Google HTML pages; no API keys are required.  
- **Self‑Cleaning** – All network sessions and thread pools are properly closed after use.  

---

## 🚀 Quick Start

1. **Clone / download** the `glassarc_safe.py` file into an empty folder.  
2. **Install dependencies** (if you don’t already have them):

   ```bash
   pip install llama-cpp-python requests beautifulsoup4 trafilatura fake-useragent flask
Download a GGUF model (if you haven’t already). The script expects a model in the models/ folder.
For example, download Qwen2.5‑3B‑Instruct (Q4_K_M):

bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen2.5-3B-Instruct-GGUF', 'qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='models')"
Or use any other GGUF file – just make sure it ends up in models/.

Run the assistant:

bash
# Terminal chat
python glassarc_safe.py

# Web interface (opens browser automatically)
python glassarc_safe.py --web
📋 Usage
Terminal Chat
text
python glassarc_safe.py
Type any message and the AI will reply.

Web search: type /web your query (e.g., /web mql4 tutorial) to get live results from DuckDuckGo + Google.

Exit: type exit or press Ctrl+C.

Web Interface
text
python glassarc_safe.py --web
Opens http://127.0.0.1:5000 in your default browser.

The text area accepts massive pastes – use Ctrl+Enter to send.

Web search: start your message with /web (e.g., /web latest AI news).

🧩 Commands
Command	Description
/web query	Search DuckDuckGo + Google for query. Shows title, URL, snippet, and extracted content (if available).
exit / quit	End the terminal session.
Inside the web UI, just type your message or /web query and press Ctrl+Enter to send.

⚙️ How It Works
AI Chat
The GGUF model is loaded with 8192 context tokens and 4 threads (adjustable inside the script).

Every user prompt is formatted with the ChatML template so that Qwen‑based models understand the instruction clearly.

If a prompt is too long, it is split into chunks of ~2000 words, each chunk is summarised by the model, and the summaries are combined into a final answer.

Web Search
Two threads scrape DuckDuckGo and Google simultaneously.

Results are deduplicated by URL.

If fetch_text=True (default), the script extracts up to 3000 characters of clean content from each result page using trafilatura.

No API keys – the search engines are accessed via their public HTML interfaces.

Rotating user‑agents (via fake-useragent) and polite delays help avoid being blocked.

Logging
glassarc_trace.log – every function call (search, extraction, model call) is logged with timestamps.

slow_ops.log – any operation that takes longer than 60 seconds is recorded here.

CRC32 checksums are computed for all stored text, enabling verification of data integrity (like ZIP files).

🧠 Model Management
The script automatically selects the best available model:

If models/qwen2.5-3b-instruct-q4_k_m.gguf exists and your system has > 4 GB free RAM, it uses that model.

Otherwise, it falls back to models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (which has only 22 layers, often referred to as a “20‑layer” model).

If neither model is found, the script exits with an error.

You can also manually swap models while the script is running by replacing the file in models/ and restarting.

🔧 Configuration
Inside glassarc_safe.py, you can adjust these variables (near the top of the file):

N_CTX – context size in tokens (default 8192). Increase it if your GPU/RAM allows.

Threads – n_threads=4 inside the Llama constructor. Match it to your CPU core count.

Model path – MODEL_PATH can be changed to point to any GGUF file.

🛟 Troubleshooting
“No model found in models/ folder!”
Make sure you placed the GGUF file inside the models/ folder.

The script expects either qwen2.5-3b-instruct-q4_k_m.gguf or tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf. You can change the names in the script if you have a different file.

Web search returns no results
Your IP might be rate‑limited by Google. The script already uses delays and rotating user‑agents.

If DuckDuckGo works but Google doesn’t, the script will still show the DuckDuckGo results.

You can also temporarily disable Google by commenting out the search_google call in smart_search.

The web UI doesn’t open / “Connection refused”
Ensure no other application is using port 5000.

If you’re on Windows, the firewall may ask for permission – allow it.

Try upgrading Flask: pip install --upgrade flask.

Gradio errors (old versions)
The current script does not use Gradio – it uses Flask. If you still see Gradio errors, you’re running an older script. Please download the latest glassarc_safe.py.

Model loads but gives nonsense answers
The ChatML template is crucial for Qwen models. If you modified the script, make sure the generate() function still wraps the prompt in <|im_start|>system, <|im_start|>user, and <|im_start|>assistant.

Try a lower temperature? (Not currently exposed, but you can add temperature=0.7 to the Llama call).

🏗️ Evolution of the Project
GlassArc went through several iterations to reach its current stable state:

v4.x‑v5.x – early versions with Gradio UI, GGUF generation, and complex model swapping.

v8.2.0 – robust, standalone web‑search engine with CRC and tracing.

Merged builds – combined AI chat + web search, but suffered from Gradio format errors and venv restart loops.

Final stable version (glassarc_safe.py) – replaces Gradio with Flask, removes venv restarts, keeps the proven search engine, and supports huge context windows.

📄 License
This project is released under the MIT License.
The GGUF models are subject to their respective original licenses (Qwen, Llama, TinyLlama).

🙏 Acknowledgements
llama.cpp – the backbone of local LLM inference

llama-cpp-python – Python bindings for llama.cpp

Qwen – the language model used by default

Trafilatura – web content extraction

Fake‑UserAgent – rotating user‑agents for web scraping

Flask – lightweight web framework for the UI

All the open‑source libraries that made this project possible.

Happy chatting – and searching!
