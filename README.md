
# Eklavya's Cheating ChatBot 🦾

This is a Streamlit-powered PDF QA chatbot that answers questions from uploaded PDF files. It uses LangChain, Sentence Transformers, and OpenRouter's GPT model for processing.

---

## 🚀 **Features**
- Upload PDF files and extract content.
- Ask questions about the PDF.
- Set a word limit for answers.
- Uses OpenRouter's GPT API for generating responses.
  
---

## 🔥 **Installation & Usage**

### 1️⃣ **Clone and Run**
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>

### 2️⃣ **Install Dependencies**
pip install -r requirements.txt

### 3️⃣ **Add API Keys**
- Create a `.env` file in the root directory.
- Add your OpenRouter API credentials:
OPENAI_API_KEY=your_openrouter_api_key
OPENAI_API_BASE=https://openrouter.ai/api/v1

### 4️⃣ **Run the Application**
streamlit run untitled.py

---

## 🛠️ **Dependencies**
- Python 3.9+
- Streamlit
- LangChain
- Sentence Transformers
- OpenRouter GPT
- dotenv

---

## 🐞 **Troubleshooting**
- If you encounter missing `PATH` issues, ensure that your Python environment is properly configured.
- Delete the `/temp` folder if you face any file path issues.
- If using MacOS, use `source ~/.zshrc` to reload the environment variables.
