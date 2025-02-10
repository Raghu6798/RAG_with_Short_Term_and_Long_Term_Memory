# GraphRAG with Conversational Memory 🤖💬

## Overview
GraphRAG is an advanced Retrieval-Augmented Generation (RAG) system that integrates Neo4j as a vector database to store and retrieve contextual information efficiently. This application enhances Generative AI capabilities by leveraging knowledge graphs and maintaining conversation history, enabling more context-aware and structured responses.

## Tech Stack
- **Neo4j** (Graph Database & Vector Store)
- **LangChain** (Retrieval-Augmented Generation Framework)
- **Gradio** (Interactive UI for chatbot)
- **Google Gemini AI** (LLM-powered responses)
- **Hugging Face Sentence Transformers** (Text embeddings)
- **Python** (Backend development)
- **Dotenv** (Environment variable management)

## Features
- Stores and retrieves contextual knowledge using **Neo4j Vector Store**
- Maintains **chat history** for contextual responses
- Uses **Hugging Face embeddings** for efficient retrieval
- Seamlessly integrates with **Google Gemini AI** for generating responses
- Provides an **interactive chat interface** using Gradio

## Installation & Setup
### Prerequisites
Ensure you have the following installed on your system:
- Python 3.8+
- Neo4j database (Cloud or Local Instance)
- Google AI Studio API Key

### Steps to Run the App
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/GraphRAG.git
   cd GraphRAG
   ```

2. **Create a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**
   Create a `.env` file in the root directory and add the following:
   ```env
   NEO4J_PASSWORD=your_neo4j_password
   GOOGLE_AI_STUDIO_API_KEY=your_google_api_key
   ```

5. **Run the Application:**
   ```bash
   python app.py
   ```

6. **Access the Chatbot:**
   Open `http://127.0.0.1:7860` in your browser.

## Contributing
Feel free to contribute by submitting issues or pull requests. Any improvements to enhance retrieval, memory, or model integrations are welcome! 🚀

## License
This project is licensed under the MIT License.

