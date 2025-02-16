A simple retrieval augmented generation (RAG) system that lets you collect emails, upload documents, generate embeddings, and chat with GPT to extract useful information from your data.

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**  
   Create a `.env` file in the project root with the following keys:

   ```env
   GMAIL_USERNAME=your_email@gmail.com
   GMAIL_PASSWORD=your_gmail_app_password
   OPENAI_API_KEY=your_openai_api_key
   ```

4. **Configure the Application**  
   Review and update `config.yaml` if necessary (e.g., vault file paths, OpenAI settings, top_k parameters).

5. **Usage**

   - **Collect Emails**:  
     Run the email collection script to fetch and process your Gmail emails:
     ```bash
     python collect_emails.py --keyword "search_term" --startdate "01.01.2022" --enddate "31.01.2022"
     ```
   - **Upload Documents**:  
     Use the Tkinter-based upload interface to import PDF, text, or JSON files into your vault:
     ```bash
     python upload.py
     ```
   - **Chat with the RAG System**:  
     Launch one of the chat interfaces to ask questions and retrieve context from your documents/emails:

     ```bash
     python emailrag.py
     ```

     or

     ```bash
     python rag.py
     ```

   - **Generate Embeddings**:  
     Regenerate embeddings for the vault content if required:
     ```bash
     python generate_embeddings.py
     ```

## License

This project is licensed under the MIT License.
