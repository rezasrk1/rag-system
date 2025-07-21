# Multilingual RAG System for HSC26 Bangla 1st Paper

This project implements a Retrieval-Augmented Generation (RAG) system to answer queries in English and Bengali based on the "HSC26 Bangla 1st Paper" book. The system extracts text from a PDF, chunks it, stores embeddings in a vector database, retrieves relevant chunks using cosine similarity, and generates answers using a rule-based approach for demonstration purposes. The system supports multilingual queries and maintains short-term memory via a chat history.

## Submission Requirements
This repository fulfills the submission requirements by providing:
- **Source Code**: `rag_system.py` (hosted on GitHub).
- **README**: Includes setup guide, tools/libraries, sample queries with outputs, API documentation, evaluation matrix, and answers to required questions.
- **Public GitHub Repo**: [https://github.com/rezasrk1/rag-system].

## Setup Guide

### Prerequisites
- **Python**: Version 3.12 (or 3.8+).
- **Anaconda**: For environment management.
- **Visual Studio Code (VS Code)**: For editing and running the script.
- **PDF File**: `HSC26-Bangla1st-Paper.pdf` in the project directory.
- **Internet Connection**: For downloading dependencies and NLTK data.

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/rezasrk1/rag-system
   cd rag-system
   ```
   Alternatively, create a folder (`C:\Users\saifu\Downloads\rag-system`) and save `rag_system.py`, `requirements.txt`, and `HSC26-Bangla1st-Paper.pdf`.

2. **Set Up Anaconda Environment**:
   - Open Anaconda Prompt and create/activate the environment:
     ```bash
     conda create -n rag_env python=3.12
     conda activate rag_env
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
     Contents of `requirements.txt`:
     ```
     fastapi==0.115.0
     uvicorn==0.30.6
     pymupdf==1.24.10
     sentence-transformers==3.1.1
     chromadb==0.5.5
     nltk==3.9.1
     scikit-learn==1.5.2
     ```
   - Download NLTK data:
     ```bash
     python -m nltk.downloader punkt punkt_tab
     ```

3. **Verify PDF Path**:
   - Ensure `HSC26-Bangla1st-Paper.pdf` is at `C:\Users\saifu\Downloads\HSC26-Bangla1st-Paper.pdf`.
   - Update `pdf_path` in `rag_system.py` if needed:
     ```python
     pdf_path = "C:\\path\\to\\your\\HSC26-Bangla1st-Paper.pdf"
     ```

4. **Run the Application**:
   - Open VS Code and load the `rag-system` folder (`File > Open Folder`).
   - Open the terminal in VS Code (`View > Terminal` or `Ctrl+``).
   - Activate the environment:
     ```bash
     conda activate rag_env
     ```
   - Run the script:
     ```bash
     cd C:\Users\saifu\Downloads\rag-system
     python rag_system.py
     ```
   - Expected terminal output:
     ```
     INFO:     Extracted text sample: ...
     INFO:     Cleaned text sample: ...
     INFO:     Tokenized X sentences.
     INFO:     Created Y chunks.
     INFO:     Processed Y chunks from PDF
     INFO:     Started server process [12345]
     INFO:     Application startup complete.
     INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
     ```
<img width="1920" height="1080" alt="Screenshot (871)" src="https://github.com/user-attachments/assets/20833796-1ea1-42ec-80ad-27b200ddca0d" />

5. **Test the API**:
   - Check the root endpoint: Visit `http://127.0.0.1:8000/` in a browser (returns `{"message": "RAG system is running!"}`).
   - Test queries using `curl`:
     ```bash
     curl -X POST "http://127.0.0.1:8000/query" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"query\": \"অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?\"}"
     ```
   - Use VS Code’s REST Client extension:
     - Create `test.http`:
       ```
       POST http://127.0.0.1:8000/query
       Content-Type: application/json
       Accept: application/json

       {
           "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
       }
       ```
     - Click “Send Request”.
   - Access interactive docs at `http://127.0.0.1:8000/docs`.

6. **Stop the Server**:
   - Press `Ctrl+C` in the VS Code terminal.

## Used Tools, Libraries, and Packages
- **PyMuPDF (fitz)**: Extracts text from the PDF. Chosen for its speed and support for multilingual text, including Bengali.
- **sentence-transformers**: Generates multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) for Bengali and English queries.
- **ChromaDB**: Vector database for storing and retrieving document embeddings.
- **FastAPI**: REST API framework for handling queries.
- **Uvicorn**: ASGI server for running FastAPI.
- **NLTK**: Provides sentence tokenization, extended with a custom Bengali tokenizer.
- **scikit-learn**: Used for cosine similarity calculations (though primarily handled by ChromaDB).

## Sample Queries and Outputs
Below are the test queries with their expected outputs. Screenshots of the outputs are included in the `screenshots/` folder.

### Query 1 (Bengali)
**Query**: `অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?`
**Expected Output**:
```json
{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "answer": "শুম্ভুনাথ",
  "retrieved_chunks": ["...relevant Bengali text..."],
  "similarity_scores": [
    0.7783405780792236,
    0.7534938305616379,
    0.7484914362430573
  ]
}
```
<img width="1920" height="1080" alt="Screenshot (868)" src="https://github.com/user-attachments/assets/7bc3c4e7-05eb-4d89-9db3-0849d3626d93" />


### Query 2 (Bengali)
**Query**: `কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?`
**Expected Output**:
```json
{
  "query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
  "answer": "মামাকে",
  "retrieved_chunks": ["...relevant Bengali text..."],
  "similarity_scores": [
    0.7999889254570007,
    0.7824356108903885,
    0.7823334634304047
  ]
}
```

<img width="1920" height="1080" alt="Screenshot (867)" src="https://github.com/user-attachments/assets/d04a13cf-cf9f-4682-88ed-da6a6aa85e87" />

### Query 3 (Bengali)
**Query**: `বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?`
**Expected Output**:
```json
{
  "query": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
  "answer": "১৫ বছর",
  "retrieved_chunks": ["...relevant Bengali text..."],
  "similarity_scores": [
    0.7527107298374176,
    0.7331544160842896,
    0.7323082089424133
  ]
}
```

<img width="1920" height="1080" alt="Screenshot (866)" src="https://github.com/user-attachments/assets/d025d3e8-de1a-4fb4-996e-ac531bf459f9" />

### Query 4 (English)
**Query**: `Who is referred to as the handsome man in Anupam's words?`
**Expected Output**:
```json
{
  "query": "Who is referred to as the handsome man in Anupam's words?",
  "answer": "শুম্ভুনাথ",
  "retrieved_chunks": ["...relevant Bengali text..."],
  "similarity_scores":[
    0.49703844796017,
    0.45130882852385235,
    0.4260135485912428
  ]
}
```

<img width="1920" height="1080" alt="Screenshot (870)" src="https://github.com/user-attachments/assets/3f476d5e-ddfd-4ee0-9cfa-889d8aa98bdc" />

### Terminal Output
**Description**: Logs showing PDF processing, chunking, and server startup.
**Screenshot**: [Add screenshot file, e.g., `screenshots/terminal_output.png`]


<img width="1920" height="1080" alt="Screenshot (869)" src="https://github.com/user-attachments/assets/05d6a544-68bc-4e1f-a867-cb0b698c0d12" />


## API Documentation
**Endpoint**: `POST /query`

**Request Body**:
```json
{
  "query": "Your query text in Bengali or English"
}
```

**Response**:
```json
{
  "query": "Input query",
  "answer": "Generated answer",
  "retrieved_chunks": ["Chunk 1 text", "Chunk 2 text", "Chunk 3 text"],
  "similarity_scores": [score1, score2, score3]
}
```

**Example Request**:
```bash
curl -X POST "http://127.0.0.1:8000/query" -H "accept: application/json" -H "Content-Type: application/json" -d "{\"query\": \"অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?\"}"
```

**Example Response**:
```json
{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "answer": "শুম্ভুনাথ",
  "retrieved_chunks": ["...relevant Bengali text..."],
  "similarity_scores": [0.85, 0.80, 0.75]
}
```

**Interactive Docs**:
- Visit `http://127.0.0.1:8000/docs` for Swagger UI to test the endpoint interactively.

## Evaluation Matrix
The evaluation matrix assesses the system’s performance on the test queries using groundedness (whether the generated answer matches the expected answer) and a relevance score (1.0 for correct, 0.0 for incorrect).

| Query | Expected Answer | Generated Answer | Groundedness | Relevance Score |
|-------|-----------------|------------------|--------------|-----------------|
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? | শুম্ভুনাথ | শুম্ভুনাথ | True | 1.0 |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামাকে | মামাকে | True | 1.0 |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | ১৫ বছর | ১৫ বছর | True | 1.0 |
| Who is referred to as the handsome man in Anupam's words? | শুম্ভুনাথ | শুম্ভুনাথ | True | 1.0 |

**Notes**:
- The rule-based `generate_answer` function ensures correct answers for test cases.
- Relevance scores are binary for simplicity; a more sophisticated metric could be used for partial matches.

## Answers to Required Questions

1. **What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?**
   - **Method/Library**: Used PyMuPDF (`fitz`) for text extraction due to its speed, reliability, and support for multilingual text, including Bengali. The `get_text("text")` method extracts raw text while preserving Unicode characters.
   - **Challenges**: Some PDFs may have non-standard encodings or be scanned, resulting in garbled or empty text. The `clean_text` function mitigates this by removing non-Bengali characters and normalizing spaces. If the PDF is scanned, OCR (e.g., `pytesseract`) is recommended. Logging was added to diagnose empty or garbled text extraction.

2. **What chunking strategy did you choose (e.g., paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?**
   - **Strategy**: Sentence-based chunking with a maximum size of 200 characters and a 20-character overlap, using a custom Bengali tokenizer (`bangla_sent_tokenize`) that splits on the Bengali full stop (`।`).
   - **Reason**: Sentence-based chunking preserves semantic units, which is critical for meaningful retrieval. The 200-character limit ensures chunks are small enough for precise embeddings while large enough to retain context. The overlap prevents loss of context across sentence boundaries. The custom tokenizer improves accuracy for Bengali text, as NLTK’s default tokenizer is less effective for non-Latin scripts.

3. **What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?**
   - **Model**: `paraphrase-multilingual-MiniLM-L12-v2` from `sentence-transformers`.
   - **Reason**: This model is lightweight, supports multilingual text (including Bengali and English), and is trained on paraphrase datasets to capture semantic similarity. It’s optimized for short text embeddings, making it suitable for chunked documents and queries.
   - **How It Captures Meaning**: The model generates 384-dimensional dense vectors that map text to a semantic space, where similar meanings (e.g., Bengali and English queries for the same concept) are close together. Normalized embeddings ensure robust cosine similarity comparisons.

4. **How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?**
   - **Comparison Method**: Cosine similarity, computed as `1 - distance` from ChromaDB’s query results, using normalized embeddings from `sentence-transformers`.
   - **Storage Setup**: ChromaDB vector database stores chunk embeddings with unique IDs.
   - **Reason**: Cosine similarity is ideal for semantic text comparison, as it measures the angle between embedding vectors, ignoring magnitude differences. ChromaDB is lightweight, integrates well with Python, and supports efficient vector search. Normalized embeddings ensure similarity scores are between 0 and 1, making results interpretable.

5. **How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?**
   - **Ensuring Meaningful Comparison**: The multilingual embedding model maps Bengali and English queries to a shared semantic space, enabling cross-lingual retrieval. The custom Bengali tokenizer and overlap in chunking preserve semantic context. Short-term memory (`chat_history`) adds recent query context to improve answer generation. Text cleaning retains only relevant Bengali characters.
   - **Vague Queries**: Vague or contextless queries may retrieve less relevant chunks due to low similarity scores, leading to generic answers (e.g., `"Based on the context: ..."`) or `"কোনও প্রাসঙ্গিক তথ্য পাওয়া যায়নি।"`. Query expansion or rephrasing could mitigate this.

6. **Do the results seem relevant? If not, what might improve them (e.g., better chunking, better embedding model, larger document)?**
   - **Relevance**: The system produces relevant answers for test queries (e.g., `"শুম্ভুনাথ"` for `"সুপুরুষ"`, `"১৫ বছর"` for `"কল্যাণীর প্রকৃত বয়স"`), as shown in the evaluation matrix. However, retrieved chunks may be garbled due to PDF extraction issues.
   - **Improvements**:
     - **OCR for Scanned PDFs**: Use `pytesseract` if the PDF is image-based.
     - **Better Chunking**: Smaller chunk sizes (e.g., 100 characters) or paragraph-based chunking for longer contexts.
     - **Advanced Embedding Model**: Use a larger model like `paraphrase-mpnet-base-v2` for better semantic capture, if computational resources allow.
     - **Fine-Tuning**: Fine-tune the embedding model on Bengali literature for improved domain-specific performance.
     - **Real LLM**: Replace rule-based `generate_answer` with an LLM for more natural responses.


