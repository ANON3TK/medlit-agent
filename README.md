# MedLit Agent

MedLit Agent is a citation-grounded medical literature assistant built with
Python and FastAPI.

It can upload PDF documents, retrieve relevant evidence, generate answers with
traceable page citations, search PubMed through tool calling, and preserve
documents and interaction records in SQLite.

This project is intended for literature assistance and engineering practice.
It is not a clinical diagnosis system.

## Main workflow

```text
PDF upload
→ PDF parsing
→ text chunking
→ embedding generation
→ vector storage
→ similarity retrieval
→ evidence-grounded answer
→ page citations
```

The Agent workflow can also use a real external tool:

```text
User question
→ model decides whether PubMed search is needed
→ application executes PubMed search
→ tool results return to the model
→ model generates the final answer
```

## Current capabilities

- PDF type, size and readability validation
- Page-level PDF text extraction
- Overlapping text chunking with page numbers
- SiliconFlow `BAAI/bge-m3` embeddings
- Document-scoped vector similarity search
- Evidence-grounded answers with page citations
- PubMed search through native function calling
- SQLite document, vector and interaction persistence
- Structured JSON logs without duplicating medical text
- Retry handling for external AI services
- Retrieval hit-rate evaluation
- FastAPI Swagger documentation
- Automated pytest coverage

## Technology stack

- Python 3.12
- FastAPI
- Uvicorn
- Pydantic
- pypdf
- httpx
- SQLite
- pytest
- SiliconFlow API
- PubMed E-utilities

## Installation

Create or activate a Python 3.12 environment, then run:

```powershell
python -m pip install -e ".[dev]"
```

## Environment variables

The application uses the following environment variables.

### SiliconFlow API key

```powershell
$env:SILICONFLOW_API_KEY = "your-api-key"
```

Do not write the real API key into source code or commit it to Git.

### SQLite database path

```powershell
$env:MEDLIT_DATABASE_PATH = "data/medlit.db"
```

When this variable is configured, documents, vectors and interaction records
are stored in SQLite and remain available after the service restarts.

Without this variable, the application uses in-memory storage.

## Run tests

```powershell
python -m pytest -q
```

The currently verified result is:

```text
54 passed
```

## Start the development server

```powershell
python -m uvicorn medlit_agent.main:app --reload --port 8765
```

Then open the Swagger interface:

```text
http://127.0.0.1:8765/docs
```

Health check:

```text
http://127.0.0.1:8765/health
```
## Run with Docker

Build the Docker image:

```powershell
docker build -t medlit-agent .
```

If Docker Hub is unavailable on the current network, the base image can be
overridden during the build:

```powershell
docker build --build-arg BASE_IMAGE=m.daocloud.io/docker.io/library/python:3.12-slim -t medlit-agent .
```

Create a named volume for SQLite persistence:

```powershell
docker volume create medlit-data
```

Start the service and pass the existing API key without writing its value into
the command:

```powershell
docker run --rm -d --name medlit-agent -p 8765:8765 --env SILICONFLOW_API_KEY --mount source=medlit-data,target=/app/data medlit-agent
```

Open the Swagger interface:

```text
http://127.0.0.1:8765/docs
```

Stop the container:

```powershell
docker stop medlit-agent
```

The container is removed because it was started with `--rm`, while the
`medlit-data` volume and its SQLite database remain available.


## Main API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Check whether the service is running |
| `POST` | `/documents` | Upload, parse, chunk and index a PDF |
| `GET` | `/documents/{document_id}` | Read stored document information |
| `POST` | `/documents/{document_id}/search` | Search relevant document chunks |
| `POST` | `/documents/{document_id}/ask` | Answer using document evidence |
| `POST` | `/documents/{document_id}/evaluate` | Evaluate page-level retrieval hit rate |
| `POST` | `/agent/ask` | Ask the Agent, which may search PubMed |
| `GET` | `/interactions` | Read recent interaction records |

## Retrieval evaluation

The evaluation endpoint accepts questions with their expected evidence pages.

Example request:

```json
{
  "cases": [
    {
      "question": "What does the document say about atrial fibrillation?",
      "expected_page_number": 2
    }
  ],
  "top_k": 3
}
```

Example response:

```json
{
  "total_cases": 1,
  "hit_count": 1,
  "hit_rate": 1.0
}
```

A case is counted as a hit when its expected page appears in the retrieved
top-k results.

## Reliability decisions

- Answers are grounded in retrieved evidence.
- The model is not called when no sufficiently relevant evidence is found.
- Search is restricted by `document_id` to avoid cross-document citations.
- External AI requests use retries and return HTTP 502 after final failure.
- API keys are read from environment variables.
- Logs exclude full questions, answers and original PDF text.
- SQLite transactions keep related records consistent.

## Limitations

- Retrieval evaluation currently measures page-level hit rate only.
- Text chunking currently uses character-based windows.
- Scanned PDFs without an OCR text layer cannot be extracted correctly.
- The system assists literature retrieval and does not provide medical advice.
