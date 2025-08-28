
# BPMN RAG AI Agent

This project implements a **Hybrid Retrieval-Augmented Generation (RAG) Agent** that stores BPMN (Business Process Model and Notation) processes in a **Neo4j Graph Database** and performs intelligent **graph + embedding-based query answering**.  

It is designed to support consultants and analysts in referencing past processes efficiently. By combining **structural graph queries** and **semantic vector search**, the agent provides accurate and context-rich answers with direct process diagram visualization.  

---

## Key Features

- **BPMN → Neo4j Mapping**: Automatically stores BPMN elements (Events, Activities, Gateways, Roles) as nodes and relationships in a graph database  
- **Hybrid RAG Search**: Combines text embeddings with graph queries for high-precision retrieval and reasoning  
- **Process Diagram Management**: Stores BPMN diagrams (PDF/Image) in MinIO and renders them directly in the Streamlit UI  
- **Containerized Infrastructure**: Runs Redis (cache) and MinIO (object storage) services with Docker Compose  
- **Interactive Streamlit UI**: Provides process explanations and embedded diagrams in a user-friendly interface  

---

## Prerequisites

### 1. Install Dependencies
- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/)  
- Python 3.9+ (virtual environment recommended)  

### 2. Configure Secrets
Create a `.streamlit/secrets.toml` file in the project root and set the following:

```toml
NEO4J_URI="neo4j+s://your_uri.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your_password"
SEGMENT_WRITE_KEY=""
OPENAI_API_KEY="your key"
```

---

## Getting Started

1. Clone the Repository
```bash
git clone https://github.com/alciakng/bpmn_rag_ai_agent.git
cd bpmn_rag_ai_agent
```

## Start Container Services
```bash
docker-compose up -d
```

- **Redis**: Caching layer  
- **MinIO**: Object storage for BPMN diagrams (default: http://localhost:9000, console: http://localhost:9001)  

---

## Run the Streamlit Application
```bash
pip install -r requirements.txt
streamlit run app.py
```


## Access the UI
Open [http://localhost:8501](http://localhost:8501) in your browser.  

---

## Project Structure
```bash
.
├── agent/                  # Hybrid RAG agent logic
├── config/                 # Configuration files (yaml, toml, etc.)
├── data/                   # BPMN source files (xml, pdf, images)
├── ingestion/              # BPMN → Neo4j ingestion scripts
├── ui/                     # Streamlit user interface
├── utils/                  # Utility functions (MinIO, helpers, etc.)
├── docker-compose.yaml     # Redis / MinIO container configuration
└── README.md
```

## Example Workflow
1. Load the **Order-to-Cash** BPMN process into Neo4j  
2. Ask a question in the Streamlit UI: *“What happens during quotation creation?”*  
3. The agent combines Neo4j graph traversal with embedding similarity search  
4. The response includes:  
   - A textual explanation of the relevant process flow  
   - A direct BPMN diagram (PDF/Image) rendered inline  

---

## Roadmap
- Automated BPMN XML → Graph transformation pipeline  
- Multilingual query support (Korean/English hybrid search)  
- Advanced RACI (Responsible/Accountable/Consulted/Informed) mapping and analysis  
- Monitoring & observability with Prometheus/Grafana  

---

## License
Distributed under the [MIT License](LICENSE).  

