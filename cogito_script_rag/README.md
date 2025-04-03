**Cogito RAG: Academic Research Assistant**
Cogito RAG is an advanced retrieval-augmented generation system that helps researchers find, 
analyze, and synthesize academic literature. The system retrieves relevant papers, extracts knowledge from PDFs,
and generates academic-quality answers with proper citations.

🚀 **Features**
Dual API Integration: Retrieves papers from both Semantic Scholar and OpenAlex APIs 
Intelligent Query Processing: Converts natural language questions into optimized search queries
Automatic PDF Processing: Downloads and extracts text from open-access academic papers
Semantic Chunking: Divides documents into meaningful segments with context preservation
Vector-Based Retrieval: Uses embeddings to find the most relevant information
Citation Validation: Ensures all citations meet user-specified year requirements
Academic Writing Style: Generates responses in formal academic style with proper citations
**
📋 Requirements**
Python 3.8+
OpenRouter API key (for LLM access)
Internet connection (for API calls)

# Clone the repository
git clone https://github.com/yourusername/cogito_rag.git
cd cogito_rag

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API key
echo "OPEN_ROUTER_API_KEY=your_api_key_here" > .env

**The system will:**
Convert your query into an optimized search term
Retrieve relevant academic papers
Download and process PDFs
Extract information and create embeddings
Generate an academic response with citations newer than 2018


🏛️** System Architecture**
Query Processor: Transforms user questions into optimized search terms
Document Retriever: Searches academic databases and downloads papers
Text Processor: Extracts and cleans text from PDFs
RAG System: Chunks text, creates embeddings, and retrieves relevant information
Response Generator: Creates academic-quality answers with proper citations
