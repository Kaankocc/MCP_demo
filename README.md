# Career Guidance Assistant

A sophisticated career guidance system that combines specific insights from professionals with general career advice using RAG (Retrieval-Augmented Generation) technology and a multi-agent routing system.

## Features

- **Multi-Agent System**: Uses specialized agents for different aspects of career guidance
  - RAG Career Agent: Provides insights from real professional interviews
  - General Career Agent: Offers broad career advice
  - Mentor Connect Agent: Helps connect students with mentors
- **Intelligent Routing**: Automatically routes queries to the most relevant agents
- **Parallel Processing**: Processes multiple agent responses simultaneously
- **Response Synthesis**: Combines insights from multiple agents into cohesive answers

## Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   Create a `.env` file in the root directory with:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── rag_agent.py          # RAG and agent implementation
├── router_agent.py       # LLM routing system
├── vectorstore.py        # Vector store operations
├── config.py             # Configuration settings
├── utils.py              # Utility functions
├── embedding.py          # Embedding utilities
└── requirements.txt      # Project dependencies
```

## Usage

1. Start the application:

```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided local URL (typically http://localhost:8501)

3. Use the interface to:
   - Ask career-related questions
   - Filter by industry
   - Filter by key takeaways
   - Clear chat history

## How It Works

1. **Query Processing**:

   - User submits a question through the Streamlit interface
   - Query is processed with any selected filters

2. **Routing**:

   - LLM Router determines the most relevant agents for the query
   - Selects top 2 agents based on relevance scores

3. **Parallel Processing**:

   - Selected agents process the query simultaneously
   - Each agent provides specialized insights

4. **Response Synthesis**:
   - Synthesizer combines responses from multiple agents
   - Produces a cohesive, well-structured answer

## API Usage

The system uses OpenAI's API for:

- LLM routing decisions
- Agent responses
- Response synthesis

Note: Be mindful of API usage as each query involves multiple API calls.

## Error Handling

The system includes error handling for:

- API quota limits
- Invalid queries
- Processing failures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License Here]

## Support

For support, please [contact information or issue tracker details]
