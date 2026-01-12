# vLLM Cost Optimization

A framework for cheap, reliable, production-grade LLM inference with full auditability.

## Features

- **Multi-model support** — Chat and embedding models via Ollama
- **Thinking model handling** — Automatic token budget for reasoning models (DeepSeek-R1, Qwen3)
- **Full auditability** — All requests logged to MySQL with reproducibility
- **External tools** — 18 data source integrations (Wikipedia, arXiv, news, crypto, etc.)
- **Reproducibility** — Temperature=0 + seed for deterministic outputs

## Quick Start

### One-Command Setup (EC2/Linux)

```bash
# Clone repo and run setup
git clone https://github.com/your-repo/vllm_cost_optimization.git
cd vllm_cost_optimization
./infra/setup_ec2.sh
```

This installs everything (Docker, Python, Ollama, MySQL), pulls models, and runs tests.

---

### Manual Setup

#### 1. Setup Environment

```bash
cd vllm_cost_optimization

# Create virtual environment
mkdir -p env && cd env
python3 -m venv venv
source venv/bin/activate
cd ..

# Install dependencies
pip install requests ruamel.yaml mysql-connector-python feedparser
```

### 2. Start MySQL

```bash
cd infra
./setup_mysql.sh
```

### 3. Start Ollama

Make sure Ollama is running with your models:

```bash
ollama serve  # If not already running
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

### 4. Update Models Config

```bash
python3 infra/update_ollama_models.py infra/config.yaml
```

## Usage

### Basic Chat

```python
from src import call_model

# Simple call (model decides token limit)
response = call_model("qwen3:8b", "What is the capital of France?")
print(response)

# With specific output length
response = call_model("qwen3:8b", "Explain quantum computing", output_tokens=200)
```

### Thinking Models

Thinking models (DeepSeek-R1, Qwen3) automatically get extra token budget for reasoning:

```python
from src import OllamaClient

client = OllamaClient.from_config("infra/config.yaml")

# Get structured response with reasoning
resp = client.chat("deepseek-r1:14b", [{"role": "user", "content": "What is 2+2?"}])

print(resp.content)      # "4" (final answer)
print(resp.reasoning)    # "Let me think through this..." (chain of thought)
print(resp.has_reasoning)  # True
```

### Embeddings

```python
from src import call_embedding

# Single text
vectors = call_embedding("nomic-embed-text:latest", "Hello world")

# Batch
vectors = call_embedding("nomic-embed-text:latest", [
    "First document",
    "Second document",
])
```

### External Tools

18 data source tools with full audit logging:

```python
from src import init_registry, call_tool, list_tools

# Initialize with database logging
init_registry("infra/config.yaml")

# Search Wikipedia
result = call_tool("wikipedia_search", query="machine learning", limit=5)
if result.success:
    for title, snippet in result.data:
        print(f"- {title}")

# Get crypto price
result = call_tool("coingecko_price", coin_id="bitcoin", vs="usd")
print(f"Bitcoin: ${result.data['usd']}")

# Search recent papers
result = call_tool("arxiv_search", query="large language models", max_results=5)

# List all available tools
for tool in list_tools():
    print(f"{tool['name']} ({tool['category']}): {tool['description']}")
```

**Available tools:**

| Category | Tools |
|----------|-------|
| wikipedia | `wikipedia_search`, `wikipedia_summary` |
| wikidata | `wikidata_search`, `wikidata_sparql` |
| worldbank | `worldbank_search_indicators`, `worldbank_country_indicator` |
| fred | `fred_search`, `fred_observations` |
| census | `census_acs_population` |
| crypto | `coingecko_search`, `coingecko_price` |
| weather | `weather_forecast` |
| nasa | `nasa_apod` |
| news | `gdelt_news_search` |
| papers | `arxiv_search`, `pubmed_search`, `pubmed_summaries` |
| datasets | `datagov_search` |

## Database Schema

All requests are logged for auditability and reproducibility:

```
┌──────────────┐     ┌────────────────┐     ┌─────────────────┐
│   prompts    │◄────│ chat_requests  │────►│     models      │
│  (immutable) │     │                │     │                 │
└──────────────┘     └────────────────┘     └─────────────────┘
                            │
                     ┌──────┴──────┐
                     │             │
              response_content   reasoning_content
              (final answer)     (chain of thought)
```

### Key Tables

- **chat_requests** — All LLM calls with full context
- **embedding_requests** — All embedding calls
- **tool_calls** — All external API calls with input/output
- **prompts** — Immutable prompt storage (content-addressed)
- **models** — Model registry

### Query Examples

```bash
# Connect to MySQL
./infra/mysql.sh

# Recent chat requests
SELECT model_name, user_message, response_content 
FROM chat_requests ORDER BY id DESC LIMIT 5;

# Tool call audit
SELECT tool_name, status, response_time_ms, output_summary 
FROM tool_calls ORDER BY id DESC LIMIT 10;
```

## Reproducibility

With `temperature=0` and `seed`, outputs are deterministic:

```python
from src import OllamaClient

client = OllamaClient.from_config("infra/config.yaml")

# Reproducible call
resp = client.chat(
    "qwen3:8b",
    [{"role": "user", "content": "What is 2+2?"}],
    temperature=0,
    seed=42,
)

# Run multiple times — same output guaranteed
```

Test reproducibility:

```bash
python3 src/test_reproducibility.py
```

## Configuration

### `infra/config.yaml`

```yaml
endpoints:
  ollama:
    base_url: "http://localhost:11434/v1"
    
    chat_models:
      - name: qwen3:8b
        thinking: true
        thinking_budget: 2048
      - name: qwen2.5-coder:7b
        thinking: false
    
    embedding_models:
      - name: nomic-embed-text:latest

database:
  mysql:
    host: localhost
    port: 3306
    database: vllm
    user: vllm_user
    password: vllm_pass
```

## Scripts

| Script | Purpose |
|--------|---------|
| `infra/setup_ec2.sh` | **Full EC2/Linux setup** (installs everything, runs tests) |
| `infra/setup_mysql.sh` | Start MySQL Docker container |
| `infra/mysql.sh` | Connect to MySQL CLI |
| `infra/update_ollama_models.py` | Sync models from Ollama to config |
| `src/test_models.py` | Test all chat and embedding models |
| `src/test_reproducibility.py` | Verify deterministic outputs |
| `src/test_tools.py` | Test external data source tools |

## License

MIT
