# Technical Implementation & Setup - Debate System

This directory contains the core implementation of the multi-agent deliberation architectures evaluated in the paper. It is designed for modularity, providing implementations for single-agent (baseline), homogeneous, and heterogeneous multi-agent systems.

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   Copy `.env.example` to `.env` and add your `OPENROUTER_API_KEY`.
   ```bash
   cp .env.example .env
   ```

## Usage

Each system can be run using its respective `main.py` script. The common parameters are `--topic` for the question and `--outfile` for the results.

### Single-Agent Baseline
```bash
# Individual query
python single-agent/main.py --topic "Your question" --outfile results.json

# Batch processing
python single-agent/launch_batch.py questions_folder/ output_folder/
```

### Homogeneous Multi-Agent
```bash
python multi-agent-homogeneous/main.py --topic "Your question" --outfile results.txt
```

### Heterogeneous Multi-Agent
```bash
python multi-agent-heteregenous/main.py --topic "Your question" --outfile results.txt
```

### Batch Processing
To process multiple questions in batch, use `launch_debates.py` within the respective system directory:
```bash
python multi-agent-homogeneous/launch_debates.py questions_folder/ output_folder/ decisions_folder/
```

## Repository Structure

```
debate-system/
├── common/                # Shared clients, prompts, and RAG utilities
├── configs/               # Global settings management
├── single-agent/          # Individual LLM architecture
├── multi-agent-homogeneous/  # Homogeneous debate architecture
└── multi-agent-heteregenous/ # Heterogeneous debate architecture
```

---

For the main research objectives, architecture, and evaluation methodology, please refer to the [Root README](../README.md).