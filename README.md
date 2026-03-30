# Multi-Agent Debate System Based on Large Language Models
## Structured Deliberation and Validation in Satellite Communications

This repository contains the official implementation of the paper: **"Multi-Agent Debate System Based on Large Language Models: Structured Deliberation and Validation in Satellite Communications"**.

### Authors
- **Susana Gómez** (University of Malaga)
- **Alejandro Mozo** (University of Malaga)
- **Tomás Navarro** (European Space Agency - ESA)
- **Sergio Gálvez** (University of Malaga)
- **Francisco L. Valverde** (University of Malaga)

---

## Overview

Structured multi-agent debates among Large Language Models (LLMs) have emerged as a paradigm for enhancing reasoning reliability, factual accuracy, and argumentative coherence. This study proposes and evaluates a moderated, domain-adaptive multi-agent debate framework applied to **Satellite Communications (SatCom)**, a high-stakes technical domain central to the operational priorities of the European Space Agency (ESA).

### Key Features
- **Moderated Deliberation**: A centralized moderator agent governs the process, enforcing role adherence and synthesizing technical arguments.
- **Domain-Specialized Experts**: Specifically defined roles for *Link Design and Analysis (DLB)* and *Payload and Network Management (PNM)*.
- **Hybrid RAG Integration**: Contextual grounding using institutional ESA material (Nebula portal) and academic publications (Jábega repository).
- **Architectural Diversity**: Comparative analysis of homogeneous (Llama-3.3) and heterogeneous (Llama-3.3 + DeepSeek-R1 + Qwen-2.5) configurations.

---

## System Architecture

The framework introduces a hierarchical multi-agent deliberation system designed to achieve structured technical consensus.

### Deliberation Process
The process unfolds through an iterative, four-phase cycle:
1.  **Initialization**: The moderator defines the technical problem and establishes the interaction rules.
2.  **Argumentation**: Expert agents independently propose solutions based on their specialization and RAG-retrieved evidence.
3.  **Refutation & Adjustment**: Agents exchange counterarguments and refine their positions under moderator supervision.
4.  **Synthesis & Consensus**: The moderator summarizes key agreements and formulates a final, unified executive decision.

### Retrieval-Augmented Generation (RAG)
The system is grounded in a specific SatCom corpus featuring:
- **ESA Nebula Portal**: SatNex V programme reports and applied research.
- **Jábega Repository**: Academic and technical publications from the University of Málaga.

---

## Evaluation Framework

The system was validated using a dataset of **213 technical queries** and evaluated via **LLM-as-a-judge** (using the *gpt-oss 120B* model).

### Iterative Evaluation Phases
Performance was assessed across three progressive levels:
- **Phase I: Baseline Proficiency**: Accuracy, completeness, and adherence to engineering definitions.
- **Phase II: Strategic Reasoning**: Dialectic synthesis and technical depth in managing trade-offs.
- **Phase III: Executive Readiness**: Strategic decision-making and practical viability for mission-critical operations.

### Results
The evaluation demonstrated that while single-agent systems excel in encyclopedic tasks, **heterogeneous debates achieve superior performance in executive scenarios**, producing more robust and actionable consensus for high-complexity technical conflicts.

---

## Repository Structure

- `debate-system/`: Core implementation of the multi-agent architectures (Individual, Homogeneous, Heterogeneous).
- `evaluation/`: Scripts and prompts for the LLM-as-a-judge evaluation phases.
- `topics/`: Dataset of technical queries and RAG source metadata.

## Getting Started

For detailed instructions on how to install, configure, and run the debate system, please refer to the [Technical Documentation](debate-system/README.md).

---

## Acknowledgments
The authors acknowledge the computer resources provided by the Picasso Supercomputer (SCBI) at the University of Málaga.