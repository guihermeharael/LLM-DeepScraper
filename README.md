# HyperExtract-IA 🚀
**Autonomous Deep Data Extraction & LLM Dataset Pipeline**

**HyperExtract-IA** is an autonomous data mining ecosystem designed for exhaustive web information extraction. The system utilizes a multi-stage architecture that combines resilient browser automation with **Large Language Model (LLM)** orchestration to convert unstructured web content into high-fidelity datasets.

Originally conceived for mining technical specifications in complex marketplaces, its agnostic logic allows for deep data extraction across any web domain, making it ideal for creating knowledge bases for **LLM Fine-Tuning** (Alpaca/Instructional format).

---

## 🧠 Architecture & Senior Engineering Highlights

Unlike conventional scrapers, this pipeline implements advanced software engineering heuristics to ensure data integrity:

* **Intelligent Hybrid Pagination**: A system that prioritizes native link extraction (`href`) and applies **JavaScript injection** (`arguments.click()`) to overcome dynamic buttons without physical dimensions in the DOM.
* **Recursive Deep Extraction (Stage 2)**: An exploratory loop that actively interacts with tabs, modals, and "See more" buttons to capture hidden information under user interaction.
* **Domain-Specific Memory Mechanism**: Runtime persistence that stores successful selectors (**click memory**), drastically optimizing processing time for large-scale operations.
* **Incremental Persistence & Resilience**: Immediate disk writing (**JSONL, JSON, and CSV**) for every processed item, preventing data loss from network failures or API quota limits (Error 429).
* **Dual-Cache Management**: Strict separation between processed links and confirmed successes, allowing for fine-grained auditing and intelligent retries.

---

## 🛠️ Technology Stack

* **Browser Engine**: [DrissionPage](https://github.com/g1879/DrissionPage) (Robust and stealthy navigation).
* **LLM Orchestrator**: [ScrapegraphAI](https://github.com/VinciGit00/Scrapegraph-ai) (Multi-agent orchestration).
* **Processing**: BeautifulSoup4, Pydantic, html2text (Markdown normalization).

---

## 📋 How to Use

### Prerequisites
* **Python 3.10+**
* API Keys configured in a `.env` file (`OPENAI_API_KEY`, `GEMINI_API_KEY_PERSONAL`)

### Execution
```bash
python hyperextract_pipeline.py
```

### Inputs
* Search/Showcase URL.
* Search term for organic filtering.
* Target information (e.g., "Anatel Code", "Chemical Composition").

---

## 📂 Output Structure

The pipeline automatically organizes results to facilitate **Quality Assurance (QA)**:
* `resultados/todos_os_itens/`: Detailed logs of all visited URLs and extraction attempts.
* `resultados/itens_encontrados/`: Structured datasets containing only records where deep extraction was successful.

---

## ⚠️ Development Note (Architectural Roadmap)

This project is currently in a **Functional Monolith** phase. This structure was deliberately maintained to consolidate DOM interaction heuristics and validate LLM **Chain-of-Thought** in a synchronous and traceable flow.

### Planned Evolution
The roadmap envisions a migration to a **Distributed Software Architecture**, decoupling the **Crawler (Producer)** from the **Extractor (Consumer/Worker)** using message queues (RabbitMQ/Redis) and document-oriented database persistence to achieve massive horizontal scalability.

---
> *Developed as part of the Master's research at the **Instituto Tecnológico de Aeronáutica (ITA)**.*
