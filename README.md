Adobe India Hackathon 2025 - Team ByteMe

🚀 Project Introduction

Thank You for reviewing Team ByteMe's submission for the Adobe India Hackathon 2025! This project reimagines the traditional PDF from a static document into an intelligent, interactive knowledge companion.

Our solution for Round 1B, focuses on transforming how users interact with vast document collections. By intelligently understanding document structure and content, we enable a system that extracts and prioritizes the most relevant information tailored to a specific user persona and their immediate "job-to-be-done".

💡 The Problem & Our Vision

In a world increasingly flooded with digital documents, the real challenge isn't content scarcity, but context scarcity. Traditional reading often leaves users to manually sift through pages, connecting disparate ideas to find the knowledge they need.

Our vision is to build the future of reading and learning – an experience where PDFs actively assist users by surfacing insights and acting as a trusted research companion. This solution is a critical step towards that future.

⚙️ Solution Overview: How it Works

Our approach for Persona-Driven Document Intelligence combines robust document understanding with semantic intelligence:

    Intelligent Section Extraction: We leverage advanced PDF parsing to go beyond simple text extraction. Our system identifies logical document sections (akin to Title, H1, H2, H3 headings) by analyzing font sizes, bolding, and structural patterns within the PDF. This ensures that our "sections" represent coherent, meaningful blocks of content, rather than arbitrary text chunks. Each extracted section includes its inferred title and the full body text associated with it.

    Global Semantic Relevance Ranking:

        Utilizing a pre-trained Sentence Transformer model (all-MiniLM-L6-v2), we generate high-dimensional vector embeddings for both the user's "job-to-be-done" query and all the extracted logical sections from the entire document collection.

        Cosine similarity is then used to calculate the semantic closeness between the job description and each document section.

        All sections are ranked globally, identifying the most relevant pieces of information across the entire dataset.

    Concise Sub-Section Analysis: To address verbosity, for each of the top-ranked sections, we perform a secondary analysis. The full text of the section is further broken down into sentences, and the top N most semantically relevant sentences (e.g., 3-5) to the "job-to-be-done" are extracted. This provides a focused, actionable summary for the user.

✨ Key Features

    Persona-Driven Insights: Dynamically extracts and prioritizes content based on specific user roles and tasks.

    Structured Document Understanding: Identifies logical sections and their associated content, mimicking human comprehension of document hierarchy.

    Global Relevance Ranking: Pinpoints the most critical information across an entire collection of related PDFs.

    Concise Summarization: Provides refined, actionable text snippets for detailed sub-section analysis.

    Offline Capability: Designed to run without internet access post-setup, ideal for secure or constrained environments.

    CPU Optimized: Engineered to run efficiently on CPU-only environments, adhering to hackathon constraints.

💻 Technical Stack

    Language: Python 3.9 or higher

    PDF Processing: PyMuPDF (via fitz) for robust text and metadata extraction.

    Natural Language Processing (NLP): Sentence Transformers for efficient sentence embeddings and semantic similarity.

    Machine Learning: scikit-learn for cosine similarity calculations.

    Utility: tqdm for progress visualization.

🚀 Getting Started
1. Prerequisites

    Python 3.9+ installed.

    pip package manager.

    Familiarity with command-line interface.

2. Environment Setup & Installation

Open a terminal in the root directory of this repository.

First, install all required Python packages:
```
    pip install -r Challenge_1b/requirements.txt
```

3. Data Preparation

Organize your PDF documents and input configuration as follows:

    Create collection directories (e.g., Collection 1, Collection 2, Collection 3) in a structured path (e.g., Challenge_1b/Collection 1/).

    Within each collection directory, create a PDFs subfolder. Place all relevant PDF documents for that collection into this PDFs folder.

        Example: Challenge_1b/Collection 1/PDFs/my_document.pdf

    Ensure a challenge1b_input.json file is present in each collection directory. This file should specify the persona, the job_to_be_done, and list the input_documents (filenames of the PDFs) to be analyzed for that collection.

        Example challenge1b_input.json structure:

        {
          "persona": "Investment Analyst",
          "job_to_be_done": "Analyze revenue trends and market positioning.",
          "input_documents": [
            "Company_A_Report.pdf",
            "Company_B_Report.pdf"
          ]
        }

        Note: The provided run.py currently uses hardcoded persona and job_to_be_done for simplicity during local testing (Travel Planner and Plan a trip of 4 days for a group of 10 college friends.). For processing multiple distinct collections as implied by challenge1b_input.json, you would modify run.py's main block to dynamically read these from the JSON file for each collection.

4. Execution

To run the analysis, execute the main script from the repository's root directory:

python Challenge_1b/run.py

The script will provide progress updates in the terminal.
5. Output Retrieval

Upon completion, the processing results for each collection will be saved as output.json within the designated output directory (e.g., Challenge_1b/Outputs/).

The output JSON file adheres to the specified challenge requirements, containing:

    metadata: Details about the input documents, persona, job, and processing timestamp.

    extracted_sections: A ranked list of the most relevant high-level sections, including document name, section title, importance rank, and page number.

    subsection_analysis: More granular, concise text snippets from within the extracted sections, providing refined insights, along with document name and page number.

✅ Compliance & Performance

Our solution is engineered to meet the stringent hackathon requirements:

    CPU-Only Execution: Designed to run purely on CPU, with no GPU dependencies.

    Model Size: The all-MiniLM-L6-v2 Sentence Transformer model is approximately 90 MB, well within the le200MB (Round 1A) and le1GB (Round 1B) limits.

    Offline Operation: The model is loaded locally, ensuring no network or internet calls are made during execution after initial setup.

    Execution Time: Optimized for performance. Our current sequential processing with batch embedding aims to meet the le60 seconds for 3-5 documents constraint.

🏅 Criteria Alignment

    Our solution directly addresses the hackathon's criteria for Round 1B:

    Section Relevance: Achieved through advanced semantic similarity ranking of logically extracted sections against the job description.

    Sub-Section Relevance: Enhanced by intelligently extracting and ranking the most relevant sentences within the identified sections, providing granular and high-quality insights.

🔮 Future Enhancements

    Multi-Lingual Support: Extend the SentenceTransformer model to handle multilingual documents (e.g., Japanese), potentially earning bonus points.

    Advanced Sectioning: Explore more sophisticated layout analysis techniques (e.g., visual cues, document structure learning) to further improve heading detection beyond font sizes.

    Interactive WebApp: Integrate this backend logic with Adobe's PDF Embed API to build a user-friendly web application for interactive reading.

    Dynamic Input Handling: Implement command-line arguments or more robust configuration file parsing for persona and job-to-be-done to make the solution more flexible for testing different scenarios.

🤝 Team ByteMe

    Members: 
        1. Supriya Srivastava
        2. Gouri Sharma
        3. Pratyush Dube

This README was crafted with precision for the Adobe India Hackathon 2025.