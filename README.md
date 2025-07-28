# Adobe-India-Hackathon25-Team-ByteMe-Round1b
## Usage Instructions

### 1. Data Preparation

Place all relevant PDF documents in the `PDFs` subfolder within each collection directory (`Collection 1`, `Collection 2`, `Collection 3`).
Ensure each collection directory contains a `challenge1b_input.json` file. This file must specify the persona, the job-to-be-done, and list the filenames of the PDFs to be analyzed.

### 2. Environment Setup

Open a terminal in the root directory of the repository.
Install the required Python packages by executing:
```powershell
pip install -r Challenge_1b/requirements.txt
```

### 3. Execution

Run the main analysis script with the following command:
```powershell
python Challenge_1b/run.py
```
The script will automatically process each collection, analyze the documents, and generate the required output.

### 4. Output Retrieval

Upon completion, the results for each collection will be saved as `challenge1b_output.json` within the respective collection directory.
The output file will contain metadata, extracted and ranked document sections, and refined sub-section analyses, as specified in the challenge requirements.