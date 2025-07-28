import os
import json
import fitz  # PyMuPDF
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import re # For sentence splitting

# Load the model once globally.
# IMPORTANT: Ensure this model ("all-MiniLM-L6-v2") is downloaded/available offline
# within your Docker image. Your Dockerfile should include steps to pre-download it.
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- START: Core Logic Adapted from Round 1A Outline Extraction ---

def get_text_blocks_with_metadata(pdf_path):
    """
    Extracts all text blocks from a PDF document with their page number, text, font size,
    and bold status. This information is crucial for identifying headings.
    """
    doc = fitz.open(pdf_path)
    all_blocks = []
    
    for page_num, page in tqdm(enumerate(doc, start=1), total=doc.page_count, desc=f"Extracting blocks from {os.path.basename(pdf_path)}"):
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block["type"] == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        font_size = span.get("size", 0)
                        is_bold = "bold" in span.get("font", "").lower() 
                        if text:
                            all_blocks.append({
                                "page_number": page_num,
                                "text": text,
                                "font_size": font_size,
                                "is_bold": is_bold,
                                "bbox": span.get("bbox")
                            })
    doc.close()
    return all_blocks

def extract_sections_with_content(pdf_path):
    """
    Analyzes text blocks from a PDF to identify logical sections (heading + content)
    based on font size and bold heuristics, aiming for structured sections like a document outline.
    """
    all_blocks = get_text_blocks_with_metadata(pdf_path)
    
    logical_sections_with_content = []
    current_section_title_candidate = "Document Introduction"
    current_section_text_parts = []
    current_section_start_page = 1

    font_sizes = sorted(list(set([b["font_size"] for b in all_blocks])))
    
    H1_THRESHOLD = max(font_sizes[-1], 18) if font_sizes else 18
    H2_THRESHOLD = font_sizes[-min(2, len(font_sizes))] if len(font_sizes) >= 2 else 14
    H3_THRESHOLD = font_sizes[-min(3, len(font_sizes))] if len(font_sizes) >= 3 else 12

    H1_THRESHOLD = max(H1_THRESHOLD, H2_THRESHOLD + 2)
    H2_THRESHOLD = max(H2_THRESHOLD, H3_THRESHOLD + 1)
    
    H1_THRESHOLD += 0.1
    H2_THRESHOLD += 0.1
    H3_THRESHOLD += 0.1

    for i, block in enumerate(all_blocks):
        text = block["text"]
        font_size = block["font_size"]
        page_num = block["page_number"]
        is_bold = block["is_bold"]

        is_potential_heading = False
        if 2 < len(text.split()) < 20:
            if font_size >= H1_THRESHOLD:
                is_potential_heading = True
            elif font_size >= H2_THRESHOLD and is_bold:
                is_potential_heading = True
            elif font_size >= H3_THRESHOLD and is_bold and len(text.split()) < 15:
                is_potential_heading = True
        
        if i > 0 and block["page_number"] != all_blocks[i-1]["page_number"] and \
           len(text.split()) < 25 and font_size >= H3_THRESHOLD and not text.endswith('.'):
             is_potential_heading = True

        if is_potential_heading and len(" ".join(current_section_text_parts).strip()) > 50:
            logical_sections_with_content.append({
                "document": os.path.basename(pdf_path),
                "title": current_section_title_candidate,
                "text": "\n".join(current_section_text_parts).strip(),
                "page_number": current_section_start_page
            })
            current_section_title_candidate = text
            current_section_text_parts = []
            current_section_start_page = page_num
        else:
            current_section_text_parts.append(text)
            if not logical_sections_with_content and not current_section_text_parts and i == 0:
                current_section_start_page = page_num

    if current_section_text_parts:
        full_section_text = "\n".join(current_section_text_parts).strip()
        if len(full_section_text) > 50:
            logical_sections_with_content.append({
                "document": os.path.basename(pdf_path),
                "title": current_section_title_candidate,
                "text": full_section_text,
                "page_number": current_section_start_page
            })

    sections_data_for_ranking = []
    for section in logical_sections_with_content:
        sections_data_for_ranking.append({
            "document": section["document"],
            "page_number": section["page_number"],
            "text": section["text"],
            "section_title_candidate": section["title"]
        })
    
    return sections_data_for_ranking

# --- END: Core Logic Adapted from Round 1A Outline Extraction ---

def split_text_into_sentences(text):
    """
    Splits a text block into individual sentences.
    """
    # Simple regex for splitting into sentences (might need refinement for complex cases)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def refine_subsection_text(full_section_text, job_description, num_sentences=3):
    """
    Takes a full section text and extracts the most relevant sentences
    based on the job description.
    """
    sentences = split_text_into_sentences(full_section_text)
    if not sentences:
        return ""

    if len(sentences) <= num_sentences:
        return " ".join(sentences) # Return all if less than or equal to desired count

    job_embedding = model.encode([job_description], convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, show_progress_bar=False, convert_to_tensor=True)

    similarities = cosine_similarity(job_embedding.cpu().numpy(), sentence_embeddings.cpu().numpy())[0]

    # Pair sentences with their scores and sort
    scored_sentences = sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True)

    # Get the top N most relevant sentences
    top_sentences = [s[0] for s in scored_sentences[:num_sentences]]

    # Optionally, reorder them to appear in their original document order
    # This requires more complex tracking of original indices. For simplicity, we'll return as is.
    
    return " ".join(top_sentences)


def rank_global_sections(all_sections_data, job_description, top_k=5):
    """
    Ranks all extracted sections from across all documents based on semantic similarity
    to the job description.
    """
    job_embedding = model.encode([job_description], convert_to_tensor=True) 
    
    if not all_sections_data:
        return []

    section_texts_for_embedding = [item["text"] for item in all_sections_data]
    
    section_embeddings = model.encode(section_texts_for_embedding, show_progress_bar=True, convert_to_tensor=True)

    similarities = cosine_similarity(job_embedding.cpu().numpy(), section_embeddings.cpu().numpy())[0]

    scored_sections = []
    for i, score in enumerate(similarities):
        if isinstance(score, np.float32):
            score = float(score)
        
        section_info = all_sections_data[i]
        scored_sections.append({
            "document": section_info["document"],
            "page_number": section_info["page_number"],
            "text": section_info["text"],
            "section_title_candidate": section_info["section_title_candidate"],
            "score": score
        })

    scored_sections.sort(key=lambda x: x["score"], reverse=True)

    top_sections = scored_sections[:top_k]

    final_ranked_output = []
    for rank_idx, section in enumerate(top_sections):
        section_title = section["section_title_candidate"]
        
        if section_title == "Document Introduction" or not section_title.strip():
            lines = [line.strip() for line in section["text"].split('\n') if line.strip()]
            section_title = lines[0] if lines else "Untitled Section"
        
        if section_title.startswith('\u2022') or section_title.startswith('-') or section_title.startswith('•'):
            section_title = section_title[1:].strip()
            
        if len(section_title) > 100:
            section_title = section_title[:97] + "..."


        final_ranked_output.append({
            "document": section["document"],
            "page_number": section["page_number"],
            "section_title": section_title,
            "importance_rank": rank_idx + 1,
            "refined_text": section["text"] # Keep full text here for refining later
        })
    
    return final_ranked_output


def process_collection(input_pdf_folder, output_dir, persona, job_description, top_k_global=5, num_sentences_for_subsection=3):
    """
    Main function to process a collection of PDFs, extract relevant sections,
    and output them in the specified JSON format.
    
    Args:
        input_pdf_folder (str): Path to the folder containing input PDFs.
        output_dir (str): Path to the directory where the output JSON will be saved.
        persona (str): The persona for the analysis.
        job_description (str): The job-to-be-done description.
        top_k_global (int): Number of top globally relevant sections to extract.
        num_sentences_for_subsection (int): Number of most relevant sentences to extract
                                             for the 'refined_text' in subsection_analysis.
    """
    output = {
        "metadata": {
            "input_documents": [],
            "persona": persona,
            "job_to_be_done": job_description,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    os.makedirs(output_dir, exist_ok=True)

    pdf_files = [f for f in os.listdir(input_pdf_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {input_pdf_folder}")
        return

    all_sections_across_docs = []
    for filename in pdf_files:
        pdf_path = os.path.join(input_pdf_folder, filename)
        print(f"\nGathering sections from: {filename}")
        sections_from_doc = extract_sections_with_content(pdf_path)
        all_sections_across_docs.extend(sections_from_doc)
        output["metadata"]["input_documents"].append(filename)

    print(f"\nRanking all collected sections globally for '{job_description}'...")
    globally_ranked_sections = rank_global_sections(all_sections_across_docs, job_description, top_k=top_k_global)

    for section_info in globally_ranked_sections:
        output["extracted_sections"].append({
            "document": section_info["document"],
            "section_title": section_info["section_title"],
            "importance_rank": section_info["importance_rank"],
            "page_number": section_info["page_number"]
        })
        
        # Refine the text for subsection_analysis by extracting most relevant sentences
        refined_sub_text = refine_subsection_text(section_info["refined_text"], job_description, num_sentences=num_sentences_for_subsection)

        output["subsection_analysis"].append({
            "document": section_info["document"],
            "refined_text": refined_sub_text,
            "page_number": section_info["page_number"]
        })

    out_path = os.path.join(output_dir, "output.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Output saved to: {out_path}")

# --- ENTRY POINT FOR EXECUTION ---
if __name__ == "__main__":
    input_pdf_folder = os.environ.get("INPUT_DIR", "D:\\Coding\\Adobe-India-Hackathon25-Team-ByteMe\\Challenge_1b\\Collection 1\\PDFs")
    output_directory = os.environ.get("OUTPUT_DIR", "D:\\Coding\\Adobe-India-Hackathon25-Team-ByteMe\\Challenge_1b\\Outputs")

    persona = "Travel Planner"
    job_description = "Plan a trip of 4 days for a group of 10 college friends."

    print(f"Processing PDFs from: {input_pdf_folder}")
    print(f"Saving output to: {output_directory}")
    print(f"Persona: {persona}")
    print(f"Job Description: {job_description}")

    # Adjust num_sentences_for_subsection to control the length of refined_text
    # For a very concise output, try 2-3 sentences. For slightly more detail, 4-5.
    process_collection(input_pdf_folder, output_directory, persona, job_description, 
                       top_k_global=5, num_sentences_for_subsection=3)