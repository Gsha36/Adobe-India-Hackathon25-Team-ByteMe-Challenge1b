import os
import json
import fitz  # PyMuPDF
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_text_blocks_with_metadata(pdf_path):
    doc = fitz.open(pdf_path)
    all_blocks = []
    for page_num, page in tqdm(enumerate(doc, start=1), total=doc.page_count, desc=f"Extracting blocks from {os.path.basename(pdf_path)}"):
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block["type"] == 0:
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

def split_text_into_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def refine_subsection_text(full_section_text, job_description, num_sentences=3):
    sentences = split_text_into_sentences(full_section_text)
    if not sentences:
        return ""
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
    job_embedding = model.encode([job_description], convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, show_progress_bar=False, convert_to_tensor=True)
    similarities = cosine_similarity(job_embedding.cpu().numpy(), sentence_embeddings.cpu().numpy())[0]
    top_sentences = [s for s, _ in sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True)[:num_sentences]]
    return " ".join(top_sentences)

def rank_global_sections(all_sections_data, job_description, top_k=5):
    job_embedding = model.encode([job_description], convert_to_tensor=True)
    if not all_sections_data:
        return []
    section_embeddings = model.encode([s["text"] for s in all_sections_data], show_progress_bar=True, convert_to_tensor=True)
    similarities = cosine_similarity(job_embedding.cpu().numpy(), section_embeddings.cpu().numpy())[0]

    scored = []
    for i, sim in enumerate(similarities):
        sec = all_sections_data[i]
        scored.append({
            "document": sec["document"],
            "page_number": sec["page_number"],
            "text": sec["text"],
            "section_title_candidate": sec["section_title_candidate"],
            "score": float(sim)
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return [{
        "document": s["document"],
        "page_number": s["page_number"],
        "section_title": s["section_title_candidate"][:100],
        "importance_rank": i + 1,
        "refined_text": s["text"]
    } for i, s in enumerate(scored[:top_k])]

def process_collection(input_pdf_folder, output_dir, input_json_path, top_k_global=5, num_sentences_for_subsection=3):
    with open(input_json_path, 'r') as f:
        input_data = json.load(f)

    persona = input_data["persona"]["role"]
    job_description = input_data["job_to_be_done"]["task"]
    filenames = [doc["filename"] for doc in input_data["documents"]]

    output = {
        "metadata": {
            "input_documents": filenames,
            "persona": persona,
            "job_to_be_done": job_description,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }

    os.makedirs(output_dir, exist_ok=True)

    all_sections = []
    for filename in filenames:
        pdf_path = os.path.join(input_pdf_folder, filename)
        print(f"\nProcessing: {filename}")
        if not os.path.exists(pdf_path):
            print(f"  [!] File not found: {pdf_path}")
            continue
        sections = extract_sections_with_content(pdf_path)
        all_sections.extend(sections)

    print(f"\nRanking sections for task: '{job_description}'")
    ranked = rank_global_sections(all_sections, job_description, top_k=top_k_global)

    for section in ranked:
        output["extracted_sections"].append({
            "document": section["document"],
            "section_title": section["section_title"],
            "importance_rank": section["importance_rank"],
            "page_number": section["page_number"]
        })

        refined = refine_subsection_text(section["refined_text"], job_description, num_sentences=num_sentences_for_subsection)
        output["subsection_analysis"].append({
            "document": section["document"],
            "refined_text": refined,
            "page_number": section["page_number"]
        })

    output_file_path = os.path.join(output_dir, "challenge1b_output.json")
    with open(output_file_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Output written to {output_file_path}")

# --- ENTRY POINT ---
if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Process all collections
    collections = ["Collection 1", "Collection 2", "Collection 3"]
    
    for collection in collections:
        print(f"\n{'='*50}")
        print(f"Processing {collection}")
        print(f"{'='*50}")
        
        # Paths for current collection
        pdf_folder = os.path.join(script_dir, collection, "PDFs")
        output_directory = os.path.join(script_dir, collection)
        input_json_path = os.path.join(script_dir, collection, "challenge1b_input.json")
        
        # Check if the collection folder exists
        if not os.path.exists(os.path.join(script_dir, collection)):
            print(f"⚠️ Collection folder not found: {collection}")
            continue
            
        # Check if input files exist
        if not os.path.exists(input_json_path):
            print(f"⚠️ Input JSON not found: {input_json_path}")
            continue
            
        if not os.path.exists(pdf_folder):
            print(f"⚠️ PDF folder not found: {pdf_folder}")
            continue
        
        print(f"PDFs: {pdf_folder}")
        print(f"Output: {output_directory}")
        print(f"Using input.json: {input_json_path}")
        
        try:
            process_collection(pdf_folder, output_directory, input_json_path)
        except Exception as e:
            print(f"❌ Error processing {collection}: {str(e)}")
            continue
    
    print(f"\n{'='*50}")
    print("All collections processed!")
    print(f"{'='*50}")