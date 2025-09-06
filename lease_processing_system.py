import modal
import os
from typing import Dict, List, Optional
import json
import requests
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Create Modal app
app = modal.App("lease-hunyuan-system")

# Define image with Hunyuan dependencies
image = modal.Image.debian_slim().pip_install([
    "pinecone-client>=3.0.0", 
    "sentence-transformers>=2.2.2",
    "pydantic>=2.0.0",
    "fastapi>=0.100.0",
    "python-multipart>=0.0.6",
    "requests>=2.31.0",
    "langdetect>=1.0.9",
    "transformers>=4.35.0",
    "torch>=2.0.0",
    "accelerate>=0.21.0",
    "bitsandbytes>=0.41.0",
    "sentencepiece>=0.1.99",
    "protobuf>=3.20.0"
]).apt_install([
    "git",
    "git-lfs"
])

# Define secrets
secrets = [
    modal.Secret.from_name("pinecone-secret"),
    modal.Secret.from_name("huggingface-secret")
]

# Global model storage
hunyuan_model = None
hunyuan_tokenizer = None

@app.function(
    image=image, 
    secrets=secrets, 
    gpu="A10G",
    timeout=1800,
    memory=32768
)
def load_hunyuan_model():
    """Load Hunyuan-MT-7B model ONCE"""
    global hunyuan_model, hunyuan_tokenizer
    
    if hunyuan_model is None:
        print("🔄 Loading Hunyuan-MT-7B model...")
        
        model_name = "tencent/Hunyuan-MT-7B"
        
        hunyuan_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_auth_token=os.environ.get("HUGGINGFACE_TOKEN")
        )
        
        hunyuan_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,
            use_auth_token=os.environ.get("HUGGINGFACE_TOKEN")
        )
        
        print("✅ Hunyuan-MT-7B loaded successfully!")
    
    return hunyuan_model, hunyuan_tokenizer

@app.function(image=image, secrets=secrets, gpu="A10G", timeout=600)
def translate_text_hunyuan(text: str) -> Dict:
    """Translate using Hunyuan-MT-7B - NO FALLBACKS"""
    from langdetect import detect, DetectorFactory
    
    DetectorFactory.seed = 0
    
    # Detect language
    detected_lang = detect(text)
    print(f"🔍 Detected language: {detected_lang}")
    
    # If English, return as-is
    if detected_lang == 'en':
        return {
            "original_language": detected_lang,
            "translated_text": text,
            "translation_needed": False,
            "translation_method": "no_translation_needed"
        }
    
    # Load Hunyuan model
    model, tokenizer = load_hunyuan_model.remote()
    
    # Language mapping
    lang_map = {
        'zh': 'Chinese', 'zh-cn': 'Chinese', 'zh-tw': 'Chinese',
        'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'it': 'Italian', 'pt': 'Portuguese', 'ja': 'Japanese',
        'ko': 'Korean', 'ru': 'Russian', 'ar': 'Arabic'
    }
    
    source_language = lang_map.get(detected_lang, detected_lang.upper())
    
    # Format for Hunyuan-MT
    translation_prompt = f"Translate from {source_language} to English: {text}"
    
    print(f"🌐 Translating {source_language} to English...")
    
    # Tokenize
    inputs = tokenizer(
        translation_prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True
    )
    
    # Move to GPU
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate translation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=2048,
            num_beams=4,
            temperature=0.7,
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean output
    if "Translate from" in translated_text:
        parts = translated_text.split(":")
        if len(parts) > 1:
            translated_text = parts[-1].strip()
    
    return {
        "original_language": detected_lang,
        "translated_text": translated_text,
        "translation_needed": True,
        "translation_method": "hunyuan_mt_7b",
        "source_language_full": source_language
    }

@app.function(image=image, secrets=secrets, gpu="A10G", timeout=600)
def extract_lease_info_hunyuan(text_with_pages: List[Dict]) -> Dict:
    """Extract lease info using Hunyuan - NO FALLBACKS"""
    
    # Load Hunyuan model
    model, tokenizer = load_hunyuan_model.remote()
    
    # Combine text with page markers
    full_text = ""
    page_map = {}
    
    for page_data in text_with_pages:
        page_num = page_data.get("page", 1)
        page_text = page_data.get("text", "")
        page_marker = f"\n[PAGE {page_num}]\n"
        full_text += page_marker + page_text + "\n"
        page_map[page_num] = page_text
    
    # Hunyuan extraction prompt
    extraction_prompt = f"""Extract lease information from this document and return as JSON. Include page numbers.

Format:
{{
    "property_address": {{"value": "123 Main St", "page": 1}},
    "monthly_rent": {{"value": 1200, "page": 1}},
    "security_deposit": {{"value": 2400, "page": 1}},
    "lease_start_date": {{"value": "2024-01-01", "page": 1}},
    "lease_end_date": {{"value": "2024-12-31", "page": 1}},
    "tenant_names": {{"value": ["John Doe"], "page": 1}},
    "landlord_name": {{"value": "Property Co", "page": 1}}
}}

Document:
{full_text[:3000]}

JSON:"""
    
    # Tokenize
    inputs = tokenizer(
        extraction_prompt,
        return_tensors="pt",
        max_length=2048,
        truncation=True,
        padding=True
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=4096,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1
    
    if json_start != -1 and json_end != -1:
        json_text = response_text[json_start:json_end]
        extracted_info = json.loads(json_text)
        extracted_info["page_map"] = page_map
        extracted_info["total_pages"] = len(text_with_pages)
        return extracted_info
    else:
        # If JSON extraction fails, return minimal structure
        return {
            "error": "Failed to extract structured information",
            "page_map": page_map,
            "total_pages": len(text_with_pages)
        }

@app.function(image=image, secrets=secrets, timeout=300)
def create_embeddings(text_with_pages: List[Dict]) -> Dict:
    """Create embeddings for pages"""
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    page_embeddings = {}
    all_text = ""
    
    for page_data in text_with_pages:
        page_num = page_data.get("page", 1)
        page_text = page_data.get("text", "")
        
        if page_text.strip():
            page_embedding = model.encode(page_text).tolist()
            page_embeddings[f"page_{page_num}"] = page_embedding
            all_text += f" {page_text}"
    
    combined_embedding = model.encode(all_text).tolist()
    
    return {
        "combined_embedding": combined_embedding,
        "page_embeddings": page_embeddings,
        "total_pages": len(text_with_pages)
    }

@app.function(image=image, secrets=secrets, timeout=300)
def store_in_pinecone(document_id: str, embeddings_data: Dict, metadata: Dict) -> Dict:
    """Store in Pinecone"""
    from pinecone import Pinecone
    
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index("lease-documents")
    
    vectors_to_upsert = []
    
    # Clean metadata
    main_metadata = {}
    for key, value in metadata.items():
        if value is not None:
            if isinstance(value, (list, dict)):
                str_value = json.dumps(value)[:1000]
            else:
                str_value = str(value)[:1000]
            main_metadata[key] = str_value
    
    main_metadata["document_type"] = "full_document"
    
    # Store main document
    vectors_to_upsert.append({
        "id": document_id,
        "values": embeddings_data["combined_embedding"],
        "metadata": main_metadata
    })
    
    # Store pages
    for page_key, page_embedding in embeddings_data["page_embeddings"].items():
        page_num = page_key.split("_")[1]
        page_id = f"{document_id}_page_{page_num}"
        
        page_metadata = {
            **main_metadata,
            "document_type": "page",
            "page_number": int(page_num),
            "parent_document_id": document_id
        }
        
        vectors_to_upsert.append({
            "id": page_id,
            "values": page_embedding,
            "metadata": page_metadata
        })
    
    # Upload to Pinecone
    index.upsert(vectors=vectors_to_upsert)
    
    return {
        "success": True, 
        "document_id": document_id,
        "vectors_stored": len(vectors_to_upsert)
    }

@app.function(image=image, secrets=secrets, gpu="A10G", timeout=900)
@modal.web_endpoint(method="POST")
def process_lease_with_hunyuan(request_data: Dict) -> Dict:
    """MAIN PROCESSING ENDPOINT"""
    
    ocr_pages = request_data.get("ocr_pages", [])
    filename = request_data.get("filename", "unknown.pdf")
    file_id = request_data.get("file_id")
    additional_metadata = request_data.get("metadata", {})
    
    if not ocr_pages:
        return {"success": False, "error": "No OCR pages provided"}
    
    print(f"🔄 Processing {filename} with {len(ocr_pages)} pages")
    
    # Step 1: Translate each page
    translated_pages = []
    for page_data in ocr_pages:
        page_num = page_data.get("page", 1)
        page_text = page_data.get("text", "")
        
        if page_text.strip():
            translation_result = translate_text_hunyuan.remote(page_text)
            translated_pages.append({
                "page": page_num,
                "text": translation_result.get("translated_text", page_text)
            })
        else:
            translated_pages.append({"page": page_num, "text": page_text})
    
    # Step 2: Extract lease info
    lease_info = extract_lease_info_hunyuan.remote(translated_pages)
    
    # Step 3: Create embeddings
    embeddings_data = create_embeddings.remote(translated_pages)
    
    # Step 4: Store in Pinecone
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    document_id = file_id or f"lease_{timestamp}_{filename.replace('.pdf', '').replace(' ', '_')}"
    
    full_metadata = {
        "filename": filename,
        "file_id": file_id,
        "upload_date": datetime.now().isoformat(),
        "total_pages": len(ocr_pages),
        "ai_system": "hunyuan_only",
        **additional_metadata,
        **lease_info
    }
    
    storage_result = store_in_pinecone.remote(document_id, embeddings_data, full_metadata)
    
    return {
        "success": True,
        "document_id": document_id,
        "extracted_info": lease_info,
        "storage_result": storage_result,
        "ai_system": "hunyuan_only"
    }

@app.function(image=image)
@modal.web_endpoint(method="GET")
def health_check() -> Dict:
    return {
        "status": "healthy",
        "service": "lease-hunyuan-only",
        "model": "tencent/Hunyuan-MT-7B",
        "endpoints": [
            "/process-lease-with-hunyuan (POST)",
            "/health-check (GET)"
        ]
    }
