import modal
import os
from typing import Dict, List, Optional
import json
import requests
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

# Create Modal app
app = modal.App("lease-hunyuan-system")

# Define image with all dependencies including Hunyuan-MT-7B
image = modal.Image.debian_slim().pip_install([
    "openai>=1.3.0",
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
    "bitsandbytes>=0.41.0",  # For model quantization
    "sentencepiece>=0.1.99",  # For tokenization
    "protobuf>=3.20.0"
]).apt_install([
    "git",
    "git-lfs"
])

# Define secrets
secrets = [
    modal.Secret.from_name("openai-secret"),
    modal.Secret.from_name("pinecone-secret"),
    modal.Secret.from_name("huggingface-secret")  # For accessing Hunyuan model
]

# Global model storage
hunyuan_model = None
hunyuan_tokenizer = None

@app.function(
    image=image, 
    secrets=secrets, 
    gpu="A10G",  # GPU required for Hunyuan-MT-7B
    timeout=1800,  # 30 minutes for model loading
    memory=32768   # 32GB memory for 7B model
)
def load_hunyuan_model():
    """Load Hunyuan-MT-7B model (runs once, cached)"""
    global hunyuan_model, hunyuan_tokenizer
    
    if hunyuan_model is None:
        print("🔄 Loading Hunyuan-MT-7B model...")
        
        model_name = "tencent/Hunyuan-MT-7B"
        
        try:
            # Load tokenizer
            print("📝 Loading tokenizer...")
            hunyuan_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_auth_token=os.environ.get("HUGGINGFACE_TOKEN")
            )
            
            # Load model with optimizations
            print("🧠 Loading model with quantization...")
            hunyuan_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Half precision for memory efficiency
                device_map="auto",  # Automatic device placement
                load_in_8bit=True,  # 8-bit quantization to reduce memory
                use_auth_token=os.environ.get("HUGGINGFACE_TOKEN")
            )
            
            print("✅ Hunyuan-MT-7B loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading Hunyuan model: {e}")
            # Fallback to a smaller translation model if needed
            print("🔄 Falling back to smaller translation model...")
            hunyuan_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
            hunyuan_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
    
    return hunyuan_model, hunyuan_tokenizer

@app.function(image=image, secrets=secrets, gpu="A10G", timeout=600)
def detect_language_and_translate(text: str) -> Dict:
    """Detect language and translate using Hunyuan-MT-7B"""
    from langdetect import detect, DetectorFactory
    
    # Set seed for consistent language detection
    DetectorFactory.seed = 0
    
    try:
        # Detect source language
        detected_lang = detect(text)
        print(f"🔍 Detected language: {detected_lang}")
        
        # If already English, return as-is
        if detected_lang == 'en':
            return {
                "original_language": detected_lang,
                "translated_text": text,
                "translation_needed": False,
                "translation_method": "no_translation_needed"
            }
        
        # Load Hunyuan model
        model, tokenizer = load_hunyuan_model.remote()
        
        # Prepare translation prompt for Hunyuan
        # Hunyuan-MT format: "Translate from {source_lang} to English: {text}"
        lang_map = {
            'zh': 'Chinese',
            'zh-cn': 'Chinese',
            'zh-tw': 'Chinese',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ru': 'Russian',
            'ar': 'Arabic'
        }
        
        source_language = lang_map.get(detected_lang, 'Unknown')
        
        if source_language == 'Unknown':
            # Use OpenAI as fallback for unsupported languages
            return translate_with_openai_fallback(text, detected_lang)
        
        # Format prompt for Hunyuan-MT
        translation_prompt = f"Translate from {source_language} to English: {text}"
        
        print(f"🌐 Translating {source_language} to English with Hunyuan-MT-7B...")
        
        # Tokenize input
        inputs = tokenizer(
            translation_prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        
        # Move to GPU if available
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
        
        # Decode translation
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the output (remove the prompt part)
        if "Translate from" in translated_text:
            # Extract only the translation part
            parts = translated_text.split(":")
            if len(parts) > 1:
                translated_text = parts[-1].strip()
        
        print(f"✅ Translation completed successfully")
        
        return {
            "original_language": detected_lang,
            "translated_text": translated_text,
            "translation_needed": True,
            "translation_method": "hunyuan_mt_7b",
            "source_language_full": source_language,
            "original_text_preview": text[:200] + "..." if len(text) > 200 else text
        }
        
    except Exception as e:
        print(f"❌ Hunyuan translation error: {e}")
        # Fallback to OpenAI translation
        return translate_with_openai_fallback(text, detected_lang)

def translate_with_openai_fallback(text: str, detected_lang: str) -> Dict:
    """Fallback translation using OpenAI when Hunyuan fails"""
    import openai
    
    try:
        print(f"🔄 Using OpenAI fallback for translation...")
        
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
        translation_prompt = f"""
        Translate the following text from {detected_lang} to English.
        Maintain the legal terminology and structure. Be precise with numbers, dates, and legal terms.
        
        Text to translate:
        {text[:3000]}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert legal document translator. Translate accurately while preserving all legal terminology, numbers, dates, and document structure."},
                {"role": "user", "content": translation_prompt}
            ],
            temperature=0
        )
        
        translated_text = response.choices[0].message.content
        
        return {
            "original_language": detected_lang,
            "translated_text": translated_text,
            "translation_needed": True,
            "translation_method": "openai_gpt4_fallback",
            "original_text_preview": text[:200] + "..." if len(text) > 200 else text
        }
        
    except Exception as e:
        print(f"❌ OpenAI fallback also failed: {e}")
        return {
            "original_language": detected_lang,
            "translated_text": text,
            "translation_needed": False,
            "translation_method": "failed",
            "error": str(e)
        }

@app.function(image=image, secrets=secrets, timeout=300)
def extract_lease_information_with_pages(text_with_pages: List[Dict]) -> Dict:
    """
    Extract lease information from OCR text with page numbers
    Input: [{"page": 1, "text": "page 1 content"}, {"page": 2, "text": "page 2 content"}]
    """
    import openai
    
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # Combine all text but keep page references
    full_text = ""
    page_map = {}
    
    for page_data in text_with_pages:
        page_num = page_data.get("page", 1)
        page_text = page_data.get("text", "")
        
        # Add page markers to text
        page_marker = f"\n[PAGE {page_num}]\n"
        full_text += page_marker + page_text + "\n"
        
        # Map content to pages for later reference
        page_map[page_num] = page_text
    
    prompt = f"""
    Extract lease information from this multi-page document. For each piece of information, 
    note which page number it was found on.
    
    Return JSON in this format:
    {{
        "property_address": {{"value": "123 Main St", "page": 1}},
        "monthly_rent": {{"value": 1200, "page": 1}},
        "security_deposit": {{"value": 2400, "page": 1}},
        "lease_start_date": {{"value": "2024-01-01", "page": 1}},
        "lease_end_date": {{"value": "2024-12-31", "page": 1}},
        "lease_term_months": {{"value": 12, "page": 1}},
        "tenant_names": {{"value": ["John Doe"], "page": 1}},
        "landlord_name": {{"value": "Property Co", "page": 1}},
        "landlord_contact": {{"value": "555-0123", "page": 2}},
        "pet_policy": {{"value": "No pets", "page": 2}},
        "utilities_included": {{"value": ["water", "trash"], "page": 2}},
        "utilities_tenant_pays": {{"value": ["electricity"], "page": 2}},
        "parking_spaces": {{"value": "1 space", "page": 1}},
        "square_footage": {{"value": "850 sq ft", "page": 1}},
        "late_fee": {{"value": 50, "page": 3}},
        "grace_period_days": {{"value": 5, "page": 3}},
        "key_terms": {{"value": ["No smoking", "Quiet hours"], "page": 3}},
        "special_clauses": {{"value": ["Early termination clause"], "page": 4}},
        "move_in_date": {{"value": "2024-01-01", "page": 1}},
        "rent_due_date": {{"value": "1st of month", "page": 1}}
    }}
    
    Document text with page markers:
    {full_text[:6000]}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "Extract lease information and note the page number where each piece of information was found. Return only valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean JSON formatting
        if content.startswith('```json'):
            content = content[7:-3]
        elif content.startswith('```'):
            content = content[3:-3]
            
        extracted_info = json.loads(content)
        
        # Add page mapping for reference
        extracted_info["page_map"] = page_map
        extracted_info["total_pages"] = len(text_with_pages)
        
        return extracted_info
        
    except Exception as e:
        print(f"Extraction error: {e}")
        return {
            "error": str(e),
            "page_map": page_map,
            "total_pages": len(text_with_pages)
        }

@app.function(image=image, secrets=secrets, timeout=300)
def create_embeddings_with_pages(text_with_pages: List[Dict]) -> Dict:
    """Create embeddings for each page and combined document"""
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings for each page
    page_embeddings = {}
    all_text = ""
    
    for page_data in text_with_pages:
        page_num = page_data.get("page", 1)
        page_text = page_data.get("text", "")
        
        if page_text.strip():
            # Create embedding for this page
            page_embedding = model.encode(page_text).tolist()
            page_embeddings[f"page_{page_num}"] = page_embedding
            
            # Add to combined text
            all_text += f" {page_text}"
    
    # Create combined document embedding
    combined_embedding = model.encode(all_text).tolist()
    
    return {
        "combined_embedding": combined_embedding,
        "page_embeddings": page_embeddings,
        "total_pages": len(text_with_pages)
    }

@app.function(image=image, secrets=secrets, timeout=300)
def store_in_pinecone_with_pages(document_id: str, embeddings_data: Dict, metadata: Dict) -> Dict:
    """Store document and page embeddings in Pinecone"""
    from pinecone import Pinecone
    
    try:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index("lease-documents")
        
        vectors_to_upsert = []
        
        # Store main document
        main_metadata = {}
        for key, value in metadata.items():
            if value is not None:
                if isinstance(value, (list, dict)):
                    str_value = json.dumps(value)[:1000]
                else:
                    str_value = str(value)[:1000]
                main_metadata[key] = str_value
        
        main_metadata["document_type"] = "full_document"
        main_metadata["searchable_content"] = f"{main_metadata.get('property_address', '')} {main_metadata.get('tenant_names', '')} {main_metadata.get('landlord_name', '')}"
        
        vectors_to_upsert.append({
            "id": document_id,
            "values": embeddings_data["combined_embedding"],
            "metadata": main_metadata
        })
        
        # Store individual pages
        for page_key, page_embedding in embeddings_data["page_embeddings"].items():
            page_num = page_key.split("_")[1]
            page_id = f"{document_id}_page_{page_num}"
            
            page_metadata = {
                **main_metadata,
                "document_type": "page",
                "page_number": int(page_num),
                "parent_document_id": document_id,
                "page_text_preview": metadata.get("page_map", {}).get(int(page_num), "")[:500]
            }
            
            vectors_to_upsert.append({
                "id": page_id,
                "values": page_embedding,
                "metadata": page_metadata
            })
        
        # Batch upsert
        index.upsert(vectors=vectors_to_upsert)
        
        print(f"Stored {len(vectors_to_upsert)} vectors for document {document_id}")
        return {
            "success": True, 
            "document_id": document_id,
            "vectors_stored": len(vectors_to_upsert),
            "pages_stored": len(embeddings_data["page_embeddings"])
        }
        
    except Exception as e:
        print(f"Pinecone storage error: {e}")
        return {"success": False, "error": str(e)}

@app.function(image=image, secrets=secrets, gpu="A10G", timeout=900)
@modal.web_endpoint(method="POST")
def process_lease_with_hunyuan(request_data: Dict) -> Dict:
    """
    Main endpoint for processing lease documents with Hunyuan translation
    Input: {
        "ocr_pages": [{"page": 1, "text": "original OCR text"}, ...],
        "filename": "lease.pdf",
        "file_id": "unique_id",
        "metadata": {...}
    }
    """
    try:
        ocr_pages = request_data.get("ocr_pages", [])
        filename = request_data.get("filename", "unknown.pdf")
        file_id = request_data.get("file_id")
        additional_metadata = request_data.get("metadata", {})
        
        if not ocr_pages:
            return {
                "success": False,
                "error": "No OCR pages provided"
            }
        
        print(f"🔄 Processing {filename} with {len(ocr_pages)} pages using Hunyuan-MT-7B")
        
        # Step 1: Translate each page with Hunyuan
        translated_pages = []
        translation_summary = {
            "total_pages": len(ocr_pages),
            "pages_translated": 0,
            "translation_method": "hunyuan_mt_7b",
            "languages_detected": set()
        }
        
        for page_data in ocr_pages:
            page_num = page_data.get("page", 1)
            page_text = page_data.get("text", "")
            
            if page_text.strip():
                # Translate this page
                translation_result = detect_language_and_translate.remote(page_text)
                
                translated_pages.append({
                    "page": page_num,
                    "text": translation_result.get("translated_text", page_text),
                    "original_text": page_text,
                    "translation_info": translation_result
                })
                
                translation_summary["languages_detected"].add(translation_result.get("original_language", "unknown"))
                if translation_result.get("translation_needed", False):
                    translation_summary["pages_translated"] += 1
            else:
                # Empty page
                translated_pages.append({
                    "page": page_num,
                    "text": page_text,
                    "original_text": page_text,
                    "translation_info": {"translation_needed": False}
                })
        
        # Convert set to list for JSON serialization
        translation_summary["languages_detected"] = list(translation_summary["languages_detected"])
        
        # Step 2: Extract lease information with page references
        lease_info = extract_lease_information_with_pages.remote(translated_pages)
        
        # Step 3: Create embeddings for pages and combined document
        embeddings_data = create_embeddings_with_pages.remote(translated_pages)
        
        # Step 4: Prepare document metadata
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        document_id = file_id or f"lease_{timestamp}_{filename.replace('.pdf', '').replace(' ', '_')}"
        
        # Combine all metadata
        full_metadata = {
            "filename": filename,
            "file_id": file_id,
            "upload_date": datetime.now().isoformat(),
            "processing_timestamp": timestamp,
            "total_pages": len(ocr_pages),
            "translation_summary": translation_summary,
            "hunyuan_translation": True,
            **additional_metadata,
            **lease_info
        }
        
        # Step 5: Store in Pinecone with page references
        storage_result = store_in_pinecone_with_pages.remote(
            document_id, 
            embeddings_data, 
            full_metadata
        )
        
        return {
            "success": True,
            "document_id": document_id,
            "translation_summary": translation_summary,
            "extracted_info": lease_info,
            "embeddings_info": {
                "total_pages": embeddings_data.get("total_pages"),
                "pages_embedded": len(embeddings_data.get("page_embeddings", {}))
            },
            "storage_result": storage_result,
            "processing_timestamp": datetime.now().isoformat(),
            "translation_method": "hunyuan_mt_7b"
        }
        
    except Exception as e:
        print(f"Processing error: {e}")
        return {
            "success": False,
            "error": f"Processing failed: {str(e)}",
            "translation_method": "hunyuan_mt_7b"
        }

@app.function(image=image, secrets=secrets, timeout=300)
@modal.web_endpoint(method="POST")
def query_documents_with_pages(request_data: Dict) -> Dict:
    """Query documents and get page references"""
    from pinecone import Pinecone
    import openai
    
    try:
        question = request_data.get("question", "")
        if not question:
            return {"error": "No question provided"}
        
        print(f"Query: {question}")
        
        # Create embedding for question
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        question_embedding = model.encode(question).tolist()
        
        # Search Pinecone
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index("lease-documents")
        
        search_results = index.query(
            vector=question_embedding,
            top_k=10,
            include_metadata=True
        )
        
        if not search_results.matches:
            return {
                "question": question,
                "answer": "No relevant lease documents found.",
                "sources": [],
                "page_references": []
            }
        
        # Organize results by document and pages
        documents = {}
        page_references = []
        
        for match in search_results.matches:
            metadata = match.metadata
            doc_type = metadata.get("document_type", "full_document")
            
            if doc_type == "page":
                page_references.append({
                    "document_id": metadata.get("parent_document_id"),
                    "filename": metadata.get("filename"),
                    "page_number": metadata.get("page_number"),
                    "relevance_score": round(match.score, 3),
                    "text_preview": metadata.get("page_text_preview", "")[:200],
                    "translation_method": metadata.get("translation_summary", {}).get("translation_method", "unknown")
                })
            
            # Collect document info
            doc_id = metadata.get("parent_document_id") or match.id
            if doc_id not in documents:
                documents[doc_id] = {
                    "filename": metadata.get("filename"),
                    "property_address": metadata.get("property_address"),
                    "monthly_rent": metadata.get("monthly_rent"),
                    "tenant_names": metadata.get("tenant_names"),
                    "translation_method": metadata.get("translation_summary", {}).get("translation_method", "unknown"),
                    "pages_found": []
                }
            
            if doc_type == "page":
                documents[doc_id]["pages_found"].append(metadata.get("page_number"))
        
        # Build context for AI
        context_parts = []
        sources = []
        
        for doc_id, doc_info in documents.items():
            context_part = f"""
Document: {doc_info['filename']}
Property: {doc_info.get('property_address', 'Not specified')}
Monthly Rent: ${doc_info.get('monthly_rent', 'Not specified')}
Tenants: {doc_info.get('tenant_names', 'Not specified')}
Translation: {doc_info.get('translation_method', 'Unknown')}
Relevant Pages: {', '.join(map(str, sorted(doc_info['pages_found']))) if doc_info['pages_found'] else 'Multiple'}
            """
            context_parts.append(context_part)
            
            sources.append({
                "document_id": doc_id,
                "filename": doc_info["filename"],
                "property_address": doc_info.get("property_address"),
                "monthly_rent": doc_info.get("monthly_rent"),
                "tenant_names": doc_info.get("tenant_names"),
                "translation_method": doc_info.get("translation_method"),
                "pages_referenced": sorted(doc_info["pages_found"]) if doc_info["pages_found"] else []
            })
        
        context = "\n".join(context_parts)
        
        # Generate answer with OpenAI
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant that answers questions about lease documents. 
                    Use the provided context to give accurate answers.
                    ALWAYS mention specific page numbers when referencing information.
                    Format your response to include page citations like: "According to page 2 of lease_document.pdf..."
                    If documents were translated, mention the translation method when relevant.
                    If multiple documents contain relevant info, organize your answer by document."""
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContext from lease documents:\n{context}"
                }
            ],
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "page_references": page_references,
            "documents_searched": len(documents),
            "translation_info": {
                "hunyuan_translated_docs": len([s for s in sources if s.get("translation_method") == "hunyuan_mt_7b"])
            }
        }
        
    except Exception as e:
        print(f"Query error: {e}")
        return {"error": f"Query failed: {str(e)}"}

@app.function(image=image)
@modal.web_endpoint(method="GET")
def health_check() -> Dict:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "lease-hunyuan-system",
        "translation_model": "tencent/Hunyuan-MT-7B",
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "/process-lease-with-hunyuan (POST) - Process with Hunyuan translation",
            "/query-documents-with-pages (POST) - Query with page refs", 
            "/health-check (GET) - This endpoint"
        ]
    }

@app.function(image=image)
@modal.web_endpoint(method="GET")
def list_all_documents() -> Dict:
    """List all documents in the database"""
    from pinecone import Pinecone
    
    try:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index("lease-documents")
        
        # Get index statistics
        stats = index.describe_index_stats()
        
        return {
            "total_documents": stats.total_vector_count,
            "index_fullness": stats.index_fullness,
            "translation_model": "tencent/Hunyuan-MT-7B"
        }
        
    except Exception as e:
        return {"error": f"Failed to get document list: {str(e)}"}
