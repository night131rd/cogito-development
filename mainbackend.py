import requests
from typing import Dict, List, Optional
import fitz  # PyMuPDF
from io import BytesIO
import re
from openai import OpenAI
import json
import time
import threading
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# INISIASI LLM
system_prompt = """"Anda adalah asisten pencarian akademik ahli yang menggunakan alur pemikiran terstruktur (chain of thought). Ikuti langkah-langkah berikut untuk input pengguna:

Langkah 1: Analisis Topik
- Identifikasi 1-2 topik utama dari paragraf  
- Contoh: "Dampak rokok dan upaya regulasi"
Langkah 2: Ekstraksi Kata Kunci 
- Daftarkan kata/frasa kunci (prioritaskan istilah ilmiah/medis)  
- Contoh: ["rokok", "kesehatan", "nikotin", "regulasi"]

Langkah 3: Formulasi Query
- Gabungkan 1-3 kata kunci terkuat menjadi query  
- Jika output hanya 1 kata kunci, tambahkan kata kunci yang relevan
- Contoh: "rokok  kesehatan regulasi"

Format Output Wajib:
{
    "query": "[hasil query dari Langkah 3]"
}

Aturan Tambahan:
1. Abaikan tahun/lokasi/keterangan spesifik  
2. Gunakan akronim umum ( 
3. Fokus pada konsep yang dapat dicari di database akademik
4. Pastikan output dalam bahasa yang sama dengan input pengguna

Contoh Input-Output:

Input: "Rokok adalah produk yang terbuat dari tembakau yang digulung atau dipadatkan dalam kertas, biasanya digunakan dengan cara dibakar dan dihisap asapnya.
Rokok mengandung berbagai zat kimia berbahaya, termasuk nikotin, tar, dan karbon monoksida, yang dapat menyebabkan berbagai masalah kesehatan serius, seperti penyakit jantung,
kanker paru-paru, stroke, dan gangguan pernapasan.Selain berdampak buruk bagi perokok aktif, asap rokok juga berbahaya bagi perokok pasif 
(orang yang menghirup asap rokok dari lingkungan sekitar). Oleh karena itu, banyak negara telah menerapkan regulasi ketat terkait rokok, termasuk larangan merokok di tempat umum,
pembatasan iklan rokok, dan kampanye kesehatan untuk mengurangi konsumsi rokok."  
Output: {"query": "rokok  kesehatan regulasi"} 
Input: "Padi di Indonesia"
Output: {"query": "Padi Indonesia"}
Input: "Factors Affecting Rice Grain Size and Weight"
Output: {"query": "Rice Grain Size Weight"}
Input: "The Impact of Climate Change on Coral Reefs"
Output: {"query": "Climate Change Coral Reefs"}

Berikan output dalam format JSON yang sesuai dengan aturan di atas.
{"query": "[hasil query dari Langkah 3]"}
Hanya itu yang perlu Anda kirimkan. Terima kasih!
"""

# API LLM
messages = [
        {"role": "system", "content": system_prompt},
]  # Store conversation history
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)
# Input pengguna
try:
        pencarian = input("Cognito Disini: ") 
        tahun = int(input("Masukkan tahun maksimal: "))
        tahun =  f"{tahun}-"
except ValueError as e:
        print(f"Invalid input: {e}")
        exit()

# Menambahkan input pengguna ke dalam list messages
messages.append({"role": "user", "content": pencarian})
try:
        completion = client.chat.completions.create(
                model="google/gemini-2.0-flash-thinking-exp:free",
                messages=messages,
                temperature= 2,
        )
        content = completion.choices[0].message.content.strip()
        print("OpenAI LLM:", content)
except Exception as e:
        print(f"Error with OpenAI API: {e}")
        exit()

# Membuat query dari output LLM
try:
        json_start = content.find('{')  # Find the start of the JSON object
        json_end = content.rfind('}') + 1  # Find the end of the JSON object
        json_string = content[json_start:json_end]  # Extract the JSON part
        parsed_json = json.loads(json_string)
        input_semantic = parsed_json["query"]
        print("Input Semantic Scholar:", input_semantic)
except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing JSON: {e}")
        exit()

# Counter untuk PDF yang diunduh
list_pdf = [] # Meyimpan teks hasil ekstraksi PDF dalam list
index = 0 # Menghitung jumlah PDF yang diunduh
# Berikan jawaban pengguna berdasarkan list_pdf menggunakan OpenAI LLM
class RAGSystem:
    def __init__(self):
        # Inisialisasi model embedding
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.document_chunks = []
        self.document_embeddings = []
        self.metadata = []

    def add_document(self, document: str, doc_index: int, title: str, year: str):
        """Memproses dan menambahkan dokumen langsung saat diunduh"""
        print(f"Menambahkan dokumen ke RAG system: {title} ({year})")
        
        # Membagi dokumen menjadi chunk
        chunks = self.chunk_document(document, chunk_size=500, overlap=100)
        
        new_chunks = []
        new_metadata = []
        
        for chunk_index, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Abaikan chunk yang terlalu pendek
                continue
                
            # Simpan chunk dan metadata
            new_chunks.append(chunk)
            new_metadata.append({
                "doc_index": doc_index,
                "chunk_index": chunk_index,
                "title": title,
                "year": year
            })
        
        # Buat embedding untuk chunk baru
        if new_chunks:
            new_embeddings = self.model.encode(new_chunks)
            
            # Tambahkan ke database
            self.document_chunks.extend(new_chunks)
            self.document_embeddings = np.vstack([self.document_embeddings, new_embeddings]) if len(self.document_embeddings) > 0 else new_embeddings
            self.metadata.extend(new_metadata)
            
            print(f"Berhasil menambahkan {len(new_chunks)} chunk dari dokumen {doc_index+1}")
        else:
            print("Tidak ada chunk yang valid untuk ditambahkan")
        
    def process_documents(self, documents: List[str]):
        """Memproses dokumen dan mempersiapkan untuk retrieval"""
        print("Memproses dokumen untuk RAG...")
        # Bersihkan list sebelumnya
        self.document_chunks = []
        self.document_embeddings = []
        self.metadata = []
        
        for doc_index, document in enumerate(documents):
            # Membagi dokumen menjadi chunk
            chunks = self.chunk_document(document, chunk_size=500, overlap=100)
            
            for chunk_index, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Abaikan chunk yang terlalu pendek
                    continue
                    
                # Simpan chunk dan metadata
                self.document_chunks.append(chunk)
                self.metadata.append({
                    "doc_index": doc_index,
                    "chunk_index": chunk_index
                })
        
        # Buat embedding untuk semua chunk
        if self.document_chunks:
            self.document_embeddings = self.model.encode(self.document_chunks)
            print(f"RAG system telah memproses {len(self.document_chunks)} chunk dari {len(documents)} dokumen")
        else:
            print("Tidak ada chunk yang valid untuk diproses")
    
    def chunk_document(self, document: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Membagi dokumen menjadi chunk dengan overlap"""
        # Split dokumen berdasarkan paragraf atau halaman
        parts = re.split(r'===== PAGE \d+ =====|\n\n', document)
        chunks = []
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            # Jika bagian lebih pendek dari chunk_size, simpan sebagai chunk
            if len(part) <= chunk_size:
                chunks.append(part)
            else:
                # Bagi menjadi chunk yang lebih kecil dengan overlap
                words = part.split()
                words_per_chunk = chunk_size // 5  # Perkiraan 5 karakter per kata
                
                for i in range(0, len(words), words_per_chunk - overlap // 5):
                    chunk = ' '.join(words[i:i + words_per_chunk])
                    if chunk:
                        chunks.append(chunk)
        
        return chunks
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Mengambil dokumen yang paling relevan dengan query tanpa filter tahun"""
        if (isinstance(self.document_embeddings, np.ndarray) and self.document_embeddings.size == 0) or len(self.document_chunks) == 0:
            return []
        
        # Encoding query
        query_embedding = self.model.encode([query])[0]
        
        # Hitung similarity
        similarities = cosine_similarity([query_embedding], self.document_embeddings)[0]
        
        # Dapatkan top-k chunks berdasarkan similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": self.document_chunks[idx],
                "metadata": self.metadata[idx],
                "similarity": similarities[idx]
            })
        
        return results

    def generate_context(self, query: str, max_year: str, top_k: int = 5) -> str:
        """Membuat konteks berdasarkan hasil retrieval tanpa filter tahun"""
        retrieved_docs = self.retrieve(query, top_k)
        
        if not retrieved_docs:
            return "Tidak ada informasi yang relevan ditemukan."
        
        # Berikan informasi tahun yang jelas dalam konteks, tetapi tidak memfilter hasil
        context = f"KONTEKS UNTUK PERTANYAAN (INGAT: GUNAKAN SITASI DENGAN TAHUN <= {max_year.strip('-')} SAJA):\n\n"
        for i, doc in enumerate(retrieved_docs, 1):
            year = doc['metadata'].get('year', 'N/A')
            # Format yang menekankan tahun dokumen
            context += f"[Dokumen {i}] - TAHUN PUBLIKASI: {year}\n"
            if year != 'N/A':
                try:
                    year_int = int(year)
                    max_year_int = int(max_year.strip('-'))
                    if year_int <= max_year_int:
                        context += f"TAHUN VALID ✓ ({year} <= {max_year.strip('-')})\n"
                    else:
                        context += f"TAHUN TIDAK VALID ✗ ({year} > {max_year.strip('-')}) - JANGAN GUNAKAN UNTUK SITASI!\n"
                except ValueError:
                    pass
            context += f"{doc['text']}\n\n"
        
        return context

# Fungsi pencari jurnal
class JurnalSearch:
        def __init__(self):
                self.url_semantic = "https://api.semanticscholar.org/graph/v1"
                self.url_openalex = "https://api.openalex.org/"
                self.headers = {
                        "Accept": "application/json"
                }
                self.rag = RAGSystem()
                try:
                        response = requests.get(self.url_semantic)
                        response1 = requests.get(self.url_openalex)
                        if response.status_code == 200 and response1.status_code == 200:
                                print("Berhasil terhubung")
                                time.sleep(3)
                        else:
                                print(f"{response}")
                                print(f"{response1}")
                             
                except requests.exceptions.RequestException as e:
                        print(f"Error connecting to APIs: {e}")
                        
                        
        # Fungsi untuk mencari paper Semantic Scholar
        def search_papers(self, query: str, tahun: int) -> List[Dict]:
                """Search for papers matching the query"""
                endpoint = f"{self.url_semantic}/paper/search"
                params = {
                        "query": query,
                        "limit": 30,
                        "year": tahun,
                        "publicationTypes": "JournalArticle",
                        "openAccessPdf":True,
                        "fields": "title,abstract,authors,url,venue,openAccessPdf,isOpenAccess,year",
                }
                try:
                        response = requests.get(endpoint, params=params, headers=self.headers)
                        print(response.url)
                        all_paper = response.json().get("data", []) 
                        print(type(all_paper))
                        return all_paper
                except requests.exceptions.RequestException as e:
                        print(f"Error: {e}")
                        return []
                        
        # Fungsi untuk mencari paper OpenAlex
        def search_work(self, query: str, tahun: int,  per_page=30) -> List[Dict]:
                url = f"{self.url_openalex}/works"
                params = {
                        "search": query, 
                        "filter":f"publication_year:{tahun}",
                        "per_page": per_page
                }
                try:
                        response = requests.get(url, params=params, headers=self.headers)
                        print(response.url)
                        time.sleep(1)
                        all_paper = response.json().get("results", [])
                        print(type(all_paper))
                        return all_paper
                except requests.exceptions.RequestException as e:
                        print(f"Error: {e}")
                        return []
                
        def process_semantic_from_url(self, url: str) -> str:
                """Process PDF from URL and extract text"""
                try:
                        response = requests.get(url, timeout=5)
                        if response.status_code == 200:
                                global index
                                index +=1
                                title = paper['title']
                                year = paper.get('year', 'N/A')
                                authors = ', '.join(author['name'] for author in paper['authors'])
                                print(f"\nTitle: {title}")
                                print(f"Year: {year}")
                                print(f"Authors: {authors}")
                                print(f"Abstract: {paper.get('abstract', 'N/A')}")
                                print(f"Downloading PDF {index}...") # Counter untuk PDF yang diunduh
                                pdf_content = response.content  #Konten PDF dalam bentuk byte
                        else:
                                raise Exception(f"Gagal mengakses URL. Status code: {response.status_code}")
                except requests.exceptions.RequestException as e:
                        print(f"Error downloading PDF: {e}")
                        return ""
                
                try:
                        # Buka PDF dari byte content
                        pdf_stream = BytesIO(pdf_content)
                        doc = fitz.open(stream=pdf_stream, filetype="pdf")
                        
                        # Download PDF
                        pdf_path = f"paper{index}.pdf"
                        with open(pdf_path, "wb") as f:
                                f.write(pdf_content)

                        # Ekstrak teks
                        text = ""
                        for page in doc:
                                text += page.get_text()
                        
                        # Bersihkan teks
                        cleaned_text = self.clean_text(text)
                        
                        # Tambahkan langsung ke RAG system
                        self.rag.add_document(cleaned_text, index-1, title, str(year))
                        
                        return cleaned_text
                except Exception as e:
                        print(f"Error processing PDF: {e}")
                        return ""
                
        def process_openalex_from_url(self, url: str) -> str:
                """Process PDF from URL and extract text"""
                try:
                        response = requests.get(url, timeout=5)
                        if response.status_code == 200:
                                global index
                                index +=1
                                title = paper['title']
                                year = paper.get('publication_year', 'N/A')
                                print(f"\nTitle: {title}")
                                print(f"Year: {year}")
                                print(f"Downloading PDF {index}...") # Counter untuk PDF yang diunduh
                                pdf_content = response.content  #Konten PDF dalam bentuk byte
                        else:
                                raise Exception(f"Gagal mengakses URL. Status code: {response.status_code}")
                except requests.exceptions.RequestException as e:
                        print(f"Error downloading PDF: {e}")
                        return ""
                
                try:
                        # Buka PDF dari byte content
                        pdf_stream = BytesIO(pdf_content)
                        doc = fitz.open(stream=pdf_stream, filetype="pdf")
                        
                        # Download PDF
                        pdf_path = f"paper{index}.pdf"
                        with open(pdf_path, "wb") as f:
                                f.write(pdf_content)
                       
                        # Ekstrak teks
                        text = ""
                        for page in doc:
                                text += page.get_text()
                        
                        # Bersihkan teks
                        cleaned_text = self.clean_text(text)
                        
                        # Tambahkan langsung ke RAG system
                        self.rag.add_document(cleaned_text, index-1, title, str(year))
                        
                        return cleaned_text
                except Exception as e:
                        print(f"Error processing PDF: {e}")
                        return ""
        
        # Fungsi untuk membersihkan teks hasil ekstraksi
        def clean_text(self, text: str) :
                try:
                        # Clean text while preserving page breaks
                        cleaned_text = ""
                        # Split text by existing page breaks
                        pages = text.split("\f")
                        
                        # Process each page
                        for i, page in enumerate(pages, 1):
                                # Basic text cleaning while preserving important punctuation
                                page = re.sub(r'\s+', ' ', page)
                                page = page.strip()
                                
                                if page:
                                        # Add clear page separator
                                        cleaned_text += f"\n\n===== PAGE {i} =====\n\n"
                                        cleaned_text += page
                        
                        print(f"Cleaned text length: {len(cleaned_text)}")
                        list_pdf.append(cleaned_text)
                        return cleaned_text
                        
                except Exception as e:
                        print(f"Error in clean_text: {e}")
                        return ""
                
# Example usage
if input_semantic:
        scholar = JurnalSearch()
        semantic_papers = scholar.search_papers(input_semantic, tahun)
        openalex_papers = scholar.search_work(input_semantic, tahun)
        papers = semantic_papers + openalex_papers
        
        # Filter papers with open access PDFs and can be processed
        if papers:
                print("Memulai pencarian paper...")
                print(f"Total papers found: {len(papers)}")
        selesai = False
        for paper in papers:
                if index >= 8 or any(paper == p for p in papers[-1]):
                        selesai = True
                        print("Memulai mendapatkan jawaban...")
                        break  # Stop processing more papers once we have 4 PDFs or reached the last paper                        # Check if the paper is in the list of papers
                        
                # Proper PDF URL handling
                if paper.get('openAccessPdf'):
                        pdf_url1 = paper['openAccessPdf'].get('url', '')
                        if pdf_url1:
                                try:
                                        print(f"\nProcessing PDF from URL: {pdf_url1}")
                                        t1 = threading.Thread(target=scholar.process_semantic_from_url, args=(pdf_url1,))
                                        t1.start()
                                        t1.join()
                                except Exception as e:
                                        print(f"Error processing PDF: {e}")
                                
                                
                if paper.get('primary_location',{}).get('pdf_url'):
                        pdf_url2 = paper.get('primary_location',{}).get('pdf_url', '')
                        if pdf_url2:
                                try:
                                        print(f"\nProcessing PDF from URL: {pdf_url2}")
                                        t2 = threading.Thread(target=scholar.process_openalex_from_url, args=(pdf_url2,))
                                        t2.start()
                                        t2.join()
                                except Exception as e:
                                        print(f"Error processing PDF: {e}")
                elif not paper.get('openAccessPdf') and not paper.get('primary_location',{}).get('pdf_url'):
                        print("No open access PDF available")



# ... existing code ...

if selesai is True:
    # Gunakan RAG yang telah diisi selama unduhan
    if hasattr(scholar, 'rag') and scholar.rag.document_chunks:
        rag_system = scholar.rag
    else:
        # Jika RAG belum diisi, gunakan list_pdf
        rag_system = RAGSystem()
        rag_system.process_documents(list_pdf)
    
    # Generate konteks untuk query
    context = rag_system.generate_context(pencarian, tahun)
    
    # Perbarui prompt untuk lebih menekankan batasan tahun
    system_prompt_answer = f"""
INSTRUKSI WAJIB - ATURAN SITASI AKADEMIK

Anda adalah seorang dosen peneliti yang sedang menulis paper akademik dengan standar tinggi.

ATURAN SITASI YANG HARUS DIPATUHI:
1. Hanya gunakan sitasi dengan tahun >= {tahun.strip('-')} (Sama dengan ATAU lebih baru dari {tahun.strip('-')})
2. Rentang tahun yang DIPERBOLEHKAN: {tahun.strip('-')} sampai 2025

============= VALIDASI SITASI =============
Untuk SETIAP sitasi yang Anda gunakan:
1. Ekstrak tahun dari sitasi (misalnya: dari "(Ahmad, 2020)" ekstrak "2020")
2. Bandingkan: tahun_sitasi >= {tahun.strip('-')}?
3. Jika YA: sitasi VALID dan BOLEH digunakan
4. Jika TIDAK: sitasi TIDAK VALID dan DILARANG digunakan!
=======================================

CONTOH untuk max_tahun {tahun.strip('-')}:
- Sitasi (Ahmad, {int(tahun.strip('-'))-1}) → TIDAK VALID! {int(tahun.strip('-'))-1} < {tahun.strip('-')}
- Sitasi (Budi, {tahun.strip('-')}) → VALID! {tahun.strip('-')} = {tahun.strip('-')}
- Sitasi (Citra, {int(tahun.strip('-'))+1}) → VALID! {int(tahun.strip('-'))+1} > {tahun.strip('-')}
- Sitasi (Dewi, {int(tahun.strip('-'))+3}) → VALID! {int(tahun.strip('-'))+3} > {tahun.strip('-')}

GAYA PENULISAN:
- Tulis seperti dosen peneliti yang sedang menulis paper ilmiah
- Gunakan bahasa akademik yang formal dan presisi
- Sajikan informasi dengan logis dan terstruktur

FORMAT OUTPUT:
1. Satu paragraf akademik (5-8 kalimat)
2. Minimal 2 sitasi dengan format (Nama, Tahun) 
3. Semua tahun sitasi HARUS >= {tahun.strip('-')}

CONTOH OUTPUT:
Rokok adalah produk olahan tembakau yang dikemas dalam bentuk silinder dari kertas dengan campuran cengkeh dan bahan tambahan lainnya yang diproduksi
oleh pabrik maupun dibuat sendiri, yang digunakan dengan cara dibakar pada salah satu ujungnya dan dibiarkan membara agar asapnya dapat dihirup melalui
mulut pada ujung lainnya. Kandungan berbahaya dalam rokok meliputi lebih dari 7.000 bahan kimia, termasuk nikotin yang bersifat adiktif, tar yang dapat
menyebabkan kanker, dan karbon monoksida yang mengganggu sistem peredaran darah (Samsuri et al., 2023). Dampak kesehatan yang ditimbulkan oleh rokok tidak hanya mempengaruhi 
perokok aktif tetapi juga perokok pasif yang terpapar asap rokok di lingkungannya. Konsumsi rokok merupakan salah satu penyebab utama kematian yang dapat
dicegah di seluruh dunia, dengan estimasi lebih dari 8 juta kematian setiap tahunnya (Kholis, 2024). Berbagai penelitian ilmiahtelah membuktikan bahwa
merokok secara signifikan meningkatkan risiko berbagai penyakit serius seperti kanker paru-paru, penyakit jantung koroner, stroke, dan penyakit
paru obstruktif kronik (PPOK)..


KONTEKS YANG RELEVAN:
{context}

PERTANYAAN: {pencarian}
"""

    # Gunakan variabel messages baru untuk jawaban
    answer_messages = [
        {"role": "system", "content": system_prompt_answer},
        {"role": "user", "content": pencarian}
    ]
    
    # Perbaiki fungsi validasi sitasi untuk memeriksa tahun >= max_tahun
    def validate_citations(response_text, max_year):
        """Memvalidasi tahun pada sitasi dalam output LLM"""
        max_year_int = int(max_year.strip('-'))
        # Regex untuk mencocokkan pola sitasi (Nama, Tahun)
        citation_pattern = r'\([A-Za-z\s&]+,\s*(\d{4})\)'
        
        citations = re.findall(citation_pattern, response_text)
        invalid_citations = []
        
        for year_str in citations:
            try:
                year = int(year_str)
                # Tahun harus >= max_year_int
                if year < max_year_int:
                    invalid_citations.append(f"({year_str}) < {max_year_int}")
            except ValueError:
                continue
        
        if invalid_citations:
            return False, f"Ditemukan sitasi dengan tahun yang tidak valid: {', '.join(invalid_citations)}"
        return True, "Semua sitasi valid"

    # Implementasi two-step approach dengan validasi
    try:
        # Langkah 1: Generate respons awal
        completion = client.chat.completions.create(
                model="google/gemini-2.0-flash-thinking-exp:free",
                temperature=0.2,  # Sedikit randomness untuk kreativitas
                messages=answer_messages
        )
        response_text = completion.choices[0].message.content
        
        # Langkah 2: Validasi sitasi
        is_valid, validation_message = validate_citations(response_text, tahun)
        
        if is_valid:
            print("Jawaban Berdasarkan Jurnal:", response_text)
        else:
            # Jika tidak valid, minta LLM memperbaiki dengan pesan yang lebih tegas
            correction_prompt = f"""
KOREKSI WAJIB - KESALAHAN DALAM SITASI!

{validation_message}

ATURAN:
- Tahun sitasi HARUS >= {tahun.strip('-')}
- Gunakan HANYA sitasi dengan tahun {tahun.strip('-')} sampai 2025
- DILARANG menggunakan sitasi dengan tahun < {tahun.strip('-')}

Tulis ulang paragraf Anda dengan memperbaiki sitasi yang tidak valid.
Jawaban tetap harus berisi 5-8 kalimat dengan minimal 2 sitasi yang valid.
"""
            correction_messages = [
                {"role": "system", "content": system_prompt_answer},
                {"role": "user", "content": pencarian},
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": correction_prompt}
            ]
            
            # Minta koreksi
            correction = client.chat.completions.create(
                    model="google/gemini-2.0-flash-thinking-exp:free",
                    temperature=0,
                    messages=correction_messages
            )
            
            print("Jawaban yang Dikoreksi:", correction.choices[0].message.content)
    except Exception as e:
        print(f"Error with LLM: {e}")
