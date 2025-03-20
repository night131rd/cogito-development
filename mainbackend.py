import requests
from typing import Dict, List, Optional
import fitz  # PyMuPDF
from io import BytesIO
import re
from openai import OpenAI
import json
import time
import threading
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
    api_key="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",
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
                model="google/gemini-2.0-flash-exp:free",
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

# Fungsi pencari jurnal
class JurnalSearch:
        def __init__(self):
                self.url_semantic = "https://api.semanticscholar.org/graph/v1"
                self.url_openalex = "https://api.openalex.org/"
                self.headers = {
                        "Accept": "application/json"
                }
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
                
        # Fungsi untuk mengunduh PDF dari URL dan mengekstrak teks
        def process_semantic_from_url(self, url: str) -> str:
                """Process PDF from URL and extract text"""
                try:
                        response = requests.get(url, timeout=5)
                        if response.status_code == 200:
                                global index
                                index +=1
                                print(f"\nTitle: {paper['title']}")
                                print(f"Year: {paper.get('year', 'N/A')}")
                                print(f"Authors: {', '.join(author['name'] for author in paper['authors'])}")
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
                                f.write(pdf_stream.read())

                        # Ekstrak teks
                        text = ""
                        for page in doc:
                                text += page.get_text()
                                
                        return self.clean_text(text)
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
                                        print(f"\nTitle: {paper['title']}")
                                        print(f"Year: {paper.get('publication_year', 'N/A')}")
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
                                f.write(pdf_stream.read())
                       

                        # Ekstrak teks
                        text = ""
                        for page in doc:
                                text += page.get_text()
                                
                        return self.clean_text(text)
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
                if index >= 4 or any(paper == p for p in papers[-1]):
                        selesai = True
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

# Berikan jawaban pengguna berdasarkan list_pdf menggunakan OpenAI LLM
if selesai == True:
        # Prompt jawaban
        # Path file pdf
        prompt_answer = f"""
Anda adalah asisten penulisan akademik yang pandai melakukan sitasi pada artikel ilmiah 
Contoh cara melakukan sitasi:
Misalnya: maksimal tahun 2023
Text yang diambil dari paper: "Benih bermutu tinggi memiliki ukuran yang seragam, memastikan keseragaman pertumbuhan (Kartika, 20023)"
Cara tersebut benar karena 2023 lebih muda dari maksimal tahun yaitu 2023.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

Text yang diambil dari paper: "Pisaunya tajam dan mudah digunakan (Kartika et al., 2022)"
Mengambil paragraft tersebut tidak benar karena tahun 2022 lebih tua dari maksimal tahun yaitu 2023.

Pastikan output  menjadi paragraf dengan format kutipan sebagai berikut:

Jika hanya 1 penulis:
(Nama, Tahun)

Jika 2 penulis:
(Nama1 & Nama2, Tahun)

Jika lebih darti 2 penulis:
(Nama1 et al., Tahu)
Aturan:
-Penulis hanya ditulis nama akhir  
contoh : Muhaammad Askur Kholis 
Nama : Kholis
-Tahun mengacu pada tahun publikasi jurnal, Pastikan jurnal yang diambil tahunnya tidak lebih dari maksimal tahun {tahun}, apabila tahun jurnal yang di sitasi melebihi {tahun}, cari referensi lain 
-Halaman mengacu pada statement yang diambil dari paper berada di halaman berapa
- Output berisi 1 paragraf yang berisi 6 - 8 kalimat dengan maksimal 4 kutipan
Format kutipan ditempatkan tepatt setelah pernyataan dalam tanda kurung

Contoh Output:
Rokok adalah produk olahan tembakau yang dikemas dalam bentuk silinder dari kertas dengan campuran cengkeh dan bahan tambahan lainnya yang diproduksi
oleh pabrik maupun dibuat sendiri, yang digunakan dengan cara dibakar pada salah satu ujungnya dan dibiarkan membara agar asapnya dapat dihirup melalui
mulut pada ujung lainnya. Kandungan berbahaya dalam rokok meliputi lebih dari 7.000 bahan kimia, termasuk nikotin yang bersifat adiktif, tar yang dapat
menyebabkan kanker, dan karbon monoksida yang mengganggu sistem peredaran darah (Samsuri et al., 2023). Dampak kesehatan yang ditimbulkan oleh rokok tidak hanya mempengaruhi 
perokok aktif tetapi juga perokok pasif yang terpapar asap rokok di lingkungannya. Konsumsi rokok merupakan salah satu penyebab utama kematian yang dapat
dicegah di seluruh dunia, dengan estimasi lebih dari 8 juta kematian setiap tahunnya (Kholis, 2024). Berbagai penelitian ilmiahtelah membuktikan bahwa
merokok secara signifikan meningkatkan risiko berbagai penyakit serius seperti kanker paru-paru, penyakit jantung koroner, stroke, dan penyakit
paru obstruktif kronik (PPOK)..

Instruksi:
Tuliskan jaaawaban penjelasan mendetail tentang  {pencarian}, berdasarkan pengetahuan anda dan dokumen {list_pdf}  seperti contoh di atas, apabila iformasi yang diambil dari paper, pastikan untuk melakukan sitasi. Jawaban harus berisi 6-8 kalimat dan minimal maksimal 2 sitasi.
Pastikan jawaban yang anda berikan relevan dengan menggunakan alur pemikiran terstruktur (chain of thought) dan  sitasi pada jurnal dengan tahun rilis lebih muda atau pada tahun {tahun}.
"""
        
        # Mulai percakapan dengan OpenAI LLM
        messages.append({"role": "user", "content": prompt_answer})
        try:
                # Get LLM completion
                completion = client.chat.completions.create(
                        model="google/gemini-2.0-flash-exp:free",
                        temperature= 0.8,
                        messages= [{"role": "user", "content": prompt_answer}]
                )
                # Print jawaban dari OpenAI LLM
                print("Jawaban Berdasarkan Jurnal:", completion.choices[0].message.content)
        except Exception as e:
                print(f"Error with llM: {e}")
        
        try:
                with open("textpdf.txt", "w", encoding='utf-8') as f:
                        for idx, pdf_text in enumerate(list_pdf, 1):
                                f.write(pdf_text)
                                f.write("\n\n")
        except Exception as e:
                print(f"Error writing to file: {e}")


