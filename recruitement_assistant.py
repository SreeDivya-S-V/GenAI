import boto3
import os
from PyPDF2 import PdfReader
import re
from opensearchpy import OpenSearch
import requests
from time import sleep
import tempfile
import json
from fpdf import FPDF
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth


# AWS Configuration
AWS_REGION = "us-west-2"
S3_BUCKET = "recruitment-resumes-1"
DYNAMODB_TABLE = "Resumes"
OPENSEARCH_HOST = "search-mydomain-ye3hsdpcernn5cl6ipqzrkfqe4.aos.us-west-2.on.aws"
#OPENSEARCH_USER = "admin"
#OPENSEARCH_PASSWORD = "*"
INDEX_NAME = "resumes-index"
os.environ["GOOGLE_API_KEY"] = ""
#GEMINI_API_KEY = ""

# AWS Clients
s3 = boto3.client("s3", region_name=AWS_REGION)
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(DYNAMODB_TABLE)
'''opensearch_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST}],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
    use_ssl=True,
    verify_certs=True
)'''
credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, AWS_REGION)
opensearch_client = OpenSearch(
    hosts=[{"host": OPENSEARCH_HOST, "port": 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

# Function to fetch and parse PDF from S3
def parse_pdf_from_s3(file_name):
    # Download the file from S3
    temp_dir = tempfile.gettempdir()
    local_path = os.path.join(temp_dir, file_name)
    print(f"Downloading {file_name} to {local_path}...")
    s3.download_file(S3_BUCKET, file_name, local_path)
    # Verify the file exists
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File {local_path} was not found after download.")
    
    # Extract text from PDF
    reader = PdfReader(local_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Parse name (assuming the name is in the file name or first line of the text)
    name_match = re.search(r"^(?:Name|Candidate)\s*:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    name = name_match.group(1) if name_match else file_name.replace(".pdf", "").replace("_", " ").title()
    
    # Extract skills (example based on keyword matching)
    skills_keywords = ["Python", "AWS", "Spark", "Java", "SQL"]
    skills = [skill for skill in skills_keywords if skill.lower() in text.lower()]
    
    # Extract education (example based on regex)
    education_match = re.search(r"(BSc|MSc|PhD)\s.+", text, re.IGNORECASE)
    education = education_match.group(0) if education_match else "Not Found"
    
    # Extract years of experience (example based on patterns like 'X years of experience')
    experience_matches = re.findall(r"(\d+)\s+years?", text, re.IGNORECASE)
    total_experience = sum(int(years) for years in experience_matches)
    
    os.remove(local_path)
    return {
    "Name": name.strip(),
    "Skills": skills,
    "Education": education.strip(),
    "Experience": total_experience,
    "FullText": text,
    "ResumeID": file_name.split(".")[0]
    }


# Function to process and store metadata in DynamoDB
def process_and_store_metadata():
    # List all files in the S3 bucket
    response = s3.list_objects_v2(Bucket=S3_BUCKET)
    if "Contents" not in response:
        print("No files found in S3 bucket.")
        return

    for obj in response["Contents"]:
        file_name = obj["Key"]
        if file_name.endswith(".pdf"):
            metadata = parse_pdf_from_s3(file_name)
            metadata["ResumeID"] = file_name.split(".")[0]
            metadata["S3Path"] = f"s3://{S3_BUCKET}/{file_name}"
            table.put_item(Item=metadata)
            print(f"Metadata stored for {metadata['Name']}")

# Function to generate embeddings using Gemini API via LangChain
def generate_embeddings(text):
    try:
        # Initialize the embeddings model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Generate the embedding for the input text
        embedding_vector = embeddings.embed_query(text)
        return embedding_vector
    except Exception as e:
        print(f"Failed to generate embeddings: {e}")
        raise

def create_index(index_name):
    index_body = {
        "settings": {
            "index": {
                "knn": True,  # Enable KNN for the index
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "properties": {
                "Embedding": {
                    "type": "knn_vector",
                    "dimension": 768  # Replace with the actual dimension of your embeddings
                },
                "ResumeID": {
                    "type": "keyword"
                }
            }
        }
    }

    if opensearch_client.indices.exists(index=index_name):
        opensearch_client.indices.delete(index=index_name)
        print(f"Index '{index_name}' deleted.")
    opensearch_client.indices.create(index=index_name, body=index_body)
    print(f"Index '{index_name}' created with KNN vector mapping.")

def store_embedding(resume_id, embedding):
    document = {
        "ResumeID": resume_id,
        "Embedding": embedding,
    }
    opensearch_client.index(index=INDEX_NAME, id=resume_id, body=document)
    print(f"Embedding stored for ResumeID: {resume_id}")

def retrieve_top_candidates(job_description, k=5):
    job_description_embedding = generate_embeddings(job_description)
    query = {
        "size": k,
        "query": {
            "knn": {
                "Embedding": {
                    "vector": job_description_embedding,
                    "k": k
                }
            }
        }
    }
    response = opensearch_client.search(index=INDEX_NAME, body=query)
    candidates = response['hits']['hits']
    
    print("\nTop Candidates:")
    for rank, candidate in enumerate(candidates, start=1):
        resume_id = candidate['_source']['ResumeID']
        metadata = table.get_item(Key={'ResumeID': resume_id}).get('Item', {})
        
        name = metadata.get("Name", "Unknown Candidate")
        skills = ", ".join(metadata.get("Skills", []))
        experience = metadata.get("Experience", "Unknown Experience")
        education = metadata.get("Education", "Unknown Education")
        suitability = "Strong match" if "Python" in skills or "AWS" in skills else "Moderate match"

        print(f"{rank}. {name}")
        print(f"   - Skills: {skills}")
        print(f"   - Experience: {experience} years")
        print(f"   - Education: {education}")
        print(f"   - Suitability: {suitability}")
        print()

# Function to create professional resume
def create_professional_resume(file_name, name, email, phone, summary, skills, experience, education, work_history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt=name, ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Email: {email} | Phone: {phone}", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=summary)
    pdf.ln(5)

    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="Skills", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=", ".join(skills))
    pdf.ln(5)

    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="Work Experience", ln=True)
    pdf.set_font("Arial", size=12)
    for job in work_history:
        pdf.cell(200, 10, txt=f"{job['title']} at {job['company']} ({job['years']} years)", ln=True)
        pdf.multi_cell(0, 10, txt=job['description'])
        pdf.ln(5)

    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, txt="Education", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=education)
    pdf.output(file_name)

# Function to upload resumes to S3 and store metadata
def upload_and_store_resumes(num_resumes):
    for i in range(1, num_resumes + 1):
        file_name = f"candidate_{i}_resume.pdf"
        name = f"Candidate {i}"
        email = f"candidate{i}@example.com"
        phone = f"+1234567890{i}"
        summary = "A highly motivated professional with expertise in software engineering and data analysis."
        skills = ["Python", "AWS", "Spark"]
        education = "BSc Computer Science"
        work_history = [
            {"title": "Software Engineer", "company": "TechCorp", "years": 2, "description": "Built scalable systems."},
            {"title": "Data Analyst", "company": "DataSolutions", "years": 1, "description": "Analyzed datasets."}
        ]

        create_professional_resume(file_name, name, email, phone, summary, skills, i + 2, education, work_history)
        s3.upload_file(file_name, S3_BUCKET, file_name)
        print(f"Uploaded {file_name} to s3://{S3_BUCKET}/{file_name}")

        metadata = parse_pdf_from_s3(file_name)
        metadata["ResumeID"] = file_name.split(".")[0]
        metadata["S3Path"] = f"s3://{S3_BUCKET}/{file_name}"
        table.put_item(Item=metadata)
        print(f"Metadata stored in DynamoDB for {metadata['Name']}")

# Main Workflow
if __name__ == "__main__":
    # Step 1: Extract and store metadata in DynamoDB
    #upload_and_store_resumes(10)
    #process_and_store_metadata()
    create_index(INDEX_NAME)
    # Step 2: Extract metadata, generate embeddings, and store them
    response = s3.list_objects_v2(Bucket=S3_BUCKET)
    for obj in response.get("Contents", []):
        file_name = obj["Key"]
        if file_name.endswith(".pdf"):
            metadata = parse_pdf_from_s3(file_name)
            metadata["ResumeID"] = metadata.get("ResumeID", file_name.split(".")[0])
            embedding = generate_embeddings(metadata["FullText"])
            store_embedding(metadata["ResumeID"], embedding)

    # Step 3: Retrieve top candidates based on job description
    job_description = "Looking for a data engineer skilled in Python and AWS with 2+ years of experience."
    retrieve_top_candidates(job_description, k=5)
