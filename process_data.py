import os
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

# --- CONFIGURATION ---
TRANSCRIPTS_BASE_PATH = os.path.join('course_data')
VECTOR_STORE_BASE_PATH = 'vector_store'

def create_vector_stores():
    """
    Scans for course directories and creates a vector store for each week within each course.
    """
    if not os.path.exists(TRANSCRIPTS_BASE_PATH):
        print(f"Error: Base transcript directory not found at '{TRANSCRIPTS_BASE_PATH}'")
        return

    # Initialize embeddings model once
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Find all course directories
    course_dirs = [d for d in os.listdir(TRANSCRIPTS_BASE_PATH) if os.path.isdir(os.path.join(TRANSCRIPTS_BASE_PATH, d))]

    if not course_dirs:
        print("No course directories found. Please structure your data in 'course_data/<course_name>/transcripts/...'")
        return

    print(f"Found courses: {', '.join(course_dirs)}")

    for course_name in course_dirs:
        course_transcript_path = os.path.join(TRANSCRIPTS_BASE_PATH, course_name, 'transcripts')
        course_vector_store_path = os.path.join(VECTOR_STORE_BASE_PATH, course_name)

        if not os.path.exists(course_transcript_path):
            print(f"Skipping '{course_name}': No 'transcripts' folder found.")
            continue

        print(f"\n--- Processing course: {course_name} ---")

        # Clean up old vector store for this course
        if os.path.exists(course_vector_store_path):
            shutil.rmtree(course_vector_store_path)
        os.makedirs(course_vector_store_path, exist_ok=True)
        
        week_folders = [f for f in os.listdir(course_transcript_path) if os.path.isdir(os.path.join(course_transcript_path, f))]

        if not week_folders:
            print(f"No weekly transcript folders found in '{course_transcript_path}'")
            continue

        for week_folder in week_folders:
            week_num = week_folder.split('_')[-1]
            full_week_path = os.path.join(course_transcript_path, week_folder)
            print(f"  -> Processing {week_folder}...")

            loader = DirectoryLoader(
                full_week_path,
                glob="*.txt",
                show_progress=True,
                use_multithreading=True
            )
            
            try:
                documents = loader.load()
                if not documents:
                    print(f"     No .txt files found in {week_folder}. Skipping.")
                    continue

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
                docs = text_splitter.split_documents(documents)
                
                db = FAISS.from_documents(docs, embeddings)
                
                index_path = os.path.join(course_vector_store_path, f"week_{week_num}_index")
                db.save_local(index_path)
                print(f"     Successfully created index for Week {week_num} at '{index_path}'")
            
            except Exception as e:
                print(f"     An error occurred while processing {week_folder}: {e}")

if __name__ == "__main__":
    create_vector_stores()

