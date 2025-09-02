import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from supabase import create_client, Client

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Supabase Setup ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)

# --- AI and RAG Setup ---
VECTOR_STORE_BASE_PATH = 'vector_store'
COURSE_DATA_BASE_PATH = 'course_data'
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store_cache = {}

def load_vector_store(course_name, week_num):
    cache_key = f"{course_name}_week_{week_num}"
    if cache_key in vector_store_cache:
        return vector_store_cache[cache_key]
    
    index_path = os.path.join(VECTOR_STORE_BASE_PATH, course_name, f"week_{week_num}_index")
    if os.path.exists(index_path):
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        vector_store_cache[cache_key] = db
        return db
    return None

prompt_full_module = PromptTemplate(
    template="""You are an expert educator creating a detailed, three-part learning module for the topic: "{topic}".
Your explanation must be in-depth, clear, and beginner-friendly. When relevant, include well-explained code examples in Python.
Use the following context from the course summary and the specific lecture transcript. Base your entire response ONLY on this context.
Context:
{context}
Generate the output as a single JSON object with a key "sections", which is an array of 3 objects:
1.  `{{ "type": "explanation", "content": "..." }}`: A detailed, long-form explanation of the core concepts.
2.  `{{ "type": "practical", "content": "..." }}`: A section with practical examples, case studies, or code samples.
3.  `{{ "type": "qna", "content": [ ... ] }}`: An array of 5 multiple-choice questions, each with "question_text", an array of "options", and "correct_answer".
Topic: "{topic}"
JSON Output:""",
    input_variables=["context", "topic"]
)

prompt_assessment = PromptTemplate(
    template="""You are an expert educator creating a quiz. Based on the entire context provided for "{topic}", generate a JSON array of {question_count} unique multiple-choice questions for a "{assessment_name}".
- For "Concept Check", the questions should focus on identifying and defining core concepts, not on their application.
- For "Quick Test" and "Assessment", the questions should cover a mix of concepts and their practical applications.
Context:
{context}
Generate ONLY the JSON array of questions. Each object in the array must have "question_text", an array of "options", and "correct_answer".
JSON Output:""",
    input_variables=["context", "topic", "question_count", "assessment_name"]
)

prompt_chatbot = PromptTemplate(
    template="""You are a helpful AI assistant for students. Answer the user's question concisely.
If context from lecture summaries is provided, base your answer primarily on that context.
Context: {context}
Question: {question}
Answer:""",
    input_variables=["context", "question"]
)


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.8)
llm_fixer = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)
chain_full = LLMChain(llm=llm, prompt=prompt_full_module)
chain_assessment = LLMChain(llm=llm, prompt=prompt_assessment)
chain_chatbot = LLMChain(llm=llm, prompt=prompt_chatbot)

def get_context_for_lecture(course_name, week_num, lecture_num):
    db = load_vector_store(course_name, week_num)
    if db is None: return None
    query = f"Content for Week {week_num} Lecture {lecture_num}"
    all_docs = db.similarity_search(query, k=50) 
    final_docs = []
    lecture_filename_part = f"W{week_num}_L{lecture_num}"
    for doc in all_docs:
        source_path = doc.metadata.get('source', '')
        source_filename = os.path.basename(source_path) 
        if "summary.txt" in source_filename or lecture_filename_part in source_filename:
            final_docs.append(doc)
    unique_contents = {doc.page_content for doc in final_docs}
    context_text = "\n\n---\n\n".join(unique_contents)
    if not context_text:
         context_docs = db.similarity_search(query, k=10)
         context_text = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
    return context_text

def get_context_for_week(course_name, week_num):
    db = load_vector_store(course_name, week_num)
    if db is None: return None
    # Perform a broad search to get a wide variety of docs from the week
    docs = db.similarity_search(f"all content for week {week_num}", k=50)
    return "\n\n---\n\n".join([doc.page_content for doc in docs])
    
def get_context_for_chatbot(course_name, week_num):
    db = load_vector_store(course_name, week_num)
    if db is None: return None
    query = f"Summary for Week {week_num}"
    docs = db.similarity_search(query, k=5)
    
    # Filter for summary files
    summary_docs = [doc for doc in docs if "summary.txt" in doc.metadata.get('source', '')]
    
    if not summary_docs:
        # If no summary found, get general context for the week
        return get_context_for_week(course_name, week_num)
        
    return "\n\n---\n\n".join([doc.page_content for doc in summary_docs])


def parse_and_fix_json(result_text):
    try:
        if "```json" in result_text:
            result_text = result_text.replace("```json\n", "").replace("\n```", "")
        return json.loads(result_text)
    except json.JSONDecodeError:
        correction_prompt = f"The following text is broken JSON. Please fix it and return only the valid JSON array or object.\n\n{result_text}"
        corrected_result = llm_fixer.invoke(correction_prompt).content
        if "```json" in corrected_result:
            corrected_result = corrected_result.replace("```json\n", "").replace("\n```", "")
        return json.loads(corrected_result)

# --- FLASK ROUTES ---
@app.route('/')
def selector():
    return render_template('selector.html')

@app.route('/course/<course_name>')
def dashboard(course_name):
    structure_path = os.path.join(COURSE_DATA_BASE_PATH, course_name, 'structure.json')
    try:
        with open(structure_path, 'r') as f:
            curriculum = json.load(f)
        return render_template('dashboard.html', curriculum=curriculum, course_name=course_name)
    except FileNotFoundError:
        return "Course not found", 404

# --- API ENDPOINTS ---
@app.route('/api/get_cached_lecture', methods=['POST'])
def get_cached_lecture():
    data = request.json
    cache_key = data.get('cache_key')
    try:
        response = supabase.table('cached_lectures').select('content').eq('cache_key', cache_key).execute()
        if response.data:
            return jsonify(response.data[0]['content'])
        else:
            return jsonify(None)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cache_lecture', methods=['POST'])
def cache_lecture():
    data = request.json
    cache_key = data.get('cache_key')
    content = data.get('content')
    try:
        supabase.table('cached_lectures').upsert({'cache_key': cache_key, 'content': content}).execute()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_content', methods=['POST'])
def get_content():
    data = request.json
    course_name, week_num, lecture_num = data.get('course'), data.get('week'), data.get('lecture')
    try:
        context_text = get_context_for_lecture(course_name, week_num, lecture_num)
        if context_text is None: return jsonify({"error": "Vector store not found."}), 404
        topic = f"{course_name.replace('_', ' ').title()} - Week {week_num} Lecture {lecture_num}"
        result = chain_full.run(context=context_text, topic=topic)
        content = parse_and_fix_json(result)
        return jsonify(content)
    except Exception as e:
        return jsonify({"error": f"Failed to generate content: {e}"}), 500

@app.route('/api/generate_assessment', methods=['POST'])
def generate_assessment():
    data = request.json
    course_name = data.get('course')
    week_num = data.get('week')
    assessment_type = data.get('type')

    assessment_config = {
        "concept_check": {"count": 10, "name": "Concept Check"},
        "quick_test": {"count": 15, "name": "Quick Test"},
        "assessment": {"count": 30, "name": "Assessment"}
    }
    
    config = assessment_config.get(assessment_type)
    if not config:
        return jsonify({"error": "Invalid assessment type"}), 400

    try:
        context_text = get_context_for_week(course_name, week_num)
        if context_text is None:
            return jsonify({"error": "Vector store for the week not found."}), 404

        topic = f"{course_name.replace('_', ' ').title()} - Week {week_num}"
        
        result = chain_assessment.run(
            context=context_text,
            topic=topic,
            question_count=config["count"],
            assessment_name=config["name"]
        )
        
        questions = parse_and_fix_json(result)
        return jsonify(questions)

    except Exception as e:
        return jsonify({"error": f"Failed to generate assessment: {e}"}), 500

# --- NEW CHATBOT ENDPOINT ---
@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    course_name = data.get('course')
    week_num = data.get('week')
    message = data.get('message')
    use_context = data.get('use_context')

    context_text = "No context provided."
    if use_context and week_num:
        context_text = get_context_for_chatbot(course_name, week_num) or "Could not find context for this week."

    try:
        response = chain_chatbot.run(context=context_text, question=message)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Failed to get response from AI: {e}"}), 500


if __name__ == '__main__':
    # Use port defined by environment variable, or 5000 if not defined
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)