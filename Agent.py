import os
import re
import json
import hashlib
import pickle
import time
import threading
import warnings
import psutil
import requests
import pandas as pd
import numpy as np
import pronto
import faiss
from typing import Dict, Any, Optional, Set, List, Tuple
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from Bio import Entrez
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import xml.etree.ElementTree as ET

# --- Configuration & Environment Setup ---
class Config:
    """Central configuration for paths and API keys."""
    BASE_DIR = os.getenv("BASE_DIR", "./data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    CACHE_DIR = os.path.join(BASE_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    
    # Input/Data Paths
    INPUT_CSV = os.path.join(BASE_DIR, "input.csv")
    HPO_ONTOLOGY = os.path.join(BASE_DIR, "hp.obo")
    PROMPTS_FILE = os.path.join(BASE_DIR, "prompts.json")
    RAG_INDEX_PATH = os.path.join(BASE_DIR, "rag_index")
    
    # API Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
    ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "example@email.com")
    
    # Model Costs (adjust as needed)
    INPUT_COST = 2.5
    OUTPUT_COST = 10.0
    
    # RAG Settings
    EMBEDDING_MODEL = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    RAG_TOP_K = 5
    RAG_SCORE_THRESHOLD = 0.75
    MAX_REACT_ITERATIONS = 12

# Create directories
for d in [Config.OUTPUT_DIR, Config.CACHE_DIR, Config.CHECKPOINT_DIR]:
    os.makedirs(d, exist_ok=True)

Entrez.email = Config.ENTREZ_EMAIL

# --- Utility Functions ---
def check_memory():
    """Monitors memory usage."""
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    if mem_mb > 20000:  # Warning at 20GB
        warnings.warn(f"High memory usage: {mem_mb:.0f} MB")
    return mem_mb

def call_openai_api(session: requests.Session, prompt: str, system_prompt: str, temperature: float = 0.1) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Handles API calls with retry logic."""
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set.")

    headers = {
        "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": Config.MODEL_NAME,
        "response_format": {"type": "json_object"},
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    max_retries = 5
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = session.post(Config.OPENAI_API_URL, json=payload, headers=headers, timeout=120)
            
            if response.status_code == 429:
                time.sleep(base_delay * (2 ** attempt))
                continue
                
            response.raise_for_status()
            data = response.json()
            content = json.loads(data['choices'][0]['message']['content'])
            usage = data.get('usage', {})
            return content, usage

        except Exception as e:
            print(f"API Error (Attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                return None, None
            time.sleep(base_delay)
    return None, None

# --- RAG System ---
class MedicalRAG:
    """Manages Faiss vector search and BM25 keyword search."""
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.documents = None
        self.index_faiss = None
        self.bm25 = None
        self.model = None
        self._load_components()

    def _load_components(self):
        try:
            print(f"Loading RAG from {self.index_path}...")
            self.index_faiss = faiss.read_index(os.path.join(self.index_path, 'docs.faiss'))
            with open(os.path.join(self.index_path, 'bm25.pkl'), 'rb') as f:
                self.bm25 = pickle.load(f)
            with open(os.path.join(self.index_path, 'documents.pkl'), 'rb') as f:
                self.documents = pickle.load(f)
            self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
            print(f"RAG Loaded: {len(self.documents)} docs.")
        except Exception as e:
            print(f"RAG Load Failed: {e}")

    def search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Hybrid search implementation (Vector + BM25)."""
        if not self.documents: return []
        
        # 1. Vector Search
        embeddings = self.model.encode(queries, convert_to_tensor=False).astype('float32')
        distances, indices = self.index_faiss.search(embeddings, Config.RAG_TOP_K * 5)
        
        candidates = {}
        # Processing vector results...
        for q_idx in range(len(queries)):
            for i, doc_idx in enumerate(indices[q_idx]):
                if doc_idx == -1: continue
                if doc_idx not in candidates:
                    candidates[doc_idx] = {'vector_dist': distances[q_idx][i], 'keyword_score': 0.0}

        # 2. Keyword Search (BM25)
        for query in queries:
            scores = self.bm25.get_scores(query.lower().split())
            top_n = np.argpartition(scores, -Config.RAG_TOP_K * 5)[-Config.RAG_TOP_K * 5:]
            for idx in top_n:
                if scores[idx] > 0:
                    if idx in candidates:
                        candidates[idx]['keyword_score'] += scores[idx]
                    else:
                        candidates[idx] = {'vector_dist': 999.0, 'keyword_score': scores[idx]}

        # 3. Rerank
        results = []
        for idx, scores in candidates.items():
            doc = self.documents[idx]
            # Simple scoring logic (can be tuned)
            final_score = (1 / (1 + scores['vector_dist'])) * 0.7 + (scores['keyword_score'] * 0.3)
            results.append({'doc': doc, 'score': final_score})

        results.sort(key=lambda x: x['score'], reverse=True)
        return [r['doc'] for r in results[:Config.RAG_TOP_K]]

# --- Data Structures & State Management ---
class AgentAction(Enum):
    GET_HPO_CONTEXT = "get_hpo_context"
    GET_MESH_TERM = "get_mesh_term"
    TRANSFORM_QUERY = "transform_query"
    SEARCH_RAG = "search_rag"
    SEARCH_PUBMED = "search_pubmed"
    EXPAND_SEARCH = "expand_search"
    CLASSIFY = "classify"
    INSUFFICIENT_INFO = "insufficient_info"

@dataclass
class ReActStep:
    iteration: int
    thought: str
    action: AgentAction
    action_input: Dict
    observation: str

@dataclass
class ReActContext:
    hpo_id: str
    hpo_context: Optional[Dict] = None
    mesh_term: Optional[str] = None
    transformed_queries: List[str] = field(default_factory=list)
    rag_articles: List[Dict] = field(default_factory=list)
    pubmed_articles: List[Dict] = field(default_factory=list)
    steps: List[ReActStep] = field(default_factory=list)
    final_classification: Optional[Dict] = None
    # State flags
    mesh_searched: bool = False
    rag_searched: bool = False
    pubmed_searched: bool = False
    search_expanded: bool = False
    total_tokens_in: int = 0
    total_tokens_out: int = 0

class StateManager:
    """Handles Caching, Checkpoints, and Logging."""
    def __init__(self, lock: threading.Lock):
        self.lock = lock
        self.completed = self._load_checkpoint()
        self.cache = self._load_cache()

    def _load_checkpoint(self) -> Set[str]:
        path = os.path.join(Config.CHECKPOINT_DIR, "completed.json")
        if os.path.exists(path):
            with open(path, 'r') as f: return set(json.load(f))
        return set()

    def _load_cache(self) -> Dict:
        path = os.path.join(Config.CACHE_DIR, "cache.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f: return pickle.load(f)
        return {}

    def save_result(self, hpo_id: str, result: Dict):
        with self.lock:
            # Update CSV
            df = pd.DataFrame([result])
            out_path = os.path.join(Config.OUTPUT_DIR, "results.csv")
            header = not os.path.exists(out_path)
            df.to_csv(out_path, mode='a', header=header, index=False)
            
            # Update Checkpoint
            self.completed.add(hpo_id)
            with open(os.path.join(Config.CHECKPOINT_DIR, "completed.json"), 'w') as f:
                json.dump(list(self.completed), f)
            
            # Update Cache
            self.cache[hpo_id] = result
            with open(os.path.join(Config.CACHE_DIR, "cache.pkl"), 'wb') as f:
                pickle.dump(self.cache, f)

    def log_trace(self, context: ReActContext):
        log_entry = {
            "hpo_id": context.hpo_id,
            "steps": [
                {"iter": s.iteration, "action": s.action.value, "thought": s.thought} 
                for s in context.steps
            ]
        }
        with self.lock:
            with open(os.path.join(Config.OUTPUT_DIR, "agent_logs.json"), 'a') as f:
                f.write(json.dumps(log_entry) + "\n")

# --- External Tool Functions ---
def get_hpo_details(hpo_id: str, ontology: pronto.Ontology) -> Dict:
    try:
        term = ontology[hpo_id]
        return {
            "id": hpo_id,
            "name": term.name,
            "definition": str(term.definition),
            "synonyms": [s.description for s in term.synonyms if s.scope == 'EXACT']
        }
    except KeyError:
        return {}

def search_pubmed(query: str, max_results: int = 10) -> List[Dict]:
    """Basic PubMed search wrapper."""
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        ids = record.get("IdList", [])
        if not ids: return []
        
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
        data = handle.read()
        root = ET.fromstring(data)
        
        results = []
        for article in root.findall("PubmedArticle"):
            try:
                title = article.find(".//ArticleTitle").text
                abstract = "".join([x.text for x in article.findall(".//AbstractText") if x.text])
                pmid = article.find(".//PMID").text
                if abstract:
                    results.append({
                        "source": "PubMed", "title": title, 
                        "content": abstract, "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    })
            except: continue
        return results
    except Exception as e:
        print(f"PubMed Error: {e}")
        return []

# --- ReAct Agent Core ---

def execute_action(action: AgentAction, context: ReActContext, tools: Dict) -> str:
    """Executes the specific tool/action requested by the agent."""
    ontology = tools['ontology']
    rag = tools['rag']
    session = tools['session']
    prompts = tools['prompts']

    if action == AgentAction.GET_HPO_CONTEXT:
        context.hpo_context = get_hpo_details(context.hpo_id, ontology)
        return f"Retrieved: {context.hpo_context.get('name', 'Unknown')}"

    elif action == AgentAction.TRANSFORM_QUERY:
        # Load prompt from dictionary/json instead of hardcoding
        prompt = prompts.get("transform_query").format(term=context.hpo_context['name'])
        response, usage = call_openai_api(session, prompt, "Generate search queries.")
        if response:
            context.transformed_queries = response.get("queries", [])
            context.total_tokens_in += usage['prompt_tokens']
            return f"Generated {len(context.transformed_queries)} queries."
        return "Failed to generate queries."

    elif action == AgentAction.SEARCH_RAG:
        context.rag_searched = True
        results = rag.search(context.transformed_queries)
        context.rag_articles = results
        return f"Found {len(results)} RAG documents."

    elif action == AgentAction.SEARCH_PUBMED:
        context.pubmed_searched = True
        term = context.hpo_context.get('name')
        results = search_pubmed(f"{term} AND (genetics OR congenital)")
        context.pubmed_articles = results
        return f"Found {len(results)} PubMed articles."
    
    elif action == AgentAction.EXPAND_SEARCH:
        context.search_expanded = True
        return "Search expanded strategy enabled."

    elif action == AgentAction.CLASSIFY:
        # Load large prompt from file
        final_prompt = prompts.get("classification_prompt").format(
            hpo_data=json.dumps(context.hpo_context),
            rag_data=json.dumps(context.rag_articles[:3]), # Limit context
            pubmed_data=json.dumps(context.pubmed_articles[:3])
        )
        response, usage = call_openai_api(session, final_prompt, "Classify HPO term.", temperature=0.0)
        if response:
            context.final_classification = response
            context.total_tokens_in += usage['prompt_tokens']
            return "Classification complete."
        return "Classification failed."

    return "Action not recognized."

def run_agent_loop(hpo_id: str, tools: Dict, state_mgr: StateManager):
    """Main ReAct Loop."""
    if hpo_id in state_mgr.completed:
        return

    context = ReActContext(hpo_id=hpo_id)
    prompts = tools['prompts']
    session = tools['session']
    
    print(f"Starting agent for {hpo_id}")
    
    for i in range(Config.MAX_REACT_ITERATIONS):
        # 1. Plan
        state_desc = f"Steps taken: {len(context.steps)}"
        plan_prompt = prompts.get("planning_prompt").format(state=state_desc)
        
        plan_json, usage = call_openai_api(session, plan_prompt, "You are a ReAct agent.")
        if not plan_json: break
        
        context.total_tokens_in += usage['prompt_tokens']
        
        # 2. Parse
        try:
            action_enum = AgentAction(plan_json.get("action"))
            thought = plan_json.get("thought")
        except ValueError:
            print(f"Invalid action: {plan_json.get('action')}")
            continue

        # 3. Execute
        observation = execute_action(action_enum, context, tools)
        
        # 4. Record
        step = ReActStep(i, thought, action_enum, plan_json.get("action_input", {}), observation)
        context.steps.append(step)
        
        if action_enum == AgentAction.CLASSIFY:
            break

    # Save Results
    if context.final_classification:
        result_flat = {
            "hpo_id": hpo_id,
            "classification": json.dumps(context.final_classification),
            "cost": (context.total_tokens_in/1e6 * Config.INPUT_COST)
        }
        state_mgr.save_result(hpo_id, result_flat)
        state_mgr.log_trace(context)

# --- Main Entry Point ---
def main():
    # 1. Load Resources
    try:
        with open(Config.PROMPTS_FILE, 'r') as f:
            prompts = json.load(f)
        ontology = pronto.Ontology(Config.HPO_ONTOLOGY)
        rag_system = MedicalRAG(Config.RAG_INDEX_PATH)
        input_df = pd.read_csv(Config.INPUT_CSV)
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    # 2. Setup Threading
    lock = threading.Lock()
    state_mgr = StateManager(lock)
    hpo_ids = input_df['hpo_id'].unique().tolist()
    
    # 3. Execute
    with requests.Session() as session:
        tools = {
            "ontology": ontology,
            "rag": rag_system,
            "session": session,
            "prompts": prompts
        }
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(run_agent_loop, hid, tools, state_mgr) 
                for hid in hpo_ids
            ]
            for _ in tqdm(futures, desc="Processing HPO Terms"):
                pass

if __name__ == "__main__":
    main()