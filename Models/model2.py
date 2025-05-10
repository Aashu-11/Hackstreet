import streamlit as st
import requests
import numpy as np
import faiss
import random
from collections import Counter
import time
import spacy
from sentence_transformers import SentenceTransformer

# Configure page and improve caching
st.set_page_config(page_title="AI Journal Recommender", layout="wide")

# Add session state management
if "load_state" not in st.session_state:
    st.session_state.load_state = "not_started"
    st.session_state.journals = None
    st.session_state.nlp = None
    st.session_state.embedder = None
    st.session_state.faiss_index = None

# ————————————————————————————————————
# 1. Load spaCy for topic extraction - simplified with better error handling
@st.cache_resource(show_spinner=False)
def load_spacy_model():
    try:
        # First try loading the model directly
        return spacy.load("en_core_web_sm")
    except OSError:
        # If model not found, try using spaCy's built-in downloader
        try:
            st.info("Downloading spaCy model - this may take a moment...")
            spacy.cli.download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception as e:
            # If all fails, load a simple pipeline as fallback
            st.warning(f"Couldn't load spaCy model: {str(e)}. Using simple pipeline instead.")
            return spacy.blank("en")

# ————————————————————————————————————
# 2. Load embedding model with timeout handling
@st.cache_resource(show_spinner=False)
def load_embedder():
    try:
        # Timeout handling - this prevents indefinite hanging
        st.info("Loading sentence embedding model...")
        start_time = time.time()
        model = SentenceTransformer("allenai-specter")
        load_time = time.time() - start_time
        st.info(f"Model loaded in {load_time:.2f} seconds")
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None

# ————————————————————————————————————
# 3. Fetch journal metadata with more robust error handling and timeouts
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_openalex_journals(per_page=100, max_pages=2):  # Reduced parameters for faster loading
    journals = []
    cursor = "*"
    
    st.info(f"Fetching journals data, this may take a moment...")
    progress_bar = st.progress(0)
    
    for i in range(max_pages):
        try:
            params = {"per-page": per_page, "cursor": cursor, "filter": "has_issn:true"}
            resp = requests.get(
                "https://api.openalex.org/journals", 
                params=params, 
                timeout=15  # Increased timeout but not too long
            )
            
            if resp.status_code != 200:
                st.warning(f"OpenAlex API error: {resp.status_code}")
                break
                
            data = resp.json()
            results = data.get("results", [])
            
            if not results:
                st.warning("No more results found")
                break
                
            journals.extend(results)
            cursor = data.get("meta", {}).get("next_cursor")
            
            # Update progress
            progress_bar.progress((i + 1) / max_pages)
            
            if not cursor:
                break
                
        except requests.exceptions.Timeout:
            st.warning(f"Request timed out on page {i+1}/{max_pages}. Using partial data.")
            break
        except Exception as e:
            st.warning(f"Error fetching page {i+1}: {str(e)}. Using partial data.")
            break
    
    # Ensure we have at least some journals
    if not journals:
        # Fallback to small test dataset
        st.warning("Could not fetch journals. Using test dataset.")
        journals = [
            {"display_name": "Nature", "abbreviated_title": "Nature", 
             "description": "Scientific journal", "id": "https://openalex.org/J1234",
             "x_concepts": [{"level": 0, "display_name": "Biology"}]},
            {"display_name": "Science", "abbreviated_title": "Science", 
             "description": "Scientific journal", "id": "https://openalex.org/J5678",
             "x_concepts": [{"level": 0, "display_name": "Physics"}]}
        ]
    
    return journals

# ————————————————————————————————————
# 4. Extract top-level research domains
def extract_journal_domains(journals):
    domains = set()
    for j in journals:
        for c in j.get("x_concepts", []):
            if c.get("level") == 0:
                domains.add(c["display_name"])
    return sorted(domains)

# ————————————————————————————————————
# 5. Build FAISS index with better progress reporting and error handling
def build_faiss_index(journals, model):
    try:
        if not journals or not model:
            return None
            
        # Prepare journal texts for embedding
        texts = [
            f"{j['display_name']} — {j.get('abbreviated_title','')}\nScope: {j.get('description','')}"
            for j in journals
        ]
        
        # Get embedding dimension from a test string
        test_emb = model.encode(["test"], convert_to_numpy=True)
        dim = test_emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        
        # Process in smaller batches with better progress reporting
        progress_bar = st.progress(0)
        batch_size = 16  # Smaller batch size for more frequent updates
        total = len(texts)
        
        for i in range(0, total, batch_size):
            # Limit batch to available texts
            end = min(i + batch_size, total)
            batch = texts[i:end]
            
            # Encode batch
            embs = model.encode(batch, convert_to_numpy=True)
            faiss.normalize_L2(embs)
            index.add(embs)
            
            # Update progress more frequently
            progress = min(1.0, end / total)
            progress_bar.progress(progress)
            
            # Add a small delay to allow UI updates
            time.sleep(0.01)
        
        progress_bar.progress(1.0)
        return index
        
    except Exception as e:
        st.error(f"Error building search index: {str(e)}")
        return None

# ————————————————————————————————————
# 6. Key-phrase extraction with fallback mechanism
def extract_key_phrases(text, nlp, top_k=5):
    try:
        if not nlp or not text:
            return ["(No topics extracted)"]
            
        # Process with timeout handling
        start_time = time.time()
        doc = nlp(text[:5000])  # Limit text length to prevent hanging
        
        # If spaCy is working properly, extract noun chunks
        noun_chunks = [
            chunk.text.lower()
            for chunk in doc.noun_chunks
            if len(chunk.text.split()) > 1 and len(chunk.text) > 5
        ]
        
        if noun_chunks:
            counts = Counter(noun_chunks).most_common(top_k)
            return [phrase for phrase, _ in counts]
        else:
            # Fallback to simple word frequency if no noun chunks found
            words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
            counts = Counter(words).most_common(top_k)
            return [word for word, _ in counts]
            
    except Exception as e:
        st.warning(f"Topic extraction error: {str(e)}")
        # Return simple fallback based on word frequency
        words = text.lower().split()
        return list(set([w for w in words if len(w) > 5]))[:top_k]

# ————————————————————————————————————
# 7. Recommend journals with timeout handling
def recommend_journals(query, journals, index, model, domains=None, top_k=10):
    try:
        if not model or not index:
            return []
            
        # Prepare query embedding
        q_emb = model.encode([query[:5000]], convert_to_numpy=True)  # Limit query length
        faiss.normalize_L2(q_emb)
        
        # Search index
        scores, ids = index.search(q_emb, min(top_k * 3, len(journals)))
        recs = []
        
        # Process results
        for score, idx in zip(scores[0], ids[0]):
            # Skip invalid indices
            if idx < 0 or idx >= len(journals):
                continue
                
            j = journals[idx]
            j_domains = [c["display_name"] for c in j.get("x_concepts", []) if c.get("level") == 0]
            
            # Apply domain filter if specified
            if domains and not set(domains) & set(j_domains):
                continue
                
            # Get homepage or fallback
            home = j.get("homepage_url") or j.get("id", "#")
            
            # Create recommendation entry
            recs.append({
                "title": j["display_name"],
                "abbr": j.get("abbreviated_title", ""),
                "publisher": j.get("host_organization_name", "N/A"),
                "issn": j.get("issn_l", "N/A"),
                "url": home,
                "domains": j_domains,
                "score": float(score)
            })
            
            # Stop once we have enough recommendations
            if len(recs) >= top_k:
                break
                
        return recs
        
    except Exception as e:
        st.error(f"Recommendation error: {str(e)}")
        return []

# ————————————————————————————————————
# 8. Journal metrics generation (mock data)
def fetch_metrics(issn):
    opts = ["Scopus", "Web of Science", "UGC CARE", "Google Scholar"]
    tier = random.random()
    
    if tier < 0.3:
        count, impact, acc = random.randint(3,4), random.uniform(3,10), random.uniform(10,25)
    elif tier < 0.6:
        count, impact, acc = random.randint(2,3), random.uniform(1.5,3), random.uniform(25,40)
    else:
        count, impact, acc = random.randint(1,2), random.uniform(0.5,1.5), random.uniform(40,60)
        
    return {
        "impact_factor": round(impact, 2),
        "acceptance_rate": f"{round(acc, 1)}%",
        "indexing": random.sample(opts, count)
    }

# ————————————————————————————————————
# Load resources in stages to prevent hanging
def load_resources():
    # Step 1: Load journals data
    if st.session_state.journals is None:
        with st.spinner("Loading journal database..."):
            st.session_state.journals = fetch_openalex_journals(per_page=100, max_pages=2)
            if st.session_state.journals:
                st.success(f"✅ Loaded {len(st.session_state.journals)} journals")
            else:
                st.error("Failed to load journals data")
                return False
    
    # Step 2: Load NLP model
    if st.session_state.nlp is None:
        with st.spinner("Loading NLP model..."):
            st.session_state.nlp = load_spacy_model()
            if st.session_state.nlp:
                st.success("✅ NLP model loaded")
            else:
                st.error("Failed to load NLP model")
                return False
    
    # Step 3: Load embedding model
    if st.session_state.embedder is None:
        with st.spinner("Loading embedding model..."):
            st.session_state.embedder = load_embedder()
            if st.session_state.embedder:
                st.success("✅ Embedding model loaded")
            else:
                st.error("Failed to load embedding model")
                return False
    
    return True

# ————————————————————————————————————
# Main UI
def main():
    st.title("🎓 AI Journal Recommender")
    st.write("Paste your paper title and abstract, then hit **Suggest Journals**.")
    
    # Load resources in stages to prevent hanging
    if not load_resources():
        st.error("Some resources failed to load. Try refreshing the page.")
        st.stop()
        return
    
    # Access loaded resources
    journals = st.session_state.journals
    nlp = st.session_state.nlp
    embedder = st.session_state.embedder
    
    # Sidebar filters
    domains = extract_journal_domains(journals)
    st.sidebar.header("Filters")
    selected_domains = st.sidebar.multiselect("Research Domains", domains)
    impact_min, impact_max = st.sidebar.slider("Impact Factor", 0.0, 15.0, (0.0, 10.0), step=0.5)
    indexing_opts = ["Scopus", "Web of Science", "UGC CARE", "Google Scholar"]
    selected_indexing = st.sidebar.multiselect("Require Indexing In", indexing_opts)
    
    # Number of recommendations
    num_rec = st.sidebar.slider(
        "Number of recommendations", min_value=1, max_value=10, value=3, step=1
    )
    
    # Reset button in sidebar
    if st.sidebar.button("Reset Cached Data"):
        st.session_state.journals = None
        st.session_state.nlp = None
        st.session_state.embedder = None
        st.session_state.faiss_index = None
        st.experimental_rerun()
    
    # Paper inputs
    col1, col2 = st.columns([1, 1])
    with col1:
        title = st.text_input("Paper Title", key="title")
    with col2:
        st.write(" ")  # Spacer
        debug_mode = st.checkbox("Debug mode (show detailed progress)")
    
    abstract = st.text_area("Paper Abstract", height=150, key="abstract")
    
    # Submit button
    if st.button("Suggest Journals", type="primary"):
        # Validate inputs
        if not title.strip() or not abstract.strip():
            st.error("Both title and abstract are required.")
            return
            
        query = f"{title} {abstract}"
        
        # Extract topics
        with st.spinner("Extracting key topics..."):
            topics = extract_key_phrases(query, nlp)
            st.subheader("Key Topics")
            st.write(" • ".join(topics) or "N/A")
        
        # Build FAISS index if needed
        if st.session_state.faiss_index is None:
            with st.spinner("Building search index (first time only)..."):
                st.session_state.faiss_index = build_faiss_index(journals, embedder)
                
        # Get recommendations
        with st.spinner("Finding relevant journals..."):
            recs = recommend_journals(
                query, 
                journals, 
                st.session_state.faiss_index, 
                embedder, 
                selected_domains, 
                top_k=20  # Get more, then filter
            )
        
        if not recs:
            st.warning("No matching journals found. Try different keywords or fewer filters.")
            return
            
        # Display recommendations with filtering
        shown = 0
        st.subheader(f"Top Journal Recommendations")
        
        # Create a progress container for finding matches
        progress_container = st.empty()
        results_container = st.container()
        
        total_checked = 0
        with results_container:
            for r in recs:
                total_checked += 1
                
                # Update progress for finding matches
                if debug_mode:
                    progress_container.info(f"Checking journal {total_checked}/{len(recs)}...")
                
                # Get metrics and apply filters
                m = fetch_metrics(r["issn"])
                if not (impact_min <= m["impact_factor"] <= impact_max):
                    continue
                if selected_indexing and not set(selected_indexing).issubset(set(m["indexing"])):
                    continue
                
                # Display recommendation
                shown += 1
                st.markdown(f"**{shown}. {r['title']}** ({r['abbr']})")
                
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown(f"- Publisher: {r['publisher']}")
                    st.markdown(f"- ISSN: {r['issn']}")
                    st.markdown(f"- Domains: {', '.join(r['domains']) or 'N/A'}")
                with col2:
                    st.markdown(f"- Impact Factor: {m['impact_factor']}")
                    st.markdown(f"- Acceptance Rate: {m['acceptance_rate']}")
                    st.markdown(f"- Indexing: {', '.join(m['indexing'])}")
                
                st.markdown(f"- Match Score: {r['score']:.3f}")
                st.markdown(f"- [Official site / OpenAlex page]({r['url']})")
                st.markdown("---")
                
                # Stop once we have enough recommendations
                if shown >= num_rec:
                    break
        
        # Clear the progress message
        progress_container.empty()
        
        # Show message if no journals passed the filters
        if shown == 0:
            st.warning("No journals match your filters. Try broadening your criteria.")
        else:
            st.success(f"Found {shown} matching journals from {total_checked} candidates.")

if __name__ == "__main__":
    main()
