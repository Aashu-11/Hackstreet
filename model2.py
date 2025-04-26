import streamlit as st
import requests
import numpy as np
import faiss
import random
import spacy
from collections import Counter
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="AI Journal Recommender", layout="wide")

# ————————————————————————————————————
# 1. Load spaCy for topic extraction
@st.cache_resource(show_spinner=False)
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        return spacy.load("en_core_web_sm")

# ————————————————————————————————————
# 2. Load embedding model
@st.cache_resource(show_spinner=False, ttl=24*3600)
def load_embedder():
    return SentenceTransformer("allenai-specter")

# ————————————————————————————————————
# 3. Fetch journal metadata once per session
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_openalex_journals(per_page: int = 200, max_pages: int = 5):
    journals = []
    cursor = "*"
    for _ in range(max_pages):
        params = {"per-page": per_page, "cursor": cursor}
        resp = requests.get("https://api.openalex.org/journals", params=params, timeout=10)
        if resp.status_code != 200:
            st.warning(f"OpenAlex API error: {resp.status_code}")
            break
        data = resp.json()
        journals.extend(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
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
# 5. Build FAISS index over journal embeddings (fixed progress bar)
@st.cache_resource(show_spinner=False)
def build_faiss_index(journals, _model):
    texts = [
        f"{j['display_name']} — {j.get('abbreviated_title','')}\nScope: {j.get('description','')}"
        for j in journals
    ]

    # Precompute embedding dimension
    dim = _model.encode(["test"], convert_to_numpy=True).shape[1]
    index = faiss.IndexFlatIP(dim)

    # Create a progress bar
    progress_bar = st.progress(0.0)

    batch_size = 32
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        embs = _model.encode(batch, convert_to_numpy=True)
        faiss.normalize_L2(embs)
        index.add(embs)

        # Clamp progress between 0.0 and 1.0
        progress = min(1.0, (i + batch_size) / total)
        progress_bar.progress(progress)

    return index

# ————————————————————————————————————
# 6. Key-phrase extraction
def extract_key_phrases(text, nlp, top_k=5):
    doc = nlp(text)
    noun_chunks = [
        chunk.text.lower()
        for chunk in doc.noun_chunks
        if len(chunk.text.split()) > 1 and len(chunk.text) > 5
    ]
    counts = Counter(noun_chunks).most_common(top_k)
    return [phrase for phrase, _ in counts]

# ————————————————————————————————————
# 7. Recommend journals based on query embedding + optional domain filter
def recommend_journals(query, journals, index, model, domains=None, top_k=10):
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, top_k * 3)
    recs = []

    for score, idx in zip(scores[0], ids[0]):
        j = journals[idx]
        j_domains = [c["display_name"] for c in j.get("x_concepts", []) if c["level"] == 0]
        if domains and not set(domains) & set(j_domains):
            continue

        home = j.get("homepage_url") or j.get("id")
        recs.append({
            "title": j["display_name"],
            "abbr": j.get("abbreviated_title", ""),
            "publisher": j.get("host_organization_name", "N/A"),
            "issn": j.get("issn_l","N/A"),
            "url": home,
            "domains": j_domains,
            "score": float(score)
        })
        if len(recs) >= top_k:
            break

    return recs

# ————————————————————————————————————
# 8. Dummy metrics (replace with real API if available)
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
        "impact_factor": round(impact,2),
        "acceptance_rate": f"{round(acc,1)}%",
        "indexing": random.sample(opts, count)
    }

# ————————————————————————————————————
# 9. Streamlit UI
def main():
    st.title("🎓 AI Journal Recommender")
    st.write("Paste your paper title and abstract, then hit **Suggest Journals**.")

    # load or fetch journals once
    if "journals" not in st.session_state:
        with st.spinner("Loading journal database…"):
            st.session_state.journals = fetch_openalex_journals()
    journals = st.session_state.journals

    # sidebar filters
    domains = extract_journal_domains(journals)
    st.sidebar.header("Filters")
    selected_domains = st.sidebar.multiselect("Research Domains", domains)
    impact_min, impact_max = st.sidebar.slider("Impact Factor", 0.0, 20.0, (0.0, 10.0), step=0.1)
    indexing_opts = ["Scopus", "Web of Science", "UGC CARE", "Google Scholar"]
    selected_indexing = st.sidebar.multiselect("Require Indexing In", indexing_opts)

    # inputs
    title = st.text_input("Paper Title")
    abstract = st.text_area("Paper Abstract", height=200)

    if st.button("Suggest Journals"):
        if not title.strip() or not abstract.strip():
            st.error("Both title and abstract are required.")
            return

        query = f"{title} {abstract}"
        embedder = load_embedder()
        nlp = load_spacy_model()

        # topics
        with st.spinner("Extracting key topics…"):
            topics = extract_key_phrases(query, nlp)
        st.subheader("Key Topics")
        st.write(" • ".join(topics) or "N/A")

        # build and query index
        with st.spinner("Building recommendation index…"):
            index = build_faiss_index(journals, embedder)

        recs = recommend_journals(query, journals, index, embedder, selected_domains, top_k=10)

        # apply metric filters and show top 3
        shown = 0
        st.subheader("Top Recommendations")
        for r in recs:
            m = fetch_metrics(r["issn"])
            if not (impact_min <= m["impact_factor"] <= impact_max):
                continue
            if selected_indexing and not set(selected_indexing).issubset(set(m["indexing"])):
                continue

            shown += 1
            st.markdown(f"**{shown}. {r['title']}** ({r['abbr']})")
            st.markdown(f"- Publisher: {r['publisher']}")
            st.markdown(f"- ISSN: {r['issn']}")
            st.markdown(f"- Similarity: {r['score']:.3f}")
            st.markdown(f"- Domains: {', '.join(r['domains']) or 'N/A'}")
            st.markdown(f"- Impact Factor: {m['impact_factor']} | Acceptance: {m['acceptance_rate']}")
            st.markdown(f"- Indexing: {', '.join(m['indexing'])}")
            st.markdown(f"- [Official site / OpenAlex page]({r['url']})")
            st.write("")  # spacer
            if shown >= 3:
                break

        if shown == 0:
            st.warning("No journals match your filters. Try broadening your criteria.")

if __name__ == "__main__":
    main()