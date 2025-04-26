# app.py
import streamlit as st
import requests
import numpy as np
import faiss
import random
from sentence_transformers import SentenceTransformer

# 1. Fetch journal metadata from OpenAlex
@st.cache_data(show_spinner=False)
def fetch_openalex_journals(per_page: int = 200, max_pages: int = 5):
    journals = []
    cursor = "*"
    for _ in range(max_pages):
        params = {"per-page": per_page, "cursor": cursor}
        resp = requests.get("https://api.openalex.org/journals", params=params)
        if resp.status_code != 200:
            st.warning(f"OpenAlex API returned status {resp.status_code}")
            break
        data = resp.json()
        journals.extend(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
    return journals

# 2. Load embedding model
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("allenai-specter")

# 3. Build FAISS index over journal embeddings
@st.cache_resource(show_spinner=False)
def build_faiss_index(journals, _model):
    texts = [f"{j['display_name']}. Scope: {j.get('description', '')}" for j in journals]
    embs = _model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index

# 4. Recommendation function
def recommend_journals(query: str, journals, index, model, top_k: int = 3):
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, top_k)
    recs = []
    for score, idx in zip(scores[0], ids[0]):
        journal = journals[idx]
        recs.append({
            "title": journal["display_name"],
            "score": float(score),
            "issn": journal.get("issn_l", "N/A"),
            "url": journal.get("homepage_url", ""),
        })
    return recs

# 5. Stub: generate random metrics and indexing
@st.cache_data
def fetch_metrics(issn: str):
    return {
        "impact_factor": round(random.uniform(0.5, 10.0), 2),
        "acceptance_rate": f"{round(random.uniform(10, 30), 1)}%",
        # "indexing": ["Scopus", "Web of Science", "UGC CARE", "Google Scholar"]
    }

# 6. UI

def main():
    st.title("🎓 AI Journal Recommender")
    st.write("Get your top 3 journal matches plus impact factor and indexing filters.")

    # Sidebar controls
    st.sidebar.markdown("## Settings & Filters")
    per_page = st.sidebar.number_input(
        "Journals per page (OpenAlex)", 50, 400, 200, 50, key="per_page"
    )
    max_pages = st.sidebar.number_input(
        "Pages to fetch", 1, 10, 5, 1, key="max_pages"
    )
    impact_min, impact_max = st.sidebar.slider(
        "Impact Factor Range", 0.0, 20.0, (0.0, 10.0), step=0.1, key="if_range"
    )
    # indexing_opts = ["Scopus", "Web of Science", "UGC CARE", "Google Scholar"]
    # selected_indexing = st.sidebar.multiselect(
    #     "Require Indexing In", indexing_opts, default=indexing_opts, key="index_filter"
    # )
    st.sidebar.caption("Filter by impact factor coverage.")

    # Main inputs
    title = st.text_input("Paper Title", key="paper_title")
    abstract = st.text_area("Paper Abstract", height=200, key="paper_abstract")

    if st.button("Suggest Journals", key="suggest_btn"):
        if not title.strip() or not abstract.strip():
            st.error("Enter both title and abstract.")
            return

        with st.spinner("Fetching & indexing…"):
            journals = fetch_openalex_journals(per_page, max_pages)
            model = load_embedder()
            index = build_faiss_index(journals, model)

        query = f"{title.strip()} {abstract.strip()}"
        recs = recommend_journals(query, journals, index, model, top_k=3)

        st.subheader("Top 3 Recommendations")
        for i, r in enumerate(recs, 1):
            metrics = fetch_metrics(r["issn"])
            # apply filters
            if not (impact_min <= metrics["impact_factor"] <= impact_max):
                continue
            # if not set(selected_indexing).issubset(set(metrics["indexing"])):
            #     continue

            st.markdown(
                f"**{i}. {r['title']}**\n"
                f"- ISSN: {r['issn']}\n"
                f"- Similarity: {r['score']:.3f}\n"
                f"- Impact Factor: {metrics['impact_factor']}\n"
                f"- Acceptance Rate: {metrics['acceptance_rate']}\n"
                # f"- Indexing: {', '.join(metrics['indexing'])}\n"
                f"- [Homepage]({r['url']})"
            )

if __name__ == "__main__":
    main()
