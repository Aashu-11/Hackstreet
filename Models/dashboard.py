import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# Set page configuration
st.set_page_config(
    page_title="Scholarly Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f2f6;
    }
    .stat-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .stat-label {
        font-size: 1rem;
        color: #6B7280;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to fetch data from OpenAlex API
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_openalex_data(endpoint, params=None):
    """
    Fetch data from OpenAlex API
    
    Parameters:
    endpoint (str): API endpoint (works, authors, venues, institutions, concepts)
    params (dict): Query parameters
    
    Returns:
    dict: JSON response
    """
    base_url = "https://api.openalex.org"
    url = f"{base_url}/{endpoint}"
    
    # Add your email for polite usage
    if params is None:
        params = {}
    params['mailto'] = 'example@domain.com'  # Replace with your email
    
    headers = {
        'User-Agent': 'OpenAlex_Scholarly_Dashboard/1.0',
        'Accept': 'application/json'
    }
    
    try:
        with st.spinner(f"Fetching data from OpenAlex {endpoint} endpoint..."):
            # Add delay between API calls to avoid rate limiting
            time.sleep(0.5)
            response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text[:100]}")
            return None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to process data for the dashboard
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def process_data(topic, start_year, end_year, limit=200):
    """Process data for dashboard visualizations"""
    
    # Fixed filter to match OpenAlex API requirements
    params = {
        'filter': f'title.search:{topic},publication_year:{start_year}-{end_year}',
        'per_page': min(limit, 100),  # OpenAlex limits to 100 per page
        'sort': 'publication_date:desc'
    }
    
    works_data = fetch_openalex_data('works', params)
    
    # If API call fails or no results, try different filter
    if not works_data or 'results' not in works_data or len(works_data['results']) == 0:
        # Try with abstract search instead
        params['filter'] = f'abstract.search:{topic},publication_year:{start_year}-{end_year}'
        works_data = fetch_openalex_data('works', params)
    
    # If still failing, return sample data
    if not works_data or 'results' not in works_data or len(works_data['results']) == 0:
        st.warning(f"No data available for '{topic}' between {start_year}-{end_year}. Using sample data for demonstration.")
        return generate_sample_data(start_year, end_year)
    
    # Process the actual data
    works_df = pd.json_normalize(works_data['results'])
    
    # Extract basic information
    if 'publication_date' in works_df.columns:
        works_df['publication_year'] = pd.to_datetime(works_df['publication_date']).dt.year
    else:
        works_df['publication_year'] = works_df.get('publication_year', start_year)
    
    # Publication trends data
    pub_trends = works_df['publication_year'].value_counts().sort_index()
    pub_trends_df = pd.DataFrame({'year': pub_trends.index, 'publications': pub_trends.values})
    
    # Citation metrics
    citation_data = works_df[['publication_year', 'cited_by_count']].copy() if 'cited_by_count' in works_df.columns else pd.DataFrame({
        'publication_year': works_df['publication_year'],
        'cited_by_count': 0
    })
    
    yearly_citations = citation_data.groupby('publication_year')['cited_by_count'].agg(['mean', 'sum', 'count'])
    yearly_citations.reset_index(inplace=True)
    yearly_citations.columns = ['year', 'avg_citations', 'total_citations', 'paper_count']
    
    # Open access trends - Generate both actual and fallback data
    try:
        if 'open_access.is_oa' in works_df.columns:
            works_df['is_oa'] = works_df['open_access.is_oa'].fillna(False)
            oa_by_year = works_df.groupby('publication_year')['is_oa'].agg(['sum', 'count'])
            oa_by_year.columns = ['open_access', 'total']
            oa_by_year['percentage'] = (oa_by_year['open_access'] / oa_by_year['total'] * 100).round(1)
            oa_df = oa_by_year.reset_index()
            # Rename index to year for consistency
            oa_df = oa_df.rename(columns={'publication_year': 'year'})
        else:
            # Create fallback data using the publication trends data
            oa_df = generate_sample_oa_data(pub_trends_df)
    except Exception as e:
        st.warning(f"Error processing open access data: {str(e)}. Using estimated data.")
        oa_df = generate_sample_oa_data(pub_trends_df)
    
    # Extract institution data (if available)
    try:
        institutions = []
        for work in works_data['results']:
            if 'authorships' in work:
                for authorship in work['authorships']:
                    if 'institution' in authorship and authorship['institution']:
                        inst_name = authorship['institution'].get('display_name', 'Unknown')
                        if inst_name != 'Unknown':
                            institutions.append(inst_name)
        
        if institutions:
            inst_counts = pd.Series(institutions).value_counts().head(10)
            inst_df = pd.DataFrame({'institution': inst_counts.index, 'publications': inst_counts.values})
            
            # Calculate average citations per institution
            inst_citations = {}
            for work in works_data['results']:
                cited_by_count = work.get('cited_by_count', 0)
                if 'authorships' in work:
                    for authorship in work['authorships']:
                        if 'institution' in authorship and authorship['institution']:
                            inst_name = authorship['institution'].get('display_name', 'Unknown')
                            if inst_name != 'Unknown':
                                if inst_name not in inst_citations:
                                    inst_citations[inst_name] = []
                                inst_citations[inst_name].append(cited_by_count)
            
            for inst in inst_df['institution']:
                if inst in inst_citations:
                    total_citations = sum(inst_citations[inst])
                    inst_df.loc[inst_df['institution'] == inst, 'total_citations'] = total_citations
                else:
                    inst_df.loc[inst_df['institution'] == inst, 'total_citations'] = 0
        else:
            inst_df = generate_sample_institution_data()
    except Exception as e:
        st.warning(f"Error processing institution data: {str(e)}. Using sample data.")
        inst_df = generate_sample_institution_data()
    
    # Extract research topics/concepts if available
    try:
        concepts = []
        for work in works_data['results']:
            if 'concepts' in work:
                for concept in work['concepts']:
                    concepts.append({
                        'name': concept.get('display_name', 'Unknown'),
                        'score': concept.get('score', 0)
                    })
        
        if concepts:
            concept_df = pd.DataFrame(concepts)
            concept_df = concept_df.groupby('name')['score'].mean().reset_index()
            concept_df = concept_df.sort_values('score', ascending=False).head(15)
        else:
            concept_df = generate_sample_concept_data()
    except Exception as e:
        st.warning(f"Error processing concept data: {str(e)}. Using sample data.")
        concept_df = generate_sample_concept_data()
    
    # Summary statistics
    stats = {
        'total_papers': len(works_df),
        'total_citations': int(works_df['cited_by_count'].sum()) if 'cited_by_count' in works_df.columns else 0,
        'avg_citations': round(works_df['cited_by_count'].mean(), 1) if 'cited_by_count' in works_df.columns else 0,
        'h_index': calculate_h_index(works_df['cited_by_count'].tolist()) if 'cited_by_count' in works_df.columns else 0
    }
    
    return {
        'publication_trends': pub_trends_df,
        'citation_metrics': yearly_citations,
        'open_access': oa_df,
        'institutions': inst_df,
        'concepts': concept_df,
        'stats': stats
    }

# Helper functions for generating sample data when API fails
def generate_sample_data(start_year, end_year):
    """Generate sample data for visualization when API fails"""
    years = list(range(start_year, end_year + 1))
    
    # Publication trends
    pub_trends_df = pd.DataFrame({
        'year': years,
        'publications': [30 + (i-start_year)*5 + ((i-start_year) % 3) * 8 for i in years]
    })
    
    # Citation metrics
    yearly_citations = pd.DataFrame({
        'year': years,
        'avg_citations': [max(8 - (datetime.now().year - year) * 0.8, 0.5) for year in years],
        'paper_count': pub_trends_df['publications']
    })
    yearly_citations['total_citations'] = (yearly_citations['avg_citations'] * yearly_citations['paper_count']).astype(int)
    
    # Open access
    oa_df = pd.DataFrame({
        'year': years,
        'open_access': [int(pub_trends_df.loc[pub_trends_df['year'] == year, 'publications'].values[0] * (0.3 + (year-start_year)*0.05)) for year in years],
        'total': pub_trends_df['publications']
    })
    oa_df['percentage'] = (oa_df['open_access'] / oa_df['total'] * 100).round(1)
    
    # Institutions
    inst_df = pd.DataFrame({
        'institution': ['Harvard University', 'Stanford University', 'MIT', 'University of California, Berkeley', 'University of Oxford',
                       'University of Cambridge', 'ETH Zurich', 'Princeton University', 'Imperial College London', 'University of Toronto'],
        'publications': [25, 22, 20, 18, 17, 16, 15, 14, 13, 12],
        'total_citations': [320, 290, 270, 240, 220, 200, 190, 180, 170, 160]
    })
    
    # Concepts
    concept_df = pd.DataFrame({
        'name': ['Machine Learning', 'Data Visualization', 'Computer Science', 'Information Retrieval', 
                'Big Data', 'Artificial Intelligence', 'Human-Computer Interaction', 'Natural Language Processing',
                'Computer Vision', 'Data Mining', 'Knowledge Management', 'Information Systems', 
                'Data Science', 'Pattern Recognition', 'Social Computing'],
        'score': [0.95, 0.92, 0.85, 0.82, 0.78, 0.76, 0.72, 0.69, 0.67, 0.65, 0.62, 0.60, 0.58, 0.55, 0.52]
    })
    
    # Stats
    stats = {
        'total_papers': sum(pub_trends_df['publications']),
        'total_citations': sum(yearly_citations['total_citations']),
        'avg_citations': round(sum(yearly_citations['total_citations']) / sum(pub_trends_df['publications']), 1),
        'h_index': 15
    }
    
    return {
        'publication_trends': pub_trends_df,
        'citation_metrics': yearly_citations,
        'open_access': oa_df,
        'institutions': inst_df,
        'concepts': concept_df,
        'stats': stats
    }

def generate_sample_oa_data(pub_trends_df):
    """Generate sample open access data"""
    oa_df = pd.DataFrame({
        'year': pub_trends_df['year'],
        'total': pub_trends_df['publications'],
        'open_access': (pub_trends_df['publications'] * pub_trends_df['year'].apply(
            lambda x: min(0.3 + (x - pub_trends_df['year'].min()) * 0.05, 0.8)
        )).astype(int)
    })
    oa_df['percentage'] = (oa_df['open_access'] / oa_df['total'] * 100).round(1)
    return oa_df

def generate_sample_institution_data():
    """Generate sample institution data"""
    return pd.DataFrame({
        'institution': ['Harvard University', 'Stanford University', 'MIT', 'University of California, Berkeley', 'University of Oxford',
                       'University of Cambridge', 'ETH Zurich', 'Princeton University', 'Imperial College London', 'University of Toronto'],
        'publications': [25, 22, 20, 18, 17, 16, 15, 14, 13, 12],
        'total_citations': [320, 290, 270, 240, 220, 200, 190, 180, 170, 160]
    })

def generate_sample_concept_data():
    """Generate sample concept data"""
    return pd.DataFrame({
        'name': ['Machine Learning', 'Data Visualization', 'Computer Science', 'Information Retrieval', 
                'Big Data', 'Artificial Intelligence', 'Human-Computer Interaction', 'Natural Language Processing',
                'Computer Vision', 'Data Mining', 'Knowledge Management', 'Information Systems', 
                'Data Science', 'Pattern Recognition', 'Social Computing'],
        'score': [0.95, 0.92, 0.85, 0.82, 0.78, 0.76, 0.72, 0.69, 0.67, 0.65, 0.62, 0.60, 0.58, 0.55, 0.52]
    })

def calculate_h_index(citations):
    """Calculate h-index from citation counts"""
    if not citations:
        return 0
    sorted_citations = sorted(citations, reverse=True)
    h = 0
    for i, citation in enumerate(sorted_citations, 1):
        if citation >= i:
            h = i
        else:
            break
    return h

# Add debugging function
def safe_visualize(data_dict, key):
    """Safely access and visualize data, with fallback to sample data if key is missing"""
    if key not in data_dict:
        st.error(f"Data key '{key}' is missing. Using sample data instead.")
        if key == 'open_access':
            return generate_sample_oa_data(data_dict.get('publication_trends', pd.DataFrame({'year': [2020, 2021, 2022], 'publications': [10, 15, 20]})))
        elif key == 'institutions':
            return generate_sample_institution_data()
        elif key == 'concepts':
            return generate_sample_concept_data()
        else:
            return None
    return data_dict[key]

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">📚 OpenAlex Scholarly Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    st.sidebar.title("Query Settings")
    
    # Input for research topic
    topic = st.sidebar.text_input("Research Topic", value="data visualization")
    
    # Date range
    current_year = datetime.now().year
    start_year = st.sidebar.number_input("Start Year", min_value=2010, max_value=current_year-1, value=current_year-5)
    end_year = st.sidebar.number_input("End Year", min_value=start_year+1, max_value=current_year, value=current_year)
    
    # Data limit
    data_limit = st.sidebar.slider("Number of Papers to Analyze", min_value=50, max_value=500, value=200, step=50)
    
    # Update button
    if st.sidebar.button("Update Dashboard"):
        st.session_state.update_clicked = True
        st.session_state.data = None  # Reset cached data
    
    # Initialize session state
    if 'update_clicked' not in st.session_state:
        st.session_state.update_clicked = False
    
    if 'data' not in st.session_state or st.session_state.data is None:
        if st.session_state.update_clicked:
            with st.spinner(f"Analyzing scholarly data on '{topic}' from {start_year} to {end_year}..."):
                st.session_state.data = process_data(topic, start_year, end_year, data_limit)
        else:
            # First time loading - use default data
            with st.spinner("Loading initial dashboard data..."):
                st.session_state.data = process_data("data visualization", current_year-5, current_year, 200)
    
    data = st.session_state.data
    
    # Display summary statistics in cards
    st.markdown('<h2 class="section-header">Summary Statistics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{data['stats']['total_papers']}</div>
            <div class="stat-label">Publications</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{data['stats']['total_citations']}</div>
            <div class="stat-label">Total Citations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{data['stats']['avg_citations']}</div>
            <div class="stat-label">Avg Citations per Paper</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{data['stats']['h_index']}</div>
            <div class="stat-label">h-index</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Institutions and Research Topics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h2 class="section-header">Top Contributing Institutions</h2>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        inst_data = safe_visualize(data, 'institutions')
        if inst_data is not None:
            fig_inst = px.scatter(
                inst_data,
                x='publications',
                y='total_citations',
                size='publications',
                color='total_citations',
                hover_name='institution',
                text='institution',
                title='Institutions by Publications and Citations',
                labels={'publications': 'Number of Publications', 'total_citations': 'Total Citations'},
                template='plotly_white'
            )
            
            fig_inst.update_traces(
                textposition='top center',
                marker=dict(sizemode='area', sizeref=0.1)
            )
            
            st.plotly_chart(fig_inst, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="section-header">Related Research Topics</h2>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        concept_data = safe_visualize(data, 'concepts')
        if concept_data is not None:
            fig_concept = px.bar(
                concept_data,
                x='score',
                y='name',
                orientation='h',
                title='Top Related Research Concepts',
                labels={'score': 'Relevance Score', 'name': 'Concept'},
                color='score',
                color_continuous_scale=px.colors.sequential.Viridis,
                template='plotly_white'
            )
            
            fig_concept.update_layout(
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig_concept, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #f0f2f6;">
        <p>Powered by OpenAlex API • Data retrieved on {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()