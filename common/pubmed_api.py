# # PubMed (NCBI Entrez) API Guide

# This notebook provides examples for querying the PubMed database using the NCBI Entrez E-utilities REST API.

# **Documentation**: [https://www.ncbi.nlm.nih.gov/books/NBK25501/](https://www.ncbi.nlm.nih.gov/books/NBK25501/)

# ## Prerequisite
# No auth is strictly required for low volume, but an API Key is recommended for higher rate limits (10 req/sec vs 3 req/sec).
# You can get an API Key from your [NCBI Account Settings](https://www.ncbi.nlm.nih.gov/account/settings/).
import requests
import json
import time
import xml.etree.ElementTree as ET

# ==========================================
# CONFIGURATION
# ==========================================
# TODO: Replace with your actual API Key if you have one. Leave empty if not.
API_KEY = "" 

# TODO: It is good practice to include your email so NCBI can contact you if there are issues.
EMAIL = "sriharsha4002@gmail.com"
TOOL_NAME = "python_notebook_guide"

BASE_URI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DB = "pubmed"
def get_entrez_data(endpoint, params=None):
    """
    Helper function to make GET requests to NCBI E-utilities.
    """
    if params is None:
        params = {}
    
    # Standard parameters
    params['db'] = DB
    params['email'] = EMAIL
    params['tool'] = TOOL_NAME
    params['retmode'] = params.get('retmode', 'json') # Default to JSON where supported
    
    if API_KEY:
        params['api_key'] = API_KEY
    
    uri = f"{BASE_URI}/{endpoint}"
    try:
        # Sleep slightly to respect rate limits if no API key is used (3 req/sec)
        if not API_KEY:
            time.sleep(0.34)
            
        response = requests.get(uri, params=params)
        response.raise_for_status()
        
        # Return JSON if requested and content type allows, otherwise text
        if params['retmode'] == 'json':
            try:
                return response.json()
            except Exception as e:
                print("Error decoding JSON. Response text:")
                print(response.text[:500]) # Print first 500 chars
                return None
        else:
            return response.text
            
    except requests.exceptions.HTTPError as e:
        print(f"Error: {e}")
        return None
## 1. Search Articles (ESearch)
# Search for articles by query string.
# Endpoint: `esearch.fcgi`
# query = "COVID-19 vaccine efficacy"
# print(f"Searching for: {query}...")

# search_params = {
#     "term": query,
#     "retmax": 5, # Get top 5
#     "sort": "relevance"
# }

# search_results = get_entrez_data("esearch.fcgi", search_params)

# id_list = []
# if search_results:
#     result = search_results.get('esearchresult', {})
#     count = result.get('count')
#     id_list = result.get('idlist', [])
#     print(f"Total Found: {count}")
#     print(f"Top IDs: {id_list}")
## 2. Get Article Summaries (ESummary)
# Get basic metadata (title, authors, source) for the list of IDs.
# Endpoint: `esummary.fcgi`
if id_list:
    ids_str = ",".join(id_list)
    print(f"Fetching summaries for IDs: {ids_str}...")
    
    summary_params = {
        "id": ids_str
    }
    
    summary_data = get_entrez_data("esummary.fcgi", summary_params)
    
    if summary_data:
        # Parse JSON result
        # Note: ESummary JSON format usually has keys as properties under 'result'
        results = summary_data.get('result', {})
        
        # 'uids' list ensures we iterate in order, excluding 'uids' key itself
        uids = results.get('uids', [])
        
        for uid in uids:
            item = results[uid]
            print(f"\nPMID: {uid}")
            print(f"Title: {item.get('title')}")
            print(f"Journal: {item.get('fulljournalname')}")
            print(f"PubDate: {item.get('pubdate')}")
## 3. Fetch Full Details/Abstracts (EFetch)
# Get the full record including abstract.
# Note: `EFetch` returns XML or text usually, JSON support is limited/less common for full records.
# Endpoint: `efetch.fcgi`
# Fetch abstract for the first result
if id_list:
    target_id = id_list[0]
    print(f"Fetching details for PMID {target_id}...")
    
    fetch_params = {
        "id": target_id,
        "rettype": "abstract",
        "retmode": "text" # Get plain text abstract
    }
    
    fetch_data = get_entrez_data("efetch.fcgi", fetch_params)
    
    if fetch_data:
        print("\n--- Abstract ---")
        print(fetch_data)
        print("----------------")
## 4. Find Cited Articles (ELink)
# Find articles that cite a specific article (Pubmed to Pubmed).
# Endpoint: `elink.fcgi`
# Find articles citing our target_id
# LinkName 'pubmed_pubmed_citedconst' finds articles that citation the source
if id_list:
    target_id = id_list[0]
    print(f"Finding citations for PMID {target_id}...")
    
    link_params = {
        "dbfrom": "pubmed",
        "linkname": "pubmed_pubmed_citedin",
        "id": target_id
    }
    
    link_data = get_entrez_data("elink.fcgi", link_params)
    
    if link_data:
        linksets = link_data.get('linksets', [])
        if linksets:
             linksetdbs = linksets[0].get('linksetdbs', [])
             if linksetdbs:
                 cited_ids = linksetdbs[0].get('links', [])
                 print(f"Found {len(cited_ids)} articles citing this paper.")
                 print(f"First 5 Citing PMIDs: {cited_ids[:5]}")
             else:
                 print("No citations found in this dataset.")