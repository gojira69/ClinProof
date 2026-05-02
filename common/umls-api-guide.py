import requests
import json

# ==========================================
# CONFIGURATION
# ==========================================
# TODO: Replace with your actual API Key from https://uts.nlm.nih.gov/uts/profile
API_KEY = "c3408bd5-6840-447a-9143-2bebccfff3df"

BASE_URI = "https://uts-ws.nlm.nih.gov/rest"
VERSION = "current" # or specific release like "2023AA"
def get_umls_data(endpoint, params=None):
    """
    Helper function to make GET requests to UMLS API.
    Appends the API_KEY to the request.
    """
    if params is None:
        params = {}
    
    # Authenticate via query parameter
    params['apiKey'] = API_KEY
    
    uri = f"{BASE_URI}/{endpoint}"
    try:
        response = requests.get(uri, params=params)
        response.raise_for_status()
        response.encoding = 'utf-8'
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"Error: {e}")
        # Print response content if available for debugging auth issues
        if response.content:
            print(response.content.decode())
        return None

print(f"Searching for '{search_term}'...")
search_results = get_umls_data(search_endpoint, search_params)

if search_results:
    print(f"Found {search_results['result']['recCount']} results.")
    # Print first 5 results
    for result in search_results['result']['results'][:5]:
        print(f"CUI: {result['ui']} | Name: {result['name']}")

# Using a CUI from the previous search (or a known one like C0015802 for 'Fracture of Femur')
target_cui = "C0015802"
cui_endpoint = f"content/{VERSION}/CUI/{target_cui}"

print(f"Retrieving details for {target_cui}...")
cui_data = get_umls_data(cui_endpoint)

if cui_data:
    result = cui_data['result']
    print(f"Name: {result['name']}")
    print(f"Semantic Types: {[st['name'] for st in result['semanticTypes']]}")
    print(f"Atom Count: {result['atomCount']}")

# Atoms
atoms_endpoint = f"content/{VERSION}/CUI/{target_cui}/atoms"
print(f"\nFetching atoms for {target_cui}...")
atoms_data = get_umls_data(atoms_endpoint, params={"sabs": "SNOMEDCT_US", "pageNumber": 1})
if atoms_data:
    print(f"Found {len(atoms_data['result'])} atoms (filtered by SNOMEDCT_US):")
    for atom in atoms_data['result'][:3]:
        print(f" - [{atom['rootSource']}] {atom['name']}")

# Definitions
defs_endpoint = f"content/{VERSION}/CUI/{target_cui}/definitions"
print(f"\nFetching definitions for {target_cui}...")
defs_data = get_umls_data(defs_endpoint)
if defs_data:
    if defs_data['result']:
        for definition in defs_data['result']:
            print(f" - ({definition['rootSource']}): {definition['value']}")
    else:
        print(" - No definitions found.")

# Relations
rels_endpoint = f"content/{VERSION}/CUI/{target_cui}/relations"
print(f"\nFetching relations for {target_cui}...")
rels_data = get_umls_data(rels_endpoint)
if rels_data:
    for rel in rels_data['result'][:3]:
        print(f" - {rel['relationLabel']} -> {rel['relatedIdName']} ({rel['relatedId']})")

# Example: SNOMED CT code for 'Asthma' (195967001)
source_vocab = "SNOMEDCT_US"
source_code = "195967001"
source_endpoint = f"content/{VERSION}/source/{source_vocab}/{source_code}"

print(f"Looking up {source_vocab} code {source_code}...")
source_data = get_umls_data(source_endpoint)

if source_data:
    res = source_data['result']
    print(f"Mapped to CUI: {res['ui']}")
    print(f"Name: {res['name']}")
    # print(f"URI: {res['uri']}")

# Find ICD-10-CM codes for the SNOMED code above
crosswalk_endpoint = f"crosswalk/{VERSION}/source/{source_vocab}/{source_code}"
crosswalk_params = {"targetSource": "ICD10CM"}

print(f"Finding ICD10CM codes for SNOMED {source_code}...")
crosswalk_data = get_umls_data(crosswalk_endpoint, crosswalk_params)

if crosswalk_data:
    print(f"Crosswalk Results:")
    for result in crosswalk_data['result']:
        print(f" - {result['ui']} ({result['name']})")

# Example TUI: T047 (Disease or Syndrome)
tui = "T047"
sem_endpoint = f"semantic-network/{VERSION}/TUI/{tui}"

print(f"Looking up TUI {tui}...")
sem_data = get_umls_data(sem_endpoint)

if sem_data:
    print(f"Name: {sem_data['result']['name']}")
    print(f"Definition: {sem_data['result']['definition']}")