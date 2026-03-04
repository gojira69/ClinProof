"""
ClinProof GraphRAG Retriever (v2)
Atomic proposition decomposition -> Entity linking -> Multi-hop KG traversal -> Rich context generation
"""
import logging, pickle, re, json
from collections import defaultdict, deque
import networkx as nx

log = logging.getLogger("graph_retriever")

# Relationships that carry actual medical knowledge (skip structural junk)
USEFUL_RELS = {
    "isa", "inverse_isa", "may_treat", "may_prevent", "may_be_treated_by",
    "mechanism_of_action", "has_mechanism_of_action", "has_physiologic_effect",
    "has_active_ingredient", "has_ingredient", "causative_agent",
    "finding_site", "has_target", "associated_with", "causes",
    "has_part", "part_of", "disease_may_have_finding",
    "disease_may_have_associated_disease", "classified_as",
    "has_manifestation", "mapped_to", "due_to", "occurs_in",
    "RO", "RB", "RN", "PAR", "CHD",
    "has_tradename", "tradename_of", "precise_active_ingredient_of",
    "ingredient_of", "form_of", "has_form",
    "procedure_context_of", "associated_procedure_of",
    "component_of", "measured_by",
}

# Junk relationships to skip entirely
SKIP_RELS = {
    "has_class", "has_answer", "has_expanded_form", "expanded_form_of",
    "property_of", "time_aspect_of", "system_of", "scale_type_of",
}

# Junk node labels (administrative / LOINC noise)
JUNK_LABEL_PATTERNS = [
    r"^Dose form", r"^Frequency prescribed", r"^Refills prescribed",
    r"^Amount dispensed", r"^Days' supply", r"^Date ", r"^Medication volume",
    r"^Medication duration", r"^Quantity prescribed", r"^Reason for missed",
    r":Type:Pt:", r":Num:Pt:", r":NRat:Pt:", r":Date:Pt:", r":Vol:Pt:",
    r":Find:Pt:", r":Prid:Pt:", r":Time:Pt:",
]

def _is_junk_label(label):
    """Return True if the label looks like LOINC/admin junk."""
    for pat in JUNK_LABEL_PATTERNS:
        if re.search(pat, label):
            return True
    return False


def load_graph(graph_path):
    with open(graph_path, "rb") as f:
        return pickle.load(f)


class EntityLinker:
    """N-gram based entity linker with stop-word filtering."""

    def __init__(self, G):
        self.G = G
        self.label_map = defaultdict(list)
        for node_id, data in G.nodes(data=True):
            label = data.get("label", "")
            if label:
                self.label_map[label.lower()].append(node_id)
        log.info(f"EntityLinker: {len(self.label_map)} labels")

    def link(self, text, max_entities=20):
        """Exact n-gram matching."""
        words = re.findall(r'[a-zA-Z0-9\-]+', text.lower())
        matched, seen_labels = [], set()
        stopwords = {
            "the", "and", "with", "this", "that", "for", "which", "patient",
            "history", "effects", "following", "primary", "primarily",
            "through", "these", "those", "have", "has", "had", "been",
            "most", "likely", "would", "what", "how", "does", "used",
            "man", "woman", "year", "old", "new", "due", "commonly",
            "known", "type", "including", "also", "one", "two",
            "her", "his", "she", "day", "days", "ago", "weeks",
        }

        # Check n-grams from length 5 down to 1
        for n in range(5, 0, -1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n])
                if len(ngram) < 4 and n == 1:
                    continue
                if n == 1 and ngram in stopwords:
                    continue
                if ngram in seen_labels:
                    continue

                if ngram in self.label_map:
                    seen_labels.add(ngram)
                    for node in self.label_map[ngram]:
                        if node not in {m[0] for m in matched}:
                            matched.append((node, ngram))

            if len(matched) >= max_entities:
                break

        return [m[0] for m in matched[:max_entities]]

    def fuzzy_link(self, term, max_results=5):
        """Substring matching: find KG labels that CONTAIN the search term.
        Returns nodes sorted by: (1) UMLS nodes first, (2) shorter labels first.
        """
        term_lower = term.lower().strip()
        if len(term_lower) < 3:
            return []

        candidates = []
        for label, node_ids in self.label_map.items():
            # The label must contain our term (not the other way around)
            if term_lower in label:
                for nid in node_ids:
                    nd = self.G.nodes[nid]
                    # Priority: prefer UMLS nodes with definitions, then UMLS, then others
                    has_def = 1 if nd.get("definition", "").strip() else 0
                    is_umls = 1 if nd.get("node_type") == "umls" else 0
                    # Score: higher = better. Prefer short labels (more specific), UMLS, definitions  
                    score = has_def * 10 + is_umls * 5 - len(label) * 0.01
                    candidates.append((nid, label, score))

        # Sort by score descending, take top
        candidates.sort(key=lambda x: x[2], reverse=True)
        seen = set()
        results = []
        for nid, label, score in candidates:
            if nid not in seen:
                seen.add(nid)
                results.append(nid)
            if len(results) >= max_results:
                break
        return results


class AtomicDecomposer:
    """Uses an LLM to decompose a clinical question into atomic medical propositions."""

    DECOMPOSE_PROMPT = """You are a medical knowledge extractor. Given a clinical question, extract the key medical entities and facts that need to be verified.

Return a JSON object with:
- "entities": list of specific medical entities mentioned (drug names, disease names, procedures, anatomical terms)
- "propositions": list of atomic medical claims implied by the question that need verification

Rules:
- Only include medical/clinical terms, not general words
- Drug names should be exact (e.g., "carvedilol", not "medication")
- Disease names should be specific (e.g., "heart failure", not "condition")
- Propositions should be simple factual claims (e.g., "carvedilol is a beta-blocker")

Example for "A patient with heart failure is prescribed carvedilol. What is its mechanism?":
{
  "entities": ["carvedilol", "heart failure"],
  "propositions": ["carvedilol is used to treat heart failure", "carvedilol is a beta-blocker", "carvedilol blocks adrenergic receptors"]
}

Respond with valid JSON only."""

    def __init__(self, llm):
        self.llm = llm

    def decompose(self, question, options=None):
        """Decompose question into entities and propositions."""
        user_msg = f"Clinical question: {question}"
        if options:
            opts_str = "\n".join(f"{k}. {v}" for k, v in options.items())
            user_msg += f"\n\nAnswer options:\n{opts_str}"

        messages = [
            {"role": "system", "content": self.DECOMPOSE_PROMPT},
            {"role": "user", "content": user_msg}
        ]
        raw = self.llm.generate(messages)
        parsed = self.llm.extract_json(raw)

        entities = parsed.get("entities", [])
        propositions = parsed.get("propositions", [])

        # Always include raw entity extraction from question as fallback
        if not entities:
            entities = self._fallback_entities(question)

        return entities, propositions

    def _fallback_entities(self, text):
        """Simple regex fallback to extract capitalized medical terms."""
        # Look for capitalized multi-word terms and known patterns
        terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+)*\b', text)
        return [t.lower() for t in terms if len(t) > 3]


class GraphRetriever:
    """Multi-hop KG retriever with atomic proposition support."""

    def __init__(self, graph_path, config, llm=None):
        log.info("Loading Knowledge Graph...")
        self.G = load_graph(graph_path)
        self.linker = EntityLinker(self.G)
        self.max_hops = config.get("retrieval", {}).get("multihop", {}).get("max_hops", 2)
        self.llm = llm
        self.decomposer = AtomicDecomposer(llm) if llm else None

    def retrieve(self, query, k=20, options=None):
        """Retrieve rich KG context for a query."""

        # Step 1: Atomic decomposition (if LLM available)
        extra_entities = []
        propositions = []
        if self.decomposer:
            extra_entities, propositions = self.decomposer.decompose(query, options)
            log.info(f"Atomic decomposition: entities={extra_entities}, props={propositions}")

        # Step 2: Entity linking
        # Use exact n-gram matching on the raw query
        seed_nodes = self.linker.link(query, max_entities=10)

        # For LLM-extracted entities, ALWAYS try both exact AND fuzzy
        # Because exact might find an RxNorm node (no definition) while fuzzy finds the UMLS node (with definition)
        for ent in extra_entities:
            ent_nodes = self.linker.link(ent, max_entities=3)
            fuzzy_nodes = self.linker.fuzzy_link(ent, max_results=3)
            for n in ent_nodes + fuzzy_nodes:
                if n not in seed_nodes:
                    seed_nodes.append(n)

        # Also fuzzy-link entities from propositions
        for prop in propositions:
            prop_nodes = self.linker.link(prop, max_entities=3)
            for n in prop_nodes:
                if n not in seed_nodes:
                    seed_nodes.append(n)

        # Also link key medical terms from answer options (skip generic words)
        skip_option_words = {"selective", "inhibition", "blockade", "activation", "receptor"}
        if options:
            for opt_text in options.values():
                opt_lower = opt_text.lower()
                # Skip if it's mostly generic words
                opt_words = set(opt_lower.split())
                if len(opt_words - skip_option_words) < 2:
                    continue
                opt_nodes = self.linker.fuzzy_link(opt_lower, max_results=2)
                for n in opt_nodes:
                    if n not in seed_nodes:
                        seed_nodes.append(n)


        if not seed_nodes:
            return [], []

        # Step 3: For each seed, do a focused multi-hop traversal collecting useful context
        documents = []
        seen_node_ids = set()

        for node in seed_nodes:
            doc = self._build_entity_context(node, seed_nodes, seen_node_ids)
            if doc:
                documents.append(doc)
                seen_node_ids.add(node)

        # Step 4: If we have propositions, add them as context hints
        if propositions:
            prop_text = "Key medical claims to verify:\n" + "\n".join(f"- {p}" for p in propositions)
            documents.insert(0, {
                "title": "Atomic Propositions",
                "content": prop_text,
                "hop": 0,
                "PMID": "KG_propositions"
            })

        return documents[:k], [1.0] * min(len(documents), k)

    def _build_entity_context(self, node, all_seeds, seen):
        """Build a rich context paragraph for a single entity via multi-hop traversal."""
        nd = self.G.nodes.get(node, {})
        node_label = nd.get("label", str(node))

        # Skip junk nodes
        if _is_junk_label(node_label):
            return None

        definition = nd.get("definition", "").strip()
        sty = nd.get("sty", "")

        lines = []
        if sty:
            lines.append(f"Type: {sty}")
        if definition:
            lines.append(f"Definition: {definition[:400]}")

        # Only collect HIGH-VALUE relationship types
        high_value_out = {
            "may_treat", "may_prevent", "may_be_treated_by",
            "mechanism_of_action", "has_mechanism_of_action",
            "has_physiologic_effect", "has_active_ingredient",
            "causative_agent", "finding_site", "has_target",
            "associated_with", "causes",
            "classified_as", "disease_may_have_finding",
        }
        # Medium value - include a few items only
        medium_value_out = {"isa", "inverse_isa", "RO", "RB", "RN", "PAR", "CHD"}
        # Important inverse rels 
        high_value_in = {
            "may_treat", "may_prevent", "may_be_treated_by",
            "mechanism_of_action", "has_mechanism_of_action",
            "has_physiologic_effect", "has_active_ingredient",
            "causative_agent", "causes", "has_target",
        }

        # Collect and filter outgoing edges
        out_rels = defaultdict(list)
        for _, nbr, ed in self.G.out_edges(node, data=True):
            rel = ed.get("rel", "related_to")
            if rel in SKIP_RELS:
                continue
            nbr_data = self.G.nodes.get(nbr, {})
            nbr_label = nbr_data.get("label", str(nbr))
            if _is_junk_label(nbr_label):
                continue
            if rel in high_value_out:
                out_rels[rel].append((nbr, nbr_label, nbr_data))
            elif rel in medium_value_out and len(out_rels.get(rel, [])) < 5:
                out_rels[rel].append((nbr, nbr_label, nbr_data))

        # Collect important incoming edges
        in_rels = defaultdict(list)
        for nbr, _, ed in self.G.in_edges(node, data=True):
            rel = ed.get("rel", "related_to")
            if rel in SKIP_RELS:
                continue
            if rel not in high_value_in and rel not in medium_value_out:
                continue
            nbr_data = self.G.nodes.get(nbr, {})
            nbr_label = nbr_data.get("label", str(nbr))
            if _is_junk_label(nbr_label):
                continue
            if len(in_rels[rel]) < 5:
                in_rels[rel].append((nbr, nbr_label, nbr_data))

        # Format the relationships
        for rel, neighbors in sorted(out_rels.items()):
            unique_labels = list(dict.fromkeys(lbl for _, lbl, _ in neighbors))[:5]
            rel_display = rel.replace("_", " ").title()
            lines.append(f"- {rel_display}: {', '.join(unique_labels)}")

        for rel, neighbors in sorted(in_rels.items()):
            unique_labels = list(dict.fromkeys(lbl for _, lbl, _ in neighbors))[:5]
            rel_display = rel.replace("_", " ").title()
            lines.append(f"- [Inverse] {rel_display}: {', '.join(unique_labels)}")

        # Multi-hop: look at neighbors with definitions (especially useful for RxNorm nodes)
        hop2_lines = []
        for rel, neighbors in out_rels.items():
            for nbr_id, nbr_label, nbr_data in neighbors[:3]:
                if nbr_id in seen:
                    continue
                nbr_def = nbr_data.get("definition", "").strip()
                if nbr_def and len(nbr_def) > 30:
                    hop2_lines.append(f"  [{nbr_label}]: {nbr_def[:200]}")
                # Check if neighbor connects to other seed entities
                for other_seed in all_seeds:
                    if other_seed != node and self.G.has_edge(nbr_id, other_seed):
                        other_label = self.G.nodes[other_seed].get("label", str(other_seed))
                        bridge_rel = self.G[nbr_id][other_seed].get("rel", "related")
                        hop2_lines.append(f"  [{nbr_label}] --{bridge_rel}--> [{other_label}]")

        if hop2_lines:
            lines.append("Related details:")
            lines.extend(hop2_lines[:8])

        if not lines:
            return None

        full_text = f"Entity: {node_label}\n" + "\n".join(lines)
        return {
            "title": f"KG: {node_label}",
            "content": full_text,
            "hop": 0,
            "PMID": f"KG_{node}"
        }

