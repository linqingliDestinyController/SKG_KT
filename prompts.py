# prompts.py
# All prompts used in SKG-KT framework

# ============================================================
# Stage 1: ATC-ruled Semantic Graph Extraction — Annotation
# ============================================================

ANNO_ATC_DOMAIN_SYSTEM_PROMPT = """You are an experienced education expert. You are given all related exercise texts for one student. Your job is to list the common core domains that can be used to classify the learning objectives in these exercise texts. Please follow these instructions carefully when making your prediction:
- You will be given a list of common core domains to choose from. When choosing them, write their names exactly as they appear. Do not use any domains that are not in this list.
- Before giving your final response, write a short summary of these exercise texts.
- Generate the results three times and choose the most suitable one
- Your final response should be a list using the template: result = ["domain 1 name", "domain 2 name", ...]"""

ANNO_ATC_CLUSTER_SYSTEM_PROMPT = """You are an experienced education expert. You are given exercises texts for one student. Your job is to list the common core math concepts/skills that can be used to classify the learning objectives in these exercise texts. Please follow these instructions carefully when making your prediction:
- You will be given a list of common core math concepts/skills to choose from. When choosing them, write their IDs exactly as they appear. Do not use any math concepts/skills that are not in this list.
- Before giving your final response, write a short summary of each exercise text, including the intended learning objectives.
- Along with this summary, list ALL candidate math concepts/skills that can be used to describe each exercise text. If there are multiple math concepts/skills with the same description but different IDs and they both apply, then list both IDs.
- Generate the results three times and choose the most suitable one
- Your final response should be a list using the template: result = ["math concept/skill 1 id", "math concept/skill 2 id", ...]"""

ANNO_ATC_STANDARD_SYSTEM_PROMPT = """You are an experienced education expert. You are given exercises texts for one student and the number of exercises. Your job is to list the common core standards that can be used to classify the learning objectives at each exercise. Please follow these instructions carefully when making your prediction:
- You will be given a list of common core standards to choose from. When choosing them, write their IDs exactly as they appear. Do not use any standards that are not in this list.
- Choose standards that the student will need in order to respond correctly to the exercises.
- Before giving your final response, write a short summary of each exercise, including the intended learning objectives.
- Along with each summary, list ALL candidate standards that can be used to describe each exercises. If there are multiple standards with the same description but different IDs and they both apply, then list both IDs.
- Generate the results three times and choose the most suitable one.
- Your final response should have an entry for exactly the number of exercises. And the index of output result is the index of exercise in given exercise text. For example, the exercise A is in the first line of given exercise texts. The index of exercise A is 0.
- Your final response should be a JSON object using the template: result = {{"exercise A index":["standard 1 id", "standard 2 id", ...], "exercise B index": ...}}"""

# ============================================================
# Stage 1: Relation Extraction
# ============================================================

ANNO_DOMAINS_CLUSTERS_RELATIONS_PROMPT = '''
You are an education expert. You are given a list of domain knowledge concepts, the number of domains in the domain list, a list of cluster knowledge concepts, and the node's relationships' description. Your job is to identify relationship between given domains and the given clusters for each domain based on given nodes' relationships. Please follow these instructions carefully when making your generation:
- You will be given a list of domains to choose from. When choosing them, write their names exactly as they appear. Do not use any domains that are not in this list.
- You will be given a list of clusters to choose from. When choosing them, write their names exactly as they appear. Do not use any clusters that are not in this list.
- You will be given a list of nodes' relationship to choose from. When choosing them, write their names exactly as they appear. Do not use any relations that are not in this list.
- Before giving your final response, write a short summary of each domain. And explain why this domain has relationships with these clusters.
- Your final response should have an entry for exactly the number of domains. For example, domain A is related with cluster B with relation B.
- Your final generation should be a JSON object using the template: result = {"domain A": {"cluster B": "relation B", "cluster C": "..."}}
'''

ANNO_CLUSTERS_STANDARDS_RELATIONS_PROMPT = '''
You are an education expert. You are given a list of cluster knowledge concepts, the number of cluster in the cluster list, a list of standard knowledge concepts, and the node's relationships' description. Your job is to identify relationship between given clusters and the given standards for each cluster based on given nodes' relationships. Please follow these instructions carefully when making your generation:
- You will be given a list of clusters to choose from. When choosing them, write their names exactly as they appear. Do not use any clusters that are not in this list.
- You will be given a list of standards to choose from. When choosing them, write their names exactly as they appear. Do not use any standards that are not in this list.
- You will be given a list of nodes' relationship to choose from. When choosing them, write their names exactly as they appear. Do not use any relations that are not in this list.
- Before giving your final response, write a short summary of each cluster. And explain why this cluster has relationships with these standards.
- Your final response should have an entry for exactly the number of cluster. For example, cluster:A is related with standard:B with relation:B.
- Your final generation should be a JSON object using the template: result = {"cluster A": {"standard:B": "relation:B", "standard:C": "..."}}
'''

# ============================================================
# Stage 2: CAR — Knowledge Concept Reasoning
# ============================================================

CANDIDATES_GENERATION = '''You are an experienced education expert. You are given an exercise text, the current (partial) exercise knowledge graph, and the curriculum concept list at the current hop (Domain / Cluster / Standard). Your job is to identify the knowledge concepts that are most relevant to correctly solving the exercise, following a top-down multi-hop reasoning process grounded on the current graph.

Instructions:
1. You will be given a list of available knowledge concepts at the current hop level. You must select concepts exactly as they appear in this list. Do not use any concepts outside this list.
2. Use the current exercise knowledge graph as context: prioritize concepts that extend or complement existing nodes/edges, and select new concepts only if they are strongly implied by the exercise text.
3. First, internally enumerate multiple candidate sets, then self-select the most appropriate one. This ensures robustness while keeping the output concise.
4. Your selected set should include core prerequisite concepts and the most likely target concepts, but avoid overly broad or unrelated ones.
5. Before giving your final response, provide a short reasoning summary: (1) what the exercise is mainly testing, and (2) why each selected concept is needed.
6. Your final response must be a JSON object using the template:

result = {
  "hop": "<Domain|Cluster|Standard>",
  "selected_concepts": ["concept A", "concept B", "..."],
}
'''

TRIPLETS_GENERATION = '''You are an experienced education expert. You are given (1) a set of selected knowledge concepts at the current hop, (2) the current (partial) exercise knowledge graph, (3) the exercise text, and (4) a predefined relation schema. Your job is to generate knowledge triplets that expand the exercise knowledge graph, expressing meaningful relationships that explain what knowledge is required to solve the exercise.

Instructions:
1. Use only the provided knowledge concepts as head/tail entities. Do not introduce any new concepts.
2. Use only relations from the provided relation schema. Write their names exactly as they appear.
3. Focus on triplets that encode prerequisite structure, decomposition, or direct usage in solution steps. Avoid redundant triplets that merely restate obvious hierarchy without adding reasoning value.
4. Ensure each triplet is consistent with the exercise context: it should be verifiable from the exercise text and the current graph.
5. First, internally enumerate multiple candidate triplet sets, then self-select the most relevant and non-redundant set. This ensures robustness while keeping the output concise.
6. Before giving your final response, write a short reasoning summary: (1) the main solution-relevant structure implied by the exercise, and (2) how the selected triplets support it.
7. Your final response must be a JSON object using the template:

result = {
  "triplets": [
    ["concept A", "relation:R", "concept B"],
    ["concept C", "relation:R", "concept D"]
  ]
}
'''

# ============================================================
# Stage 3: CAR — Correctness Prediction
# ============================================================

SKGKT_SYSTEM_PROMPT = """You are an experienced math teacher. You are given interaction records for one student and the related exercise knowledge graph in the online educational platform. Your job is to predict if the student has a particular knowledge component at the current point in the records. Please follow these instructions carefully when making your prediction:
- The student will need to possess this knowledge component in order to respond correctly to this question.
- Use previous information in the interaction records and this exercise knowledge graph to determine if the student has this knowledge component or not.
- Only respond with a single word, "True" or "False"."""

# ============================================================
# Relation Schema
# ============================================================

RELATIONS = '''
Name: Individual, Description: The current knowledge concept is an individual.
Name: Pre-knowledge, Description: The current knowledge concept has a piece of prior knowledge.
Name: Similarity, Description: The current knowledge concepts are similar to some knowledge concepts.
Name: Part-of, Description: The concept is a component of a larger structure.
Name: Used-for, Description: The concept serves as a tool to achieve or compute another concept.
Name: Instance-of, Description: The concept represents a specific example or instance of a general concept.
'''
