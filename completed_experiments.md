# ClinProof v5 Ablation Study - Completed Experiments & Metrics

Based on the JSON results found in `results/v5_ablations/`, the following experiments have been completed.
The comprehensive metric extraction provides the accuracy, precision (P), recall (R), and macro-F1 scores for each run.

## Experiment Types

The ablation study is divided into several specific experiment categories (denoted by the prefix of their tag):

- **[A] Model Comparison**: Evaluates the difference between base models (e.g., Mistral, Qwen2.5) and medical-specific fine-tuned models (e.g., BioMistral).
- **[B] Retrieval Architecture**: Compares different retrieval strategies against each other, specifically isolating Dense Semantic Retrieval (PubMed MedCPT), Keyword Retrieval (BM25), Knowledge Graph (GraphRAG), and a full Hybrid pipeline.
- **[C] Recency (Keyword)**: Evaluates the impact of heavily weighting newer documents during keyword retrieval (BM25) using different alpha values for the MedChangeQA temporal dataset. *(Note: This logic is now deprecated/commented out).*
- **[D] Ensemble Strategy**: Tests whether voting mechanisms (single vote vs. 3-vote) and heterogeneous model ensembles (e.g., mixing Qwen, Meditron, and Llama) improve reasoning robustness.
- **[E] Atomic Decomposition**: Measures the impact of breaking complex clinical claims into isolated atomic propositions prior to context retrieval and reasoning.
- **[F] Combo**: Tests the combination of the best performing atomic decomposition strategy alongside recency-weighted retrieval.
- **[G] Recency (Dense)**: A parameter sweep evaluating dense semantic retrieval (MedCPT) with varying levels of chronological bias (alpha=0.1 to 0.7) to fix the 'NOT ENOUGH INFORMATION' recall collapse on temporal claims.

## Summary Table

| Tag                                           | Dataset     | Experiment Type        | Model                         | Decomp | KG | Retrieval | BM25 | PMed | Rec | Votes | N   | Acc   | P     | R     | F1    | Unan% |
| --------------------------------------------- | ----------- | ---------------------- | ----------------------------- | ------ | -- | --------- | ---- | ---- | --- | ----- | --- | ----- | ----- | ----- | ----- | ----- |
| A3_biomistral_bioasq                          | bioasq      | Model Comparison       | biomistral                    | ✓     | ✗ | Keyword   | ✓   | ✗   | 0.0 | 1     | 166 | 81.9% | 41.0% | 50.0% | 45.0% | 100%  |
| A5_mistral7b_bioasq                           | bioasq      | Model Comparison       | mistral                       | ✓     | ✗ | Keyword   | ✓   | ✗   | 0.0 | 1     | 166 | 80.7% | 67.6% | 58.4% | 60.5% | 100%  |
| A6_qwen14b_nokgbm25_bioasq                    | bioasq      | Model Comparison       | qwen2.5                       | ✓     | ✗ | Keyword   | ✓   | ✗   | 0.0 | 3     | 166 | 79.5% | 66.4% | 68.0% | 67.1% | 87%   |
| B1_dense_only_bioasq                          | bioasq      | Retrieval Architecture | qwen2.5                       | ✓     | ✗ | Dense     | ✗   | ✓   | 0.0 | 3     | 166 | 87.3% | 79.4% | 75.4% | 77.1% | 88%   |
| B2_dense_kg_bioasq                            | bioasq      | Retrieval Architecture | qwen2.5                       | ✓     | ✓ | Dense     | ✗   | ✓   | 0.0 | 3     | 166 | 84.3% | 73.6% | 74.9% | 74.2% | 93%   |
| B3_dense_bm25_bioasq                          | bioasq      | Retrieval Architecture | qwen2.5                       | ✓     | ✗ | Hybrid    | ✓   | ✓   | 0.0 | 3     | 166 | 90.4% | 84.9% | 81.1% | 82.8% | 94%   |
| B4_full_pipeline_bioasq                       | bioasq      | Retrieval Architecture | qwen2.5                       | ✓     | ✓ | Hybrid    | ✓   | ✓   | 0.0 | 3     | 166 | 88.0% | 79.9% | 78.4% | 79.1% | 92%   |
| D1_qwen14b_1vote_bioasq                       | bioasq      | Ensemble Strategy      | qwen2.5                       | ✓     | ✗ | Keyword   | ✓   | ✗   | 0.0 | 1     | 166 | 77.1% | 63.9% | 66.5% | 64.9% | 100%  |
| D2_qwen14b_3vote_bioasq                       | bioasq      | Ensemble Strategy      | qwen2.5                       | ✓     | ✗ | Keyword   | ✓   | ✗   | 0.0 | 3     | 166 | 80.1% | 67.1% | 68.4% | 67.7% | 89%   |
| D4_medensemble_3_bioasq                       | bioasq      | Ensemble Strategy      | meditron,medllama2,biomistral | ✓     | ✗ | Keyword   | ✓   | ✗   | 0.0 | 3     | 166 | 81.9% | 41.0% | 50.0% | 45.0% | 92%   |
| D5_hybridensemble_3_bioasq                    | bioasq      | Ensemble Strategy      | qwen2.5,meditron,llama3.1     | ✓     | ✗ | Keyword   | ✓   | ✗   | 0.0 | 3     | 166 | 83.7% | 72.7% | 64.1% | 66.5% | 67%   |
| B1_dense_only_healthfc_test                   | healthfc    | Retrieval Architecture | qwen2.5                       | ✓     | ✗ | Dense     | ✗   | ✓   | 0.0 | 3     | 75  | 49.3% | 45.1% | 42.2% | 42.0% | 85%   |
| B2_dense_kg_healthfc_test                     | healthfc    | Retrieval Architecture | qwen2.5                       | ✓     | ✓ | Dense     | ✗   | ✓   | 0.0 | 3     | 75  | 53.3% | 47.6% | 45.6% | 46.2% | 80%   |
| B3_dense_bm25_healthfc_test                   | healthfc    | Retrieval Architecture | qwen2.5                       | ✓     | ✗ | Hybrid    | ✓   | ✓   | 0.0 | 3     | 75  | 45.3% | 40.3% | 37.3% | 37.1% | 77%   |
| B4_full_pipeline_healthfc_test                | healthfc    | Retrieval Architecture | qwen2.5                       | ✓     | ✓ | Hybrid    | ✓   | ✓   | 0.0 | 3     | 75  | 52.0% | 50.3% | 46.2% | 46.3% | 85%   |
| E1_qwen14b_3vote_with_decomp_healthfc_test    | healthfc    | Atomic Decomposition   | qwen2.5                       | ✓     | ✗ | Keyword   | ✓   | ✗   | 0.0 | 3     | 75  | 53.3% | 47.6% | 44.8% | 45.4% | 83%   |
| E2_qwen14b_3vote_no_decomp_healthfc_test      | healthfc    | Atomic Decomposition   | qwen2.5                       | ✓     | ✗ | Keyword   | ✓   | ✗   | 0.0 | 3     | 75  | 30.7% | 53.8% | 33.7% | 25.8% | 71%   |
| C1_bm25_flat_medchangeqa                      | medchangeqa | Recency (Keyword)      | qwen2.5                       | ✓     | ✗ | Keyword   | ✓   | ✗   | 0.0 | 3     | 512 | 26.6% | 25.6% | 21.7% | 20.8% | 81%   |
| C2_bm25_recency_a0.3_medchangeqa              | medchangeqa | Recency (Keyword)      | qwen2.5                       | ✓     | ✗ | Keyword   | ✓   | ✗   | 0.3 | 3     | 512 | 26.6% | 25.8% | 21.5% | 20.7% | 82%   |
| C3_bm25_recency_a0.7_medchangeqa              | medchangeqa | Recency (Keyword)      | qwen2.5                       | ✓     | ✗ | Keyword   | ✓   | ✗   | 0.7 | 3     | 451 | 22.4% | 19.9% | 17.1% | 16.8% | 77%   |
| F2_qwen14b_recency0.3_with_decomp_medchangeqa | medchangeqa | Combo (Recency+Decomp) | qwen2.5                       | ✓     | ✗ | Keyword   | ✓   | ✗   | 0.3 | 3     | 512 | 26.6% | 26.7% | 21.7% | 21.3% | 79%   |
| G1a_pubmed_recency_a0.1_medchangeqa_test      | medchangeqa | Recency (Dense)        | qwen2.5                       | ✓     | ✗ | Dense     | ✗   | ✓   | 0.1 | 3     | 103 | 37.9% | 20.6% | 29.4% | 21.7% | 95%   |
| G1b_pubmed_recency_a0.2_medchangeqa_test      | medchangeqa | Recency (Dense)        | qwen2.5                       | ✓     | ✗ | Dense     | ✗   | ✓   | 0.2 | 3     | 103 | 37.9% | 23.8% | 29.4% | 22.2% | 89%   |
| G1c_pubmed_recency_a0.3_medchangeqa_test      | medchangeqa | Recency (Dense)        | qwen2.5                       | ✓     | ✗ | Dense     | ✗   | ✓   | 0.3 | 3     | 103 | 35.9% | 20.0% | 27.9% | 20.8% | 88%   |
| G1d_pubmed_recency_a0.5_medchangeqa_test      | medchangeqa | Recency (Dense)        | qwen2.5                       | ✓     | ✗ | Dense     | ✗   | ✓   | 0.5 | 3     | 103 | 37.9% | 24.2% | 29.4% | 22.5% | 90%   |
| G1e_pubmed_recency_a0.7_medchangeqa_test      | medchangeqa | Recency (Dense)        | qwen2.5                       | ✓     | ✗ | Dense     | ✗   | ✓   | 0.7 | 3     | 103 | 38.8% | 24.0% | 30.2% | 22.5% | 89%   |
| G1f_pubmed_flat_medchangeqa_test              | medchangeqa | Recency (Dense)        | qwen2.5                       | ✓     | ✗ | Dense     | ✗   | ✓   | 0.0 | 3     | 103 | 29.1% | 17.3% | 22.8% | 18.8% | 88%   |
