# From Transcripts to Insights: LLM-Assisted Topic Discovery in Customer Surveys

This repository contains the prompts, and model configuration for a three-stage LLM pipeline that discovers topics, assigns them per feedback item, and generates insights from short, noisy customer feedback.

## Paper

**From Transcripts to Insights: LLM-Assisted Topic Discovery in Customer Surveys**
Christabel Acquaye, Jordan Hosier, Yu Zhou, Vijay K. Gurbani
*University of Maryland, College Park · Vail Systems, Chicago*


## Overview

Telephony survey feedback is short, noisy, and frequently mistranscribed by ASR. Responses range from single words ("okay," "bad," "ten") to extended narratives, and transcription errors obscure intent. This project pairs LLM semantic understanding with topic modeling algorithms to produce verifiable, per-instance topic labels.

The pipeline has three stages:

1. **Topic modeling** — discover a coherent topic taxonomy over a feedback corpus.
2. **Topic assignment** — assign each individual feedback item its best-matching label.
3. **Insight generation** — quantify prevalence, sentiment, and growth trends per topic (illustrative demonstration; not rigorously evaluated).

We compare three topic modeling strategies:

| Method | Abbrev. | Mechanism |
|---|---|---|
| Contextualized Embedding Clustering | CE-Cluster | Task-aware embeddings → HDBSCAN → one LLM call labels clusters from representative documents |
| Parallel Topic Merge | PAR-TM | Batches processed independently for topic lists, then one LLM merge pass into a unified taxonomy |
| Sequential Topic Refinement | SEQ-TR | Batches processed successively; each batch refines/merges/extends the growing topic list. First batch seeded by K-Means |

Each method is run over three input representations (Original, Rewrite, Combined). 
We find a **method–input dependence**: SEQ-TR performs best on raw responses by accumulating context sequentially, while CE-Cluster benefits from LLM rewrites that improve embedding separability.

## Models and Configuration

### Models used

| Role | Model | Source |
|---|---|---|
| Rewrite generation, topic labeling, topic merging, topic assignment, LLM-as-judge, sentiment inference | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) |
| CE-Cluster embeddings (and BERTopic ablation) | `Qwen/Qwen3-Embedding-0.6B` | [Hugging Face](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| Baseline embeddings (K-Means, HDBSCAN, BERTopic default config) | `sentence-transformers/all-MiniLM-L6-v2` | [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Textual entailment evaluation | `cross-encoder/nli-deberta-v3-base` | [Hugging Face](https://huggingface.co/cross-encoder/nli-deberta-v3-base) |

DeepSeek-R1-Distill-Llama-70B emits reasoning before its final answer. 
All parsers in this repository strip the reasoning span and read only the content after it.

### Method hyperparameters

**CE-Cluster** 

| Parameter | Value |
|---|---|
| Embedding model | `Qwen/Qwen3-Embedding-0.6B`, `float16`|
| Embedding batch size | 32 |
| Representation | BERTopic with KeyBERTInspired and MaximalMarginalRelevance |


**PAR-TM** 

| Parameter | Value |
|---|---|
| Batch size | 20 reviews |
| Batch construction | Sequential slices over the corpus in file order |
| Merge input cap | 150 topics, sampled with `random.seed(42)` when the deduplicated list is longer |

**SEQ-TR**

| Parameter | Value |
|---|---|
| Seeding embedding model | `Qwen/Qwen3-Embedding-0.6B`, encode batch size 25 |
| First-batch construction | `MiniBatchKMeans`: `n_clusters=min(20, n_docs)`, `batch_size=256` |
| Seed selection | Review nearest each cluster centroid, via `pairwise_distances_argmin_min` |
| Processing order | Seeds first, then all remaining reviews in corpus order |


**Topic assignment**

| Parameter | Value |
|---|---|
| Batch size | 20 reviews |
| Output format | JSON with `assigned_topic` and `reasoning` |

## Prompts


### Prompt 1 — Synthesized Rewrite Generation

Produces the `REW` input variant.

```
You are an expert specializing in analyzing a large number of call transcripts, where
each transcript is the customer feedback responding to the survey prompt [REDACTED].
You will be asked to generate a summary which is a relevant summarized version of the
feedback that maintains the context of the feedback. Prefix your final answer with
Rewrite: [text].
```

### Prompt 2 — Contextualized Embedding Topic Label Generation (CE-Cluster)


```
You are an expert specialized in analyzing a large number of call transcripts, where
customers provided feedback responding to the survey prompt asking about [REDACTED].
The customers were prompted with the question: [REDACTED]
```

### Prompt 3a — Parallel Topic Labels Generation (PAR-TM)

Run independently on each batch.

```
You are an expert in topic modeling for some customer feedback responding to the
survey prompt [REDACTED]. You will be tasked with performing topic modeling on the
following documents, summarizing and identifying key topics from these customer
feedback. The output should be a numbered list of concise topic labels that represent
the main themes in the documents. You should make a good effort to reduce the number
of topics you started with for each batch.
```

### Prompt 3b — Parallel Topic Label Merge (PAR-TM)

Second pass consolidating batch-level topic lists into a unified taxonomy.

```
You are an expert in topic modeling and data analysis.
Your task is to write the results of merging the following topic modeling results.
The final output should be a consolidated and refined list of topics.
Avoid over-generalization. The final topics should be distinct and clearly represent
related themes from the input list.
```

### Prompt 4 — Sequential Topic Label Generation (SEQ-TR)

```
You are an expert in online, incremental topic modeling for some customer feedback
responding to the survey prompt [REDACTED]. You can also be provided: (1) a current
topic dictionary with names, short descriptions, and exemplar quotes; and (2) a new
batch of feedback with IDs. For each customer feedback, decide whether it clearly fits
an existing topic. If so, return the existing topic exactly as written. If not, propose
a new topic and one-sentence description for that topic. Merge existing topics if
similar themes are detected, updating names/descriptions if needed.
```


### Prompt 5 — Topic Labeling Assignment (Stage 2)

Assignment is deferred until the taxonomy is final, and is applied identically across all methods and baselines.

```
You are a topic-modeling expert tasked with assigning topic labels to individual
customer feedback. For each complaint, your task is to select the best-fitting topic
label from a provided candidate list. Labels must be chosen exactly as written; no new
labels may be invented. If no candidate is appropriate, return "NONE,".
```

### Prompt 6 — Sentiment Labeling Assignment (Stage 3, telephony only)

```
You are an expert specialized in analyzing for some customer feedback responding to
the survey prompt [REDACTED]. From each transcript, generate the sentiment as either
negative, positive, or neutral.
```

### Prompt 7 — LLM as Judge (preference evaluation)

Each item is shown with the three candidate labels — one per method — in randomized order and without attribution to method.

```
You are a topic modeling expert. For each customer feedback item, you will:
1. Review the feedback text.
2. Review 3 candidate topic labels (each proposed by a different system).
3. Select the ONE label that best captures the MAIN theme of the feedback; however if
two labels are semantically similar but differ in magnitude, like Issue vs Issues,
write both separated by a ; for the main topic
4. If a second candidate label is also clearly relevant as a secondary theme, list it
in an array separated by ";"
5. If NONE of the candidate labels are a reasonable fit, output "NONE", but make sure
the three topics are absolutely not a reasonable fit, use the context for which the
feedback are based on to decide this.
6. Include a justification for your preference
```

## Contact

For questions or feedback, please contact acquayechristabel@gmail.com.


## Acknowledgments

The GPSR validation set is derived from the [Google Play Store Reviews corpus](https://www.kaggle.com/datasets/crawlfeeds/google-play-store-reviews) distributed on Kaggle. The telephony corpus was collected by Vail Systems.
