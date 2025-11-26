"""
InsightStack Prompts
====================
Complete set of prompts used in the paper:
"LLM-Assisted Topic Modeling for Customer Feedback Analysis"

Submitted to PAKDD 2025
"""

# ==============================================================================
# STAGE 1: DATA PREPARATION
# ==============================================================================

SYNTHESIZED_REWRITE_GENERATION = """
You are an expert specializing in analyzing a large number of call transcripts, where each transcript is the customer feedback responding to the survey prompt [REDUCTED]. You will be asked to generate a summary which is a relevant summarized version of the feedback that maintains the context of the feedback. Prefix your final answer with Rewrite: [text].
"""

# ==============================================================================
# STAGE 2: TOPIC MODELING - PARALLEL APPROACH
# ==============================================================================

PARALLEL_TOPIC_GENERATION = """
You are an expert in topic modeling for some customer feedback responding to the survey prompt [REDUCTED]. You will be tasked with performing topic modeling on the following documents, summarizing and identifying key topics from these customer feedback. The output should be a numbered list of concise topic labels that represent the main themes in the documents. You should make a good effort to reduce the number of topics you started with for each batch.
"""

PARALLEL_TOPIC_MERGE = """
You are an expert in topic modeling and data analysis. Your task is to write the results of merging the following topic modeling results. The final output should be a consolidated and refined list of topics. Avoid over-generalization. The final topics should be distinct and clearly represent related themes from the input list.
"""

# ==============================================================================
# STAGE 2: TOPIC MODELING - SEQUENTIAL APPROACH
# ==============================================================================

SEQUENTIAL_TOPIC_REFINEMENT = """
You are an expert in online, incremental topic modeling for some customer feedback responding to the survey prompt [REDUCTED]. You can also be provided: (1) a current topic dictionary with names, short descriptions, and exemplar quotes; and (2) a new batch of feedback with IDs. For each customer feedback, decide whether it clearly fits an existing topic. If so, return the existing topic exactly as written. If not, propose a new topic and one-sentence description for that topic. Merge existing topics if similar themes are detected, updating names/descriptions if needed.
"""

# ==============================================================================
# STAGE 2: TOPIC MODELING - CLUSTERING APPROACH
# ==============================================================================

CONTEXTUALIZED_EMBEDDING_LABELING = """
You are an expert specialized in analyzing a large number of call transcripts, where customers provided feedback responding to the survey prompt asking about [REDUCTED]. The customers were prompted with the question: [REDUCTED]
"""

# ==============================================================================
# STAGE 3: TOPIC ASSIGNMENT
# ==============================================================================

TOPIC_ASSIGNMENT = """
You are a topic-modeling expert tasked with assigning topic labels to individual customer feedback. For each complaint, your task is to select the best-fitting topic label from a provided candidate list. Labels must be chosen exactly as written; no new labels may be invented. If no candidate is appropriate, return "NONE,".
"""

# ==============================================================================
# STAGE 4: SENTIMENT ANALYSIS
# ==============================================================================

SENTIMENT_LABELING = """
You are an expert specialized in analyzing for some customer feedback responding to the survey prompt [REDUCTED]. From each transcript, generate the sentiment as either negative, positive, or neutral.
"""

# ==============================================================================
# EVALUATION: LLM AS JUDGE
# ==============================================================================

LLM_AS_JUDGE = """
You are a topic modeling expert. For each customer feedback item, you will: 
1. Review the feedback text. 
2. Review 3 candidate topic labels (each proposed by a different system). 
3. Select the ONE label that best captures the MAIN theme of the feedback; however if two labels are semantically similar but differ in magnitude, like Issue vs Issues, write both separated by a ; for the main topic 
4. If a second candidate label is also clearly relevant as a secondary theme, list it in an array separated by ";" 
5. If NONE of the candidate labels are a reasonable fit, output "NONE", but make sure the three topics are absolutely not a reasonable fit, use the context for which the feedback are based on to decide this. 
6. Include a justification for your preference
"""

# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

def example_usage():
    """
    Example of how these prompts were used in the paper's experiments.
    """
    
    # Example 1: Generating synthesized rewrites
    original_feedback = "okay"
    # Prompt: SYNTHESIZED_REWRITE_GENERATION + original_feedback
    # Expected output: "Rewrite: Customer found the product helpful."
    
    # Example 2: Sequential topic refinement
    existing_topics = ["Helpful Service", "Product Issues"]
    new_batch = ["The service was great", "Product didn't work"]
    # Prompt: SEQUENTIAL_TOPIC_REFINEMENT + existing_topics + new_batch
    # Expected output: Updated topic list with assignments
    
    # Example 3: Topic assignment
    feedback = "The representative was very helpful"
    candidate_labels = ["Helpful Service", "Product Issues", "Billing Problems"]
    # Prompt: TOPIC_ASSIGNMENT + feedback + candidate_labels
    # Expected output: "Helpful Service"
    
    pass


# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================

MODEL_CONFIGS = {
    "topic_generation": {
        "model": "DeepSeek-R1-Distill-Llama-70B",
        "temperature": 0.7,
        "max_tokens": 4096
    },
    "embedding": {
        "model": "Qwen3-Embedding-0.6B",
        "context_aware": True
    },
    "clustering": {
        "algorithm": "HDBSCAN",
        "min_cluster_size": 15,
        "min_samples": 5,
        "metric": "euclidean"
    }
}


# ==============================================================================
# PROMPT TEMPLATES WITH VARIABLES
# ==============================================================================

def get_sequential_refinement_prompt(current_topics, new_batch, survey_question):
    """
    Generate the sequential refinement prompt with actual data.
    
    Args:
        current_topics (list): List of current topic dictionaries
        new_batch (list): List of new feedback items
        survey_question (str): The survey question asked to customers
        
    Returns:
        str: Complete prompt ready for LLM
    """
    prompt = f"""
You are an expert in online, incremental topic modeling for customer feedback responding to the survey prompt: "{survey_question}"

Current Topics:
{format_topics(current_topics)}

New Batch of Feedback:
{format_batch(new_batch)}

For each feedback, decide whether it clearly fits an existing topic. If so, return the existing topic exactly as written. If not, propose a new topic and one-sentence description. Merge existing topics if similar themes are detected.
"""
    return prompt


def format_topics(topics):
    """Format topics for display in prompt"""
    return "\n".join([f"{i+1}. {t['name']}: {t['description']}" 
                      for i, t in enumerate(topics)])


def format_batch(batch):
    """Format feedback batch for display in prompt"""
    return "\n".join([f"ID {i+1}: {fb}" for i, fb in enumerate(batch)])


# ==============================================================================
# NOTES
# ==============================================================================

"""
IMPORTANT NOTES:

1. Survey Question Context:
   The actual survey question has been redacted as [REDUCTED] in this public version.
   In the paper experiments, this was replaced with the actual question about 
   product/service helpfulness.

2. Batch Sizes:
   - Sequential refinement: 100-200 feedback items per batch
   - Parallel processing: 50-150 feedback items per batch
   - K-means initial batch: 100 representative samples

3. Prompt Engineering Details:
   - All prompts include few-shot examples in actual implementation
   - Output format specifications were added for structured responses
   - Temperature settings (0.3 for assignment and generation)

4. Reproducibility:
   - Random seed set to 42 for all experiments
   - Same prompts used across all three input types (RES, REW, COMB)

5. For full experimental setup, see paper Section 5.
"""