import json
import random
from datasets import load_dataset
from tqdm import tqdm

# --- Settings ---
TRAIN_SIZE = 4500   # Total training examples
VAL_SIZE = 500      # How many examples for validation during training
TEST_SIZE = 500     # How many examples to benchmark on
MIN_EXP_LEN = 100   # Ensure we only use questions with good explanations

# MMLU Mixing Settings
MMLU_RATIO = 0.20   # 20% MMLU, 80% MedMCQA in training data
MEDMCQA_TRAIN = int(TRAIN_SIZE * (1 - MMLU_RATIO))  # 3600
MMLU_TRAIN = int(TRAIN_SIZE * MMLU_RATIO)            # 900
MEDMCQA_VAL = int(VAL_SIZE * (1 - MMLU_RATIO))      # 400
MMLU_VAL = int(VAL_SIZE * MMLU_RATIO)                # 100

print(f"Dataset Mix: {MEDMCQA_TRAIN} MedMCQA + {MMLU_TRAIN} MMLU = {TRAIN_SIZE} training examples")
print(f"Validation: {MEDMCQA_VAL} MedMCQA + {MMLU_VAL} MMLU = {VAL_SIZE} validation examples")

def format_medmcqa(example):
    """
    Format MedMCQA: Question -> Analysis (Reasoning) -> Answer
    """
    # Filter junk
    if not example['exp'] or len(example['exp']) < MIN_EXP_LEN:
        return None

    # Map index 0-3 to A-D
    mapper = ['A', 'B', 'C', 'D']
    try:
        answer_key = mapper[example['cop']]
    except:
        return None

    question = f"Question: {example['question']}\n\nOptions:\n"
    question += f"A) {example['opa']}\n"
    question += f"B) {example['opb']}\n"
    question += f"C) {example['opc']}\n"
    question += f"D) {example['opd']}\n\n"
    question += "Provide your step-by-step analysis, then state your answer as 'Answer: X' where X is the option letter."

    # The chain-of-thought response
    response = f"Analysis: {example['exp']}\n\nAnswer: {answer_key}"

    return {
        "messages": [
            {"role": "system", "content": "You are a medical expert. Analyze medical questions step-by-step and select the best option."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ],
        "ground_truth": answer_key,
        "prompt": question,
        "source": "medmcqa"
    }

def format_mmlu(example):
    """
    Format MMLU: Question -> Concise Answer
    MMLU doesn't have explanations, so we use shorter responses
    """
    # Map index 0-3 to A-D
    mapper = ['A', 'B', 'C', 'D']
    try:
        answer_key = mapper[example['answer']]
        choices = example['choices']
    except:
        return None

    # Ensure we have 4 choices
    if len(choices) != 4:
        return None

    question = f"Question: {example['question']}\n\nOptions:\n"
    question += f"A) {choices[0]}\n"
    question += f"B) {choices[1]}\n"
    question += f"C) {choices[2]}\n"
    question += f"D) {choices[3]}\n\n"
    question += "Provide your step-by-step analysis, then state your answer as 'Answer: X' where X is the option letter."

    # Since MMLU has no explanations, use concise response
    # This teaches the model to be flexible with response length
    response = f"Answer: {answer_key}"

    return {
        "messages": [
            {"role": "system", "content": "You are a medical expert. Analyze medical questions step-by-step and select the best option."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ],
        "ground_truth": answer_key,
        "prompt": question,
        "source": "mmlu"
    }

def main():
    # ===== Load MedMCQA =====
    print("\n>>> Loading MedMCQA...")
    medmcqa_dataset = load_dataset(
        "/users/ddixit/scratch/apertus-project/huggingface_cache/datasets--openlifescienceai--medmcqa/snapshots/91c6572c454088bf71b679ad90aa8dffcd0d5868",
        split="train"
    )
    medmcqa_dataset = medmcqa_dataset.shuffle(seed=42)

    medmcqa_clean = []
    print(">>> Filtering MedMCQA data...")
    # Need extra for train + val + test
    needed_medmcqa = MEDMCQA_TRAIN + MEDMCQA_VAL + TEST_SIZE
    for ex in tqdm(medmcqa_dataset, desc="MedMCQA"):
        formatted = format_medmcqa(ex)
        if formatted:
            medmcqa_clean.append(formatted)
        if len(medmcqa_clean) >= needed_medmcqa:
            break

    print(f"✓ Collected {len(medmcqa_clean)} MedMCQA examples")

    # ===== Load MMLU Medical =====
    print("\n>>> Loading MMLU Medical Tasks...")
    mmlu_base_path = "/users/ddixit/scratch/apertus-project/huggingface_cache/datasets--cais--mmlu/snapshots/c30699e8356da336a370243923dbaf21066bb9fe"

    mmlu_tasks = [
        "clinical_knowledge",
        "professional_medicine",
        "anatomy",
        "medical_genetics",
        "college_medicine",
        "college_biology"
    ]

    mmlu_clean = []
    for task in mmlu_tasks:
        print(f"  Loading {task}...")
        try:
            task_dataset = load_dataset(mmlu_base_path, task, split="test", trust_remote_code=True)
            for ex in task_dataset:
                formatted = format_mmlu(ex)
                if formatted:
                    mmlu_clean.append(formatted)
        except Exception as e:
            print(f"Failed to load {task}: {e}")

    print(f"✓ Collected {len(mmlu_clean)} MMLU examples from {len(mmlu_tasks)} tasks")

    # Shuffle MMLU data
    random.seed(42)
    random.shuffle(mmlu_clean)

    # ===== Create Mixed Datasets =====
    print("\n>>> Creating mixed datasets...")

    # Training set: Mix MedMCQA and MMLU
    train_medmcqa = medmcqa_clean[:MEDMCQA_TRAIN]
    train_mmlu = mmlu_clean[:MMLU_TRAIN] if len(mmlu_clean) >= MMLU_TRAIN else mmlu_clean
    train_data = train_medmcqa + train_mmlu
    random.shuffle(train_data)  # Shuffle the mix

    # Validation set: Mix MedMCQA and MMLU
    val_medmcqa = medmcqa_clean[MEDMCQA_TRAIN:MEDMCQA_TRAIN + MEDMCQA_VAL]
    val_mmlu = mmlu_clean[MMLU_TRAIN:MMLU_TRAIN + MMLU_VAL] if len(mmlu_clean) >= (MMLU_TRAIN + MMLU_VAL) else []
    val_data = val_medmcqa + val_mmlu
    random.shuffle(val_data)  # Shuffle the mix

    # Test set: Only MedMCQA (to measure performance on this specific task)
    test_data = medmcqa_clean[MEDMCQA_TRAIN + MEDMCQA_VAL:MEDMCQA_TRAIN + MEDMCQA_VAL + TEST_SIZE]

    print(f"✓ Training set: {len(train_data)} examples ({len(train_medmcqa)} MedMCQA + {len(train_mmlu)} MMLU)")
    print(f"✓ Validation set: {len(val_data)} examples ({len(val_medmcqa)} MedMCQA + {len(val_mmlu)} MMLU)")
    print(f"✓ Test set: {len(test_data)} examples (MedMCQA only)")

    # ===== Save Datasets =====
    print(f"\n>>> Saving {len(train_data)} examples for Training...")
    with open("my_train.jsonl", "w") as f:
        for entry in train_data:
            # For training, we only need the 'messages'
            json.dump({"messages": entry["messages"]}, f)
            f.write("\n")

    print(f">>> Saving {len(val_data)} examples for Validation...")
    with open("my_val.jsonl", "w") as f:
        for entry in val_data:
            # For validation, we only need the 'messages'
            json.dump({"messages": entry["messages"]}, f)
            f.write("\n")

    print(f">>> Saving {len(test_data)} examples for Final Benchmarking...")
    with open("my_test.jsonl", "w") as f:
        for entry in test_data:
            # For testing, we keep the raw prompt and answer key
            json.dump(entry, f)
            f.write("\n")

    # ===== Summary =====
    print("\n" + "="*60)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*60)
    print(f"Files created:")
    print(f"  • my_train.jsonl   : {len(train_data)} examples (mixed)")
    print(f"  • my_val.jsonl     : {len(val_data)} examples (mixed)")
    print(f"  • my_test.jsonl    : {len(test_data)} examples (MedMCQA only)")
    print("\nDataset composition:")
    print(f"  Training: {len([x for x in train_data if x.get('source') == 'medmcqa'])} MedMCQA + {len([x for x in train_data if x.get('source') == 'mmlu'])} MMLU")
    print(f"  Validation: {len([x for x in val_data if x.get('source') == 'medmcqa'])} MedMCQA + {len([x for x in val_data if x.get('source') == 'mmlu'])} MMLU")
    print("\n💡 Benefits of mixed training:")
    print(f"  • Reduces catastrophic forgetting on MMLU tasks")
    print(f"  • Model learns both detailed (MedMCQA) and concise (MMLU) responses")
    print(f"  • Better generalization across medical question formats")
    print("="*60)

if __name__ == "__main__":
    main()