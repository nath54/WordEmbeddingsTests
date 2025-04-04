#
from transformers import pipeline, FeatureExtractionPipeline  # type: ignore
import numpy as np
from tqdm import tqdm  # type: ignore
import json
import os
import re

#
model: str = "answerdotai/ModernBERT-base"
model = "sentence-transformers/all-MiniLM-L6-v2"
task: str = "feature-extraction"
#
pipe: FeatureExtractionPipeline = pipeline(model=model, task=task)
#
voc_file: str = "voc_5000.txt"
embedding_file: str = "word_embeddings_modern_bert.json"
embedding_file = "word_embeddings.json"
#
w_emb: dict[str, np.ndarray] = {}
#
nb_to_save: int = 0

#

def cosine_similarity_np(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Calculates cosine similarity between two NumPy vectors.

    Args:
        vector1: The first NumPy vector.
        vector2: The second NumPy vector.

    Returns:
        The cosine similarity (a float between -1 and 1).
    """

    # Check if the vectors are 1D arrays
    if vector1.ndim != 1 or vector2.ndim != 1:
        raise ValueError("Input vectors must be 1D NumPy arrays.")

    # Check if the vectors have the same dimension
    if vector1.shape != vector2.shape:
        raise ValueError("Input vectors must have the same shape.")

    # Avoid division by zero
    norm_v1: float = np.linalg.norm(vector1).item()
    norm_v2: float = np.linalg.norm(vector2).item()

    if norm_v1 == 0 or norm_v2 == 0:
      return 0 # Or handle it differently, like raising an exception

    similarity: float = np.dot(vector1, vector2) / (norm_v1 * norm_v2)

    return similarity


def cosine_distance_np(vector1: np.ndarray, vector2: np.ndarray) -> float:
  """Calculates cosine distance between two NumPy vectors.

    Args:
        vector1: The first NumPy vector.
        vector2: The second NumPy vector.

    Returns:
        The cosine distance (a float between 0 and 2).
    """
  return 1 - cosine_similarity_np(vector1, vector2) # Cosine distance is 1 - Cosine Similarity


#
def calculate_embeddings() -> None:
    global w_emb, nb_to_save
    #
    with open(voc_file, "r", encoding="utf-8") as f:
        words: list[str] = [w for w in f.read().split("\n") if len(w) > 2]
    #
    for w in tqdm(words):
        #
        if w not in w_emb:
            v: np.ndarray = np.array(pipe(w))
            v = np.mean(v, axis=1)
            v = np.squeeze(v)
            w_emb[w] = v

#
def save_embedding(save_filepath: str = embedding_file) -> None:
    global w_emb, nb_to_save
    #
    save_dict: dict[str, list] = {}
    #
    for w in tqdm(w_emb):
        save_dict[w] = list(w_emb[w].tolist())
    #
    with open(save_filepath, "w", encoding="utf-8") as f:
        json.dump(save_dict, f)
    #
    nb_to_save = 0

#
def load_embeddings(filepath: str = embedding_file) -> None:
    global w_emb, nb_to_save
    #
    with open(filepath, "r", encoding="utf-8") as f:
        #
        load_dict: dict[str, list] = json.load(f)
    #
    for w in tqdm(load_dict):
        w_emb[w] = np.array(load_dict[w])


#
def get_closer_word_of_embedding(e: np.ndarray, exclude_words: list[str] = []) -> tuple[str, float]:
    global w_emb, nb_to_save
    #
    closer: str = ""
    closer_dist: float = -1
    dist: float
    #
    for w in tqdm(w_emb):
        #
        if w in exclude_words:
            continue
        #
        # dist = np.sqrt(np.sum((w_emb[w] - e) ** 2))  # Euler distance
        dist = cosine_distance_np(e, w_emb[w])
        #
        if closer == "" or dist < closer_dist:
            #
            closer = w
            closer_dist = dist
    #
    return closer, closer_dist

#
def get_word_embedding(word: str) -> np.ndarray:
    global w_emb, nb_to_save
    #
    if word not in w_emb:
        #
        v: np.ndarray = np.array(pipe(word))
        v = np.mean(v, axis=1)
        v = np.squeeze(v)
        w_emb[word] = v
        #
        nb_to_save += 1
    #
    return w_emb[word]


#

def split_with_separators(text: str, separators: list[str]) -> list[str]:
    """
    Splits a text with multiple separators and includes the separators in the result.

    Args:
        text: The input text string.
        separators: A list of separator strings.

    Returns:
        A list of strings, containing the split text and the separators.
    """
    # Escape the separators for use in a regular expression.
    escaped_separators: list[str] = [re.escape(sep) for sep in separators]

    # Construct the regex pattern.
    pattern: str = r"|".join(f"({sep})" for sep in escaped_separators)

    # Split the text and include the separators.
    result: list[str] = []
    parts = re.split(f"({pattern})", text)

    #
    for part in parts:
        if part:
            #
            npart: str = part.strip()
            #
            if result and (result[-1] in escaped_separators or result[-1] in separators) and (npart in escaped_separators or npart in separators):
                #
                continue
            #
            result.append(npart)

    #
    return result


#
if __name__ == "__main__":
    #
    if not os.path.exists(embedding_file):
        #
        calculate_embeddings()
        #
        save_embedding()
    #
    else:
        #
        load_embeddings()
        #
        calculate_embeddings()
        #
        save_embedding()
    #
    operation: str = ""
    worda: str = ""
    wordb: str = ""
    #
    ea: np.ndarray
    eb: np.ndarray
    #
    e: np.ndarray
    #
    state: int = 0
    #
    while True:
        #
        if state == 0:
            #
            print(f"\nnb_to_save = {nb_to_save}")
            #
            operation = input("\nEnter operation (+, -, closest, sequence, save, quit) : ")
            #
            if operation in ["+", "-", "closest", "sequence"]:
                state = 1
            #
            elif operation == "save":
                save_embedding()
            #
            elif operation == "quit":
                #
                if nb_to_save > 0:
                    save_embedding()
                #
                exit()
            #
            else:
                print("Error: unknown operation")
        #
        elif state == 1:
            #
            if operation in ["+", "-"]:
                #
                worda = input("Word 1 : ").lower()
                wordb = input("Word 2 : ").lower()
                #
                ea = get_word_embedding(worda)
                eb = get_word_embedding(wordb)
                #
                if operation == "+":
                    #
                    e = (ea + eb) / 2
                #
                elif operation == "-":
                    e = (ea - eb) / 2
                #
                print(f" `{worda}` {operation} `{wordb}` = `{get_closer_word_of_embedding(e, exclude_words=[worda, wordb])}`")
            #
            elif operation == "closest":
                #
                worda = input("Word : ").lower()
                ea = get_word_embedding(worda)
                #
                exclusion: list[str] = [worda]
                closests: list[tuple[str, float]] = []
                #
                for i in range(10):
                    closests.append( get_closer_word_of_embedding(ea, exclude_words=exclusion + [c[0] for c in closests]) )
                #
                print(f" closest of `{worda}` are `{"` `".join([str(c) for c in closests])}`")
            #
            elif operation == "sequence":
                #
                operators = ["+", "-"]
                #
                sequence_input: str = input("Sequence : ")
                #
                splited = split_with_separators(sequence_input.strip(), operators)
                #
                print(f"splited : {splited}")
                #
                if len(splited) == 0:
                    #
                    print(f"Empty sequence error")
                    state = 0
                    continue
                #
                current_embedding: np.ndarray = get_word_embedding(splited[0])
                #
                nb: int = len(splited)
                #
                error: bool = False
                #
                i = 1
                while i < nb:
                    #
                    if nb - i >= 2:
                        #
                        operator = splited[i].strip()
                        #
                        if operator not in operators:
                            #
                            print(f"Syntax error: `{operator}` not in operators")
                            error = True
                            break
                        #
                        eb = get_word_embedding(splited[i+1].strip())
                        #
                        if operator == "+":
                            #
                            current_embedding = (current_embedding + eb) / 2
                        #
                        elif operator == "-":
                            current_embedding = (current_embedding - eb) / 2
                        #
                        i += 2
                    #
                    else:
                        #
                        print(f"Sequence length error: not correct number of elements")
                        error = True
                        break

                #
                if error:
                    #
                    state = 0
                    continue

                #
                exclusion = []
                closests = []
                #
                for i in range(10):
                    closests.append( get_closer_word_of_embedding(current_embedding, exclude_words=exclusion + [c[0] for c in closests]) )
                #
                print(f"Closests of `{splited}` are :\n -`{"`\n -`".join([str(c) for c in closests])}`")

            #
            state = 0
