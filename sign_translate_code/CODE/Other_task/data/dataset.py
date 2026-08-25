# --- START OF FILE dataset.py ---

import pandas as pd
import re
import hanlp
# hanlp_common is usually implicitly available when hanlp is installed correctly
# from hanlp_common.document import Document # Might not be needed directly
import hanlp.pretrained
import numpy as np
from phrasetree.tree import Tree
import sys
import os



# with open(r"CKIP\標記列表.txt", "r",encoding='utf-8') as f:
#     pos_list = {i.split('\t')[0]:idx for idx,i in enumerate(f.read().splitlines())}
#     # print(pos_list)
pos_list =\
{'A': 0, 'Caa': 1, 'Cab': 2, 'Cba': 3, 'Cbb': 4, 'D': 5, 'Da': 6, 'Dfa': 7, 'Dfb': 8, 'Di': 9, 'Dk': 10, 'DM': 11,\
  'I': 12, 'Na': 13, 'Nb': 14, 'Nc': 15, 'Ncd': 16, 'Nd': 17, 'Nep': 18, 'Neqa': 19, 'Neqb': 20, 'Nes': 21, 'Neu': 22,\
      'Nf': 23, 'Ng': 24, 'Nh': 25, 'Nv': 26, 'P': 27, 'T': 28, 'VA': 29, 'VAC': 30, 'VB': 31, 'VC': 32, 'VCL': 33,\
          'VD': 34, 'VF': 35, 'VE': 36, 'VG': 37, 'VH': 38, 'VHC': 39, 'VI': 40, 'VJ': 41, 'VK': 42, 'VL': 43, 'V_2': 44,\
              'DE': 45, 'SHI': 46, 'FW': 47, 'COLONCATEGORY': 48, 'COMMACATEGORY': 49, 'DASHCATEGORY': 50, 'DOTCATEGORY': 51,\
                  'ETCCATEGORY': 52, 'EXCLAMATIONCATEGORY': 53, 'PARENTHESISCATEGORY': 54, 'PAUSECATEGORY': 55, 'PERIODCATEGORY': 56,\
                      'QUESTIONCATEGORY': 57, 'SEMICOLONCATEGORY': 58, 'SPCHANGECATEGORY': 59, 'WHITESPACE': 60}

# 讓 Python 能找到 CODE 這個目錄
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from CODE.preprocess import ckipnlptool
from CODE.nlptool import merge_pos_into_con

# label_map = {}
# label_index = 1
def get_node_depths(tree_str):
    """Calculates the depth of each node in a constituency tree string."""
    try:
        tree = Tree.fromstring(tree_str)
    except ValueError as e:
        print(f"Error parsing tree string: {tree_str}\nError: {e}")
        return {} # Return empty dict if parsing fails

    result = {}
    node_queue = [(tree, 0)] # Use a queue for BFS: (node, depth)
    node_counts = {} # To handle duplicate labels

    while node_queue:
        current_node, current_depth = node_queue.pop(0)
        
        if not isinstance(current_node, Tree): # Skip leaf nodes (strings)
            continue

        node_label = current_node.label()
        
        # if node_label not in label_map:
        #     global label_index
        #     label_map[node_label] = label_index
        #     label_index += 1

        # Handle duplicate labels by appending a counter
        count = node_counts.get(node_label, 0)
        unique_label = f"{node_label}_{count}" if count > 0 else node_label
        node_counts[node_label] = count + 1
        
        result[unique_label] = current_depth
        
        # Add children to the queue
        for child in current_node:
            if isinstance(child, Tree):
                node_queue.append((child, current_depth + 1))
                
    return result


def clean_text(item):
    if not isinstance(item, str):
        return item
    item = item.replace("(", "").replace(")", "")
    item = re.sub(r'\{.*?\}', '', item)
    item = re.sub(r'\[.*?\]', '', item)
    return item.strip()

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_excel(r"output2.xlsx")
    df = df[df['是否採用'] == 1].reset_index(drop=True)
    df = df[['口語', '手語']].rename(columns={'口語': 'input_text', '手語': 'target_text'})
    df['target_text'] = df['target_text'].apply(clean_text)
    print(f"Loaded {len(df)} samples.")

    print("Initializing CKIP tool...")
    ckiptool = ckipnlptool()
    print("Segmenting sentences with CKIP...")
    # Note: ckiptool.seg returns a list of lists (words for each sentence)
    seg_sentences = ckiptool.seg(df['input_text'].tolist())
    print("Segmentation done.")

    print("Loading HanLP models...")
    # Load HanLP components
    try:
        con = hanlp.load(hanlp.pretrained.constituency.CTB9_CON_FULL_TAG_ERNIE_GRAM)
        dep = hanlp.load(hanlp.pretrained.dep.CTB9_UDC_ELECTRA_SMALL, conll=False)
    except Exception as e:
        print(f"Error loading HanLP models: {e}")
        print("Ensure HanLP models are downloaded/accessible.")
        sys.exit(1)




    print("Setting up HanLP pipeline...")
    # Define HanLP pipeline - uses CKIP word segmentation ('tok') as input for POS tagger
    nlp = hanlp.pipeline() \
        .append(ckiptool.tagger, input_key='tok', output_key='pos') \
        .append(dep, input_key='tok', output_key='dep') \
        .append(con, input_key='tok', output_key='con') \
        .append(merge_pos_into_con, input_key='*')
    print("HanLP pipeline ready.")


    # Initialize columns in DataFrame
    df['ckip_tok'] = None
    df['pos'] = None
    df['pos_ids'] = None
    df['con'] = None
    df['tree_depth'] = None
    df['dep'] = None


    print(f"Processing {len(seg_sentences)} sentences...")
    # Process each sentence
    for idx, sentence_words in enumerate(seg_sentences):
        if idx % 50 == 0: # Print progress
             print(f"Processing sentence {idx+1}/{len(seg_sentences)}...")

        original_text = df.at[idx, 'input_text']


        # 2. HanLP Pipeline Processing (using CKIP segmented words)
        try:
            # The pipeline expects 'tok' key to be a list of words for a single sentence
            doc = nlp(tok=[sentence_words]) # Pass the CKIP segmented words

            # Store HanLP pipeline results
            df.at[idx, 'ckip_tok'] = doc['tok'][0] # Store the words used by the pipeline
            df.at[idx, 'pos'] = doc['pos'][0]
            df.at[idx, 'pos_ids'] = [pos_list.get(pos, -1) + 1 for pos in doc['pos'][0]] # Use get with default -1, +1 -> 0 for unknowns
            con_tree_str = str(doc['con'][0]).replace("\n", "")
            con_str = re.sub(r'\s+', ' ', con_tree_str)
            df.at[idx, 'con'] = con_str
            df.at[idx, 'tree_depth'] = get_node_depths(con_tree_str.replace(" ", "")) # Calculate depth from the string
            df.at[idx, 'dep'] = doc['dep'][0] # Assuming dep returns list of tuples [(token, head, label), ...]

        except Exception as e:
             print(f"Error during HanLP processing for index {idx}: {' '.join(sentence_words)}\nError: {e}")
             # Assign None or empty lists to columns in case of error
             df.at[idx, 'ckip_tok'] = sentence_words # Store original words if pipeline failed
             df.at[idx, 'pos'] = []
             df.at[idx, 'pos_ids'] = []
             df.at[idx, 'con'] = ""
             df.at[idx, 'tree_depth'] = {}
             df.at[idx, 'dep'] = []


    print("Processing finished.")

    # Define output paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, r'data.csv')
    output_sim_path = os.path.join(current_dir, r'data_sim.csv')

    print(f"Saving traditional Chinese data to {output_path}...")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print("Data saved.")

    # --- Simplified Chinese Conversion ---
    try:
        import opencc
        print("Converting to simplified Chinese...")
        cc = opencc.OpenCC('tw2sp') # Initialize converter once

        dfsim = df.copy()
        dfsim['input_text'] = dfsim['input_text'].apply(lambda x: cc.convert(x) if isinstance(x, str) else x)
        dfsim['target_text'] = dfsim['target_text'].apply(lambda x: cc.convert(x) if isinstance(x, str) else x)
        dfsim['con'] = dfsim['con'].apply(lambda x: cc.convert(x) if isinstance(x, str) else x)
        dfsim['ckip_tok'] = dfsim['ckip_tok'].apply(lambda lst: [cc.convert(item) for item in lst] if isinstance(lst, list) else lst)

        print(f"Saving simplified Chinese data to {output_sim_path}...")
        dfsim.to_csv(output_sim_path, index=False, encoding='utf-8-sig')
        print("Simplified data saved.")
    except ImportError:
        print("OpenCC not installed. Skipping simplified Chinese conversion.")
    except Exception as e:
        print(f"Error during simplified Chinese conversion: {e}")

    # try:
    #     # Save label map to a CSV file
    #     label_map_df = pd.DataFrame(list(label_map.items()), columns=['Label', 'Index'])
    #     label_map_path = os.path.join(current_dir, 'label_map.csv')
    #     label_map_df.to_csv(label_map_path, index=False, encoding='utf-8-sig')
    #     print(f"Label map saved to {label_map_path}.")
    # except Exception as e:
    #     print(f"Error saving label map: {e}")

    print("Script finished.")
# --- END OF FILE dataset.py ---