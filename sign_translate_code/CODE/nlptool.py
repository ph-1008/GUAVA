import hanlp
from hanlp_common.document import Document


def tree_depth(tree):
    if not isinstance(tree, list) or not tree:
        return 0
    return 1 + max((tree_depth(subtree) for subtree in tree), default=0)

def get_tree_depth(tree):
    return [tree_depth(tree)]

# Function to merge POS tags into constituency trees
def merge_pos_into_con(doc: Document):
    flat = isinstance(doc['pos'][0], str)
    if flat:
        doc = Document((k, [v]) for k, v in doc.items())
    
    for tree, tags in zip(doc['con'], doc['pos']):
        offset = 0
        for subtree in tree.subtrees(lambda t: t.height() == 2):
            tag = subtree.label()
            if tag == '_':
                subtree.set_label(tags[offset])
            offset += 1
        print(offset)
    
    if flat:
        doc = doc.squeeze()
    return doc


if __name__ == '__main__':
    # Load HanLP pretrained constituency parser
    from preprocess import ckipnlptool
    con = hanlp.load(hanlp.pretrained.constituency.CTB9_CON_FULL_TAG_ERNIE_GRAM)

    # Initialize CKIP tool
    ckiptool = ckipnlptool()

    # Input sentences
    sentence = ['你是聾人嗎?', '我喜歡健康的生活和運動', '我吃了媽媽買的蘋果']

    # Segment sentences using CKIP tool
    seg_sentence = ckiptool.seg(sentence)

    # Build NLP pipeline
    nlp = hanlp.pipeline() \
        .append(ckiptool.tagger, input_key='tok', output_key='pos') \
        .append(con, input_key='tok', output_key='con') \
        .append(merge_pos_into_con, input_key='*')

    # Process each segmented sentence and print results
    for s in seg_sentence:
        doc = nlp(tok=[s])
        doc.pretty_print()
        print('-' * 50)
