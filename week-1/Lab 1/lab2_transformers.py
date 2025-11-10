from transformers import BertTokenizer, GPT2Tokenizer

# --- 1. 加载一个预训练的 BERT 分词器 ---
# BERT 使用一种叫做 WordPiece 的算法，它和 BPE 非常相似。
print("--- BERT (WordPiece) Tokenizer ---")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# --- 2. 加载一个预训练的 GPT-2 分词器 ---
# GPT-2 使用的正是 BPE！ [cite: 33]
print("\n--- GPT-2 (BPE) Tokenizer ---")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# --- 3. 定义要测试的句子 ---
sentences = [
    "Hello world.",
    "This is a new sentence with Sennrich and Kudo.",
    "Abwasserbehandlungsanlage", 
    "人对爱和永远应该有幻觉。" 
]

# --- 4. 运行并比较 ---
for sentence in sentences:
    print(f"\n>>> 原始句子: {sentence}")
    
    # BERT
    bert_tokens = bert_tokenizer.tokenize(sentence)
    print(f"  BERT 分词: {bert_tokens}")
    
    # GPT-2 (BPE)
    gpt2_tokens = gpt2_tokenizer.tokenize(sentence)
    print(f"  GPT-2 分词: {gpt2_tokens}")