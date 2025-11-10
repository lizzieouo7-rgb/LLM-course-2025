import sentencepiece as spm
import os

text_content = text_content = """
Hello world.
Neural Machine Translation of Rare Words with Subword Units
SentencePiece: A simple and language independent subword tokenizer
こんにちは世界。
Abwasserbehandlungsanlage
My name is Sennrich.
"""
with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write(text_content)

print("--- 训练开始 ---")

# --- 2. 训练 BPE 模型 ---
spm.SentencePieceTrainer.Train(
    input='corpus.txt', 
    model_prefix='bpe_model', 
    vocab_size=50, 
    model_type='bpe' 
)
print("BPE 模型训练完毕。")

# --- 3. 训练 Unigram 模型 (SentencePiece 的另一个选项) ---
spm.SentencePieceTrainer.Train(
    input='corpus.txt', 
    model_prefix='unigram_model', 
    vocab_size=50, 
    model_type='unigram' 
)
print("Unigram 模型训练完毕。")

print("\n--- 加载模型并开始分词 ---")

# --- 4. 加载训练好的模型 ---
sp_bpe = spm.SentencePieceProcessor(model_file='bpe_model.model')
sp_unigram = spm.SentencePieceProcessor(model_file='unigram_model.model')

# --- 5. 定义要测试的句子 ---
sentences = [
    "Hello world.",
    "This is a new sentence with Sennrich and Kudo.", # 测试 OOV 名字
    "Abwasserbehandlungsanlage", # 测试复合词 
    "こんにちは世界。" # 测试非拉丁、无空格语言 [cite: 449]
]

# --- 6. 运行并比较 ---
for sentence in sentences:
    print(f"\n>>> 原始句子: {sentence}")
    
    # BPE
    bpe_tokens = sp_bpe.encode(sentence, out_type=str)
    print(f"  BPE 分词: {bpe_tokens}")
    
    # Unigram
    unigram_tokens = sp_unigram.encode(sentence, out_type=str)
    print(f"  Unigram 分词: {unigram_tokens}")

    # 演示可逆性 (Lossless Tokenization) [cite: 491]
    bpe_detokenized = sp_bpe.decode(bpe_tokens)
    print(f"  BPE 复原: {bpe_detokenized}")
    
    # 检查复原是否完美
    assert sentence == bpe_detokenized