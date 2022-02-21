import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 転移学習済みモデル
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# トークナイザー（SentencePiece）
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR, is_fast=True)

# 学習済みモデル
trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

# GPUの利用有無
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    trained_model.cuda()

# MAX_SOURCE_LENGTH = args.max_input_length   # 入力される記事本文の最大トークン数
# MAX_TARGET_LENGTH = args.max_target_length  # 生成されるタイトルの最大トークン数
MAX_SOURCE_LENGTH = 64   # 入力される記事本文の最大トークン数
MAX_TARGET_LENGTH = 64  # 生成されるタイトルの最大トークン数


def normalize_text(text):
    assert "\n" not in text and "\r" not in text
    text = text.replace("\t", " ")
    text = text.strip()
    # text = normalize_neologd(text)
    text = text.lower()
    return text


def preprocess(text):
    return normalize_text(text.replace("\n", ""))


def generate(text):
    # 推論モード設定
    trained_model.eval()

    # 前処理とトークナイズを行う
    inputs = [preprocess(text)]
    batch = tokenizer.batch_encode_plus(
        inputs, max_length=MAX_SOURCE_LENGTH, truncation=True,
        padding="longest", return_tensors="pt")

    input_ids = batch['input_ids']
    input_mask = batch['attention_mask']
    if USE_GPU:
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()

    # 生成処理を行う
    outputs = trained_model.generate(
        input_ids=input_ids, attention_mask=input_mask,
        max_length=MAX_TARGET_LENGTH,
        # temperature=1.0,  # 生成にランダム性を入れる温度パラメータ
        # num_beams=10,  # ビームサーチの探索幅
        # diversity_penalty=1.0,  # 生成結果の多様性を生み出すためのペナルティパラメータ
        # num_beam_groups=10,  # ビームサーチのグループ
        # num_return_sequences=10,  # 生成する文の数
        repetition_penalty=8.0,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
    )

    # 生成されたトークン列を文字列に変換する
    generated_replies = [tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for ids in outputs]

    # 生成された文章を返す
    return "\n".join(generated_replies)
