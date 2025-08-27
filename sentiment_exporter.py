import os
import sys
import sqlite3
import argparse
from typing import List, Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

DB_PATH = os.getenv("SENTIMENT_DB_PATH", "./market_data.db")
DEFAULT_MODEL = os.getenv("SENTIMENT_MODEL", "koheiduck/bert-japanese-finetuned-sentiment")

# スコア対象フィールド例（既存DBに合わせて適宜変更可）
TARGET_FIELDS = [
    "company_overview",
    "performance",
    "topics_title",
    "topics_body",
    "risk_title",
    "risk_body",
    "investment_view",
    "shikiho_gaiyo",
    "top_holders",
    "executives",
    "analyst_comment",
    "rating_comment",
]

def connect_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA busy_timeout=8000;")
    conn.commit()
    return conn

def ensure_tables(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS sbi_sentiment_wide (
            target_date   TEXT,
            code          TEXT,
            summary       TEXT,
            {", ".join(f"{f} REAL" for f in TARGET_FIELDS)},
            model_name    TEXT,
            PRIMARY KEY (target_date, code, model_name)
        )
    """)
    conn.commit()
    cur.execute("PRAGMA table_info(sbi_sentiment_wide)")
    existing = {row[1] for row in cur.fetchall()}
    required = ["target_date", "code", "summary"] + TARGET_FIELDS + ["model_name"]
    to_add = [c for c in required if c not in existing]
    for col in to_add:
        coltype = "TEXT" if col in ("target_date","code","summary","model_name") else "REAL"
        cur.execute(f"ALTER TABLE sbi_sentiment_wide ADD COLUMN {col} {coltype}")
    if to_add:
        conn.commit()

def load_texts(conn: sqlite3.Connection, target_date: str, code: str) -> List[Tuple[str, str]]:
    cur = conn.cursor()
    cols = ", ".join(TARGET_FIELDS)
    cur.execute(f"SELECT {cols} FROM sbi_text_reports WHERE target_date=? AND code=? LIMIT 1",(target_date,code))
    row = cur.fetchone()
    if not row: return []
    return [(field,(row[i] or "").strip()) for i,field in enumerate(TARGET_FIELDS) if row[i]]

def choose_device(arg_device: str) -> str:
    if arg_device=="cpu": return "cpu"
    if arg_device=="mps": return "mps" if torch.backends.mps.is_available() else "cpu"
    if arg_device=="cuda": return "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def build_pipeline(model_name: str, device_str: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    if device_str=="cuda": mdl.to("cuda"); device_idx=0
    elif device_str=="mps": mdl.to("mps"); device_idx=0
    else: device_idx=-1
    return pipeline("sentiment-analysis", model=mdl, tokenizer=tok, device=device_idx, top_k=None, truncation=True)

def to_posneg_score(probs: List[Dict[str, Any]]) -> float:
    p_pos=p_neg=0.0
    for d in probs:
        lab=d.get("label","").lower(); sc=float(d.get("score",0.0))
        if "pos" in lab: p_pos=sc
        elif "neg" in lab: p_neg=sc
    return p_pos-p_neg

def chunk_text(text: str, max_chars:int=1200) -> List[str]:
    text=text.strip()
    return [text[i:i+max_chars] for i in range(0,len(text),max_chars)] if len(text)>max_chars else [text]

def gen_summary(row: Dict[str,float]) -> str:
    scores=[v for v in row.values() if isinstance(v,(int,float)) and v is not None]
    if not scores: return "スコアなし"
    avg=sum(scores)/len(scores)
    if avg>0.5: return "総合的にポジティブ"
    if avg>0.2: return "ややポジティブ"
    if avg<-0.5: return "総合的にネガティブ"
    if avg<-0.2: return "ややネガティブ"
    return "ニュートラル"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("-a","--target_date",required=True)
    ap.add_argument("--code",required=True)
    ap.add_argument("--model",default=DEFAULT_MODEL)
    ap.add_argument("--device",default="auto",choices=["auto","cpu","cuda","mps"])
    args=ap.parse_args()

    conn=connect_db(DB_PATH)
    ensure_tables(conn)
    cur=conn.cursor()

    items=load_texts(conn,args.target_date,args.code)
    if not items:
        print("⚠️ データなし"); conn.close(); sys.exit(0)

    device_str=choose_device(args.device)
    nlp=build_pipeline(args.model,device_str)

    result_row={f:None for f in TARGET_FIELDS}
    for field,text in items:
        chunks=chunk_text(text)
        outputs=nlp(chunks)
        scores=[to_posneg_score(out) for out in outputs]
        result_row[field]=sum(scores)/max(len(scores),1)

    summary=gen_summary(result_row)

    placeholders=",".join("?" for _ in range(3+len(TARGET_FIELDS)+1))
    sql=f"INSERT OR REPLACE INTO sbi_sentiment_wide (target_date,code,summary,{','.join(TARGET_FIELDS)},model_name) VALUES ({placeholders})"
    values=[args.target_date,args.code,summary]+[result_row[f] for f in TARGET_FIELDS]+[args.model]
    cur.execute(sql,values); conn.commit(); conn.close()
    print(f"✅ done: {args.target_date} {args.code} summary={summary}")

if __name__=="__main__":
    main()
