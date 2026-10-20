#!/usr/bin/env venv/bin/python3
"""
Chroma DB 和 SQLite 可视化查看工具
用法:
  python3 db_viewer.py chroma              # 查看向量库概览
  python3 db_viewer.py chroma --all       # 查看所有向量记录
  python3 db_viewer.py chroma --search 关键词  # 语义搜索
  python3 db_viewer.py sqlite             # 查看聊天记忆
  python3 db_viewer.py -h                 # 帮助
"""
import argparse
import sqlite3
import sys
import os

# Chroma API
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "venv/lib/python3.12/site-packages"))
import chromadb


# ─────────────────────────────────────────────
#  样式
# ─────────────────────────────────────────────
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
MAGENTA= "\033[35m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def section(title):
    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")

def kv(key, value, indent=2):
    print(f"{' '*indent}{DIM}{key}: {RESET}{value}")


# ─────────────────────────────────────────────
#  Chroma DB 查看
# ─────────────────────────────────────────────
def view_chroma(show_all=False, search=None, limit=20):
    client = chromadb.PersistentClient(path="chroma_db")
    collections = client.list_collections()
    
    section("Chroma 向量数据库")
    print(f"  路径: {os.path.abspath('chroma_db')}")
    print(f"  Collection 数量: {len(collections)}")
    
    for col in collections:
        c = client.get_collection(col.name)
        print(f"\n{GREEN}▶ Collection: {col.name}{RESET}")
        print(f"  向量数量: {c.count()}")
        print(f"  嵌入模型维度: ", end="")
        
        # 取一条样例确定维度
        sample = c.get(limit=1, include=["embeddings"])
        emb = sample["embeddings"]
        if emb is not None and len(emb) > 0:
            print(len(emb[0]))
        else:
            print("未知")
        
        # 语义搜索
        if search:
            print(f"\n  {YELLOW}🔍 搜索: \"{search}\"{RESET}")
            results = c.query(query_texts=[search], n_results=min(5, c.count()), include=["documents","metadatas","distances"])
            for i, (doc, meta, dist) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0])):
                print(f"\n  {MAGENTA}结果 {i+1} (距离: {dist:.4f}){RESET}")
                print(f"    ID:     {results['ids'][0][i]}")
                print(f"    来源:   {meta.get('source','?')}")
                print(f"    内容:   {doc[:300]}{'...' if len(doc)>300 else ''}")
            return
        
        # 列出记录
        print(f"\n  {DIM}记录列表 (前 {limit} 条){RESET}")
        records = c.get(include=["documents","metadatas","embeddings"])
        count = len(records["ids"])
        to_show = min(limit, count) if not show_all else count
        
        for i in range(to_show):
            print(f"\n  {GREEN}#{i+1}  ID: {records['ids'][i]}{RESET}")
            print(f"    来源: {records['metadatas'][i].get('source','?')}")
            doc = records['documents'][i]
            print(f"    内容: {doc[:200]}{'...' if len(doc)>200 else ''}")
            emb = records['embeddings'][i]
            print(f"    向量: [{emb[0]:.4f}, {emb[1]:.4f}, {emb[2]:.4f}, ...]  (共{len(emb)}维)")
        
        if not show_all and count > limit:
            print(f"\n  {DIM}... 还有 {count - limit} 条记录 (用 --all 全部显示){RESET}")


# ─────────────────────────────────────────────
#  SQLite 查看
# ─────────────────────────────────────────────
def view_sqlite():
    db_path = "chat_memory.sqlite3"
    if not os.path.exists(db_path):
        print(f"{YELLOW}警告: {db_path} 不存在{RESET}")
        return
    
    section("SQLite 聊天记忆数据库")
    print(f"  路径: {os.path.abspath(db_path)}")
    print(f"  文件大小: {os.path.getsize(db_path)/1024:.1f} KB")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # 列出所有表
    c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in c.fetchall()]
    print(f"\n  表: {tables}")
    
    for table in tables:
        c.execute(f"SELECT COUNT(*) FROM {table}")
        n = c.fetchone()[0]
        print(f"\n  {GREEN}▶ 表: {table} ({n} 行){RESET}")
        
        # 列信息
        c.execute(f"PRAGMA table_info({table})")
        cols = [(r[1], r[2]) for r in c.fetchall()]
        print(f"    列: {[f'{n} ({t})' for n,t in cols]}")
        
        # 前几条
        try:
            c.execute(f"SELECT * FROM {table} LIMIT 3")
            rows = c.fetchall()
            for j, row in enumerate(rows):
                print(f"\n    行 {j+1}:")
                for field, val in zip([col[0] for col in cols], row):
                    val_str = str(val)
                    if len(val_str) > 100:
                        val_str = val_str[:100] + "..."
                    # BLOB 标记
                    if isinstance(val, bytes):
                        val_str = f"<BLOB {len(val)} bytes>"
                    print(f"      {field}: {val_str}")
        except Exception as e:
            print(f"    {DIM}读取数据失败: {e}{RESET}")
    
    conn.close()


# ─────────────────────────────────────────────
#  入口
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Chroma DB 和 SQLite 可视化查看工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("db", nargs="?", choices=["chroma","sqlite"], 
                        help="查看哪个数据库 (chroma / sqlite)")
    parser.add_argument("--all", action="store_true", help="显示所有记录")
    parser.add_argument("--search", "-s", type=str, help="在向量库中语义搜索")
    parser.add_argument("--limit", "-n", type=int, default=20, help="最多显示多少条 (默认 20)")
    parser.add_argument("--list", "-l", action="store_true", help="列出所有数据库概览")
    
    args = parser.parse_args()
    
    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if args.list or not args.db:
        view_chroma(limit=args.limit)
        view_sqlite()
    elif args.db == "chroma":
        view_chroma(show_all=args.all, search=args.search, limit=args.limit)
    elif args.db == "sqlite":
        view_sqlite()

if __name__ == "__main__":
    main()
