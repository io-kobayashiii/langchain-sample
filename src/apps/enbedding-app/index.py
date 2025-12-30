from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
import os
from pathlib import Path

# ドキュメントを読み込んでチャンクに分割する
# スクリプトのディレクトリを基準にパスを構築
script_dir = Path(__file__).parent
test_file_path = script_dir / 'test.txt'
raw_documents = TextLoader(str(test_file_path)).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
    chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# チャンクごとに埋め込みを生成してベクトルストアに保存する
embeddings_model = OpenAIEmbeddings()
connection = os.getenv("DATABASE_URL")
db = PGVector.from_documents(documents, embeddings_model, connection=connection)
results = db.similarity_search("How to use LangChain?", k=4)

# 検索結果を見やすく整形して表示
print("=" * 80)
print(f"検索結果: {len(results)}件")
print("=" * 80)
print()

for i, doc in enumerate(results, 1):
    print(f"【結果 {i}】")
    print(f"ID: {doc.id}")
    print(f"メタデータ: {doc.metadata}")
    print(f"コンテンツ:")
    print("-" * 80)
    print(doc.page_content)
    print("=" * 80)
    print()
