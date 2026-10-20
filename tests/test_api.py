"""接口层测试：用 FastAPI TestClient 直接请求接口（不调用大模型）。

⚠️ 会往 uploads/ 里写一个测试用文件（该目录已 gitignore）。
"""
from fastapi.testclient import TestClient

import main
import pytest

client = TestClient(main.app)


def test_hello():
    r = client.get("/")
    assert r.status_code == 200


def test_metrics_shape():
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "calls" in body
    assert "total_tokens" in body


@pytest.mark.requires_kb
def test_rag_documents_lists_sources():
    r = client.get("/rag/documents")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert isinstance(body["documents"], list)
    if body["documents"]:
        first = body["documents"][0]
        assert "source" in first and "chunks" in first


@pytest.mark.requires_kb
def test_rag_upload_returns_success():
    txt = "测试文档内容".encode("utf-8")
    r = client.post("/rag/upload", files={"file": ("test.txt", txt)})
    body = r.json()
    assert body["success"] is True
    assert "chunks" in body


def test_rag_upload_rejects_empty_file():
    r = client.post("/rag/upload", files={"file": ("empty.txt", b"")})
    body = r.json()
    assert body["success"] is False


@pytest.mark.requires_kb
def test_rag_chat_responds():
    r = client.post("/rag/chat", json={"message": "你好"})
    assert r.status_code == 200
    body = r.json()
    assert "response" in body or "error" in body
