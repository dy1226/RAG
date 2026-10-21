"""长期记忆（情境记忆 + 语义记忆）测试：用临时库和假的大模型函数，不调用真实模型。"""
import json

import pytest

from memory import MemoryStore, _extract_json


def make_msgs(n_turns):
    msgs = []
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"问题{i}"})
        msgs.append({"role": "assistant", "content": f"回答{i}"})
    return msgs


class FakeLLM:
    """按 label 返回预设输出，并记录调用"""

    def __init__(self, summary="讨论了部署方案", ops=None):
        self.summary = summary
        self.ops = ops if ops is not None else []
        self.extract_raw = None      # 不为 None 时，偏好抽取直接返回这段原始文本
        self.calls = []

    def __call__(self, prompt, label):
        self.calls.append((label, prompt))
        if label == "memory-summary":
            return json.dumps({"summary": self.summary, "topics": ["部署"]}, ensure_ascii=False)
        if self.extract_raw is not None:
            return self.extract_raw
        return "前缀文字 " + json.dumps({"ops": self.ops}, ensure_ascii=False) + " 后缀"

    def labels(self):
        return [c[0] for c in self.calls]


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def store(tmp_path, llm):
    return MemoryStore(str(tmp_path / "mem.sqlite3"), llm, vectorstore=None,
                       summary_every_n_turns=3)


class TestEpisodic:
    def test_below_threshold_does_not_call_llm(self, store, llm):
        store.process_session("u1", "s1", make_msgs(2))
        assert llm.calls == []
        assert store.list_episodes("u1") == []

    def test_reaching_threshold_writes_summary(self, store, llm):
        store.process_session("u1", "s1", make_msgs(3))
        eps = store.list_episodes("u1")
        assert len(eps) == 1
        assert eps[0]["summary"] == "讨论了部署方案"
        assert eps[0]["topics"] == ["部署"]
        assert eps[0]["turn_count"] == 3

    def test_same_session_upserts_instead_of_duplicating(self, store, llm):
        store.process_session("u1", "s1", make_msgs(3))
        llm.summary = "更新后的摘要"
        store.process_session("u1", "s1", make_msgs(6))
        eps = store.list_episodes("u1")
        assert len(eps) == 1
        assert eps[0]["summary"] == "更新后的摘要"
        assert eps[0]["turn_count"] == 6

    def test_repeat_call_without_new_turns_is_noop(self, store, llm):
        store.process_session("u1", "s1", make_msgs(3))
        n = len(llm.calls)
        store.process_session("u1", "s1", make_msgs(3))
        assert len(llm.calls) == n

    def test_force_processes_remaining_turns(self, store, llm):
        store.process_session("u1", "s1", make_msgs(2), force=True)
        assert len(store.list_episodes("u1")) == 1

    def test_unparseable_summary_is_not_stored_and_retried(self, store, llm):
        llm.summary = ""          # 空摘要视为解析失败
        store.process_session("u1", "s1", make_msgs(3))
        assert store.list_episodes("u1") == []
        llm.summary = "第二次成功"
        n_before = len(llm.calls)
        store.process_session("u1", "s1", make_msgs(3))
        assert len(llm.calls) > n_before          # 失败时不记进度，下次会重试
        assert store.list_episodes("u1")[0]["summary"] == "第二次成功"


def add_op(content, conf=0.8, category="style"):
    return {"op": "add", "category": category, "content": content, "confidence": conf}


class TestSemantic:
    def test_add_preference(self, store, llm):
        llm.ops = [add_op("喜欢简短回答")]
        store.process_session("u1", "s1", make_msgs(3))
        prefs = store.list_preferences("u1")
        assert [p["content"] for p in prefs] == ["喜欢简短回答"]

    def test_low_confidence_is_rejected(self, store, llm):
        llm.ops = [add_op("可能喜欢表格", conf=0.3)]
        store.process_session("u1", "s1", make_msgs(3))
        assert store.list_preferences("u1") == []

    def test_duplicate_is_not_added_twice(self, store, llm):
        llm.ops = [add_op("喜欢简短回答"), add_op("喜欢 简短回答。")]
        store.process_session("u1", "s1", make_msgs(3))
        assert len(store.list_preferences("u1")) == 1

    def test_update_and_delete(self, store, llm):
        llm.ops = [add_op("喜欢简短回答")]
        store.process_session("u1", "s1", make_msgs(3))
        pid = store.list_preferences("u1")[0]["id"]

        llm.ops = [{"op": "update", "id": pid, "content": "喜欢详细回答", "confidence": 0.9}]
        store.process_session("u1", "s1", make_msgs(6))
        assert store.list_preferences("u1")[0]["content"] == "喜欢详细回答"

        llm.ops = [{"op": "delete", "id": pid}]
        store.process_session("u1", "s1", make_msgs(9))
        assert store.list_preferences("u1") == []

    def test_unknown_id_and_malformed_ops_are_ignored(self, store, llm):
        llm.ops = [{"op": "update", "id": "nope", "content": "x", "confidence": 0.9},
                   {"op": "delete", "id": ["bad"]}, "not-a-dict", {"op": "add"}]
        store.process_session("u1", "s1", make_msgs(3))
        assert store.list_preferences("u1") == []

    def test_unparseable_extract_output_is_retried(self, store, llm):
        llm.extract_raw = "不是 JSON"
        store.process_session("u1", "s1", make_msgs(3))
        state = store._get_state("u1", "s1")
        assert state["extracted_turns"] == 0       # 失败不记进度，下次会重试
        assert state["summarized_turns"] == 3      # 摘要不受影响

        llm.extract_raw = None
        llm.ops = [add_op("喜欢简短回答")]
        store.process_session("u1", "s1", make_msgs(3))
        assert [p["content"] for p in store.list_preferences("u1")] == ["喜欢简短回答"]

class FakeVectorStore:
    """最小的向量库替身：记录写入，检索时按 filter 过滤后原样返回（不做真实相似度计算）"""

    def __init__(self):
        self.docs = {}

    def add_documents(self, documents, ids):
        for d, i in zip(documents, ids):
            self.docs[i] = d

    def similarity_search(self, query, k=4, filter=None):
        out = [d for d in self.docs.values()
               if all(d.metadata.get(key) == v for key, v in (filter or {}).items())]
        return out[:k]

    def delete(self, ids):
        for i in ids:
            self.docs.pop(i, None)


class TestIsolationAndPrompt:
    def test_users_are_isolated(self, store, llm):
        llm.ops = [add_op("喜欢简短回答")]
        store.process_session("u1", "s1", make_msgs(3))
        assert store.list_preferences("u2") == []
        assert store.list_episodes("u2") == []
        assert store.build_prompt("u2", "你好") == ""

    def test_same_session_id_for_different_users_does_not_collide(self, store, llm):
        store.process_session("u1", "s1", make_msgs(3))
        llm.summary = "另一个用户的摘要"
        store.process_session("u2", "s1", make_msgs(3))
        assert store.list_episodes("u1")[0]["summary"] == "讨论了部署方案"
        assert store.list_episodes("u2")[0]["summary"] == "另一个用户的摘要"

    def test_prompt_contains_preferences_and_other_sessions_only(self, store, llm):
        llm.ops = [add_op("喜欢简短回答")]
        store.process_session("u1", "old", make_msgs(3))
        store.process_session("u1", "current", make_msgs(3))
        prompt = store.build_prompt("u1", "部署怎么做", session_id="current")
        assert "喜欢简短回答" in prompt
        assert prompt.count("讨论了部署方案") == 1       # 只有 old 会话，当前会话被排除
        assert "仍必须调用工具检索" in prompt            # 带有“记忆不是知识库事实”的约束

    def test_empty_memory_gives_empty_prompt(self, store):
        assert store.build_prompt("nobody", "你好") == ""

    def test_disabled_store_skips_everything(self, tmp_path, llm):
        s = MemoryStore(str(tmp_path / "off.sqlite3"), llm, enabled=False, summary_every_n_turns=1)
        s.process_session("u1", "s1", make_msgs(3), force=True)
        assert llm.calls == []
        assert s.build_prompt("u1", "x") == ""

    def test_read_errors_do_not_raise(self, tmp_path, llm):
        errors = []
        s = MemoryStore(str(tmp_path / "e.sqlite3"), llm, on_error=lambda l, m: errors.append(l))
        s._conn.close()
        assert s.build_prompt("u1", "x") == ""
        assert errors == ["memory-read"]


class TestVectorAndDelete:
    def test_vector_search_excludes_current_session_and_other_users(self, tmp_path, llm):
        vs = FakeVectorStore()
        s = MemoryStore(str(tmp_path / "v.sqlite3"), llm, vectorstore=vs, summary_every_n_turns=3)
        s.process_session("u1", "a", make_msgs(3))
        s.process_session("u1", "b", make_msgs(3))
        s.process_session("u2", "c", make_msgs(3))
        found = s.search_episodes("u1", "部署", exclude_session="b")
        assert [e["session_id"] for e in found] == ["a"]

    def test_delete_episode_removes_vector_too(self, tmp_path, llm):
        vs = FakeVectorStore()
        s = MemoryStore(str(tmp_path / "v.sqlite3"), llm, vectorstore=vs, summary_every_n_turns=3)
        s.process_session("u1", "a", make_msgs(3))
        ep_id = s.list_episodes("u1")[0]["id"]
        assert s.delete_episode("u2", ep_id) is False       # 不能删别人的
        assert s.delete_episode("u1", ep_id) is True
        assert vs.docs == {}

    def test_delete_preference_is_scoped_to_user(self, store, llm):
        llm.ops = [add_op("喜欢简短回答")]
        store.process_session("u1", "s1", make_msgs(3))
        pid = store.list_preferences("u1")[0]["id"]
        assert store.delete_preference("u2", pid) is False
        assert store.delete_preference("u1", pid) is True

    def test_clear_user_only_clears_that_user(self, tmp_path, llm):
        vs = FakeVectorStore()
        llm.ops = [add_op("喜欢简短回答")]
        s = MemoryStore(str(tmp_path / "v.sqlite3"), llm, vectorstore=vs, summary_every_n_turns=3)
        s.process_session("u1", "a", make_msgs(3))
        s.process_session("u2", "b", make_msgs(3))
        assert s.clear_user("u1") == {"episodes": 1, "preferences": 1}
        assert s.list_episodes("u1") == [] and s.list_preferences("u1") == []
        assert len(s.list_episodes("u2")) == 1 and len(vs.docs) == 1

    def test_preference_count_is_capped(self, store, llm):
        llm.ops = [add_op(f"偏好{i}") for i in range(30)]
        store.process_session("u1", "s1", make_msgs(3))
        assert len(store.list_preferences("u1")) == 20


class TestExtractJson:
    def test_with_prefix_and_suffix(self):
        assert _extract_json('好的 {"a": 1} 完毕') == {"a": 1}

    @pytest.mark.parametrize("raw", ["", None, "没有 JSON", "{坏的", "[1, 2]"])
    def test_invalid_returns_none(self, raw):
        assert _extract_json(raw) is None
