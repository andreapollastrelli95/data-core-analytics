"""Parti dell'assistente che non richiedono la rete: chiave, cronologia, messaggi, errori."""
from types import SimpleNamespace

import pytest

import ai_config
import ai_engine as ai
import ai_tools
from cleaner import DataError


def test_api_key_from_environment(monkeypatch):
    monkeypatch.setenv(ai_config.API_KEY_ENV, "  chiave-di-prova ")
    assert ai.api_key() == "chiave-di-prova" and ai.has_api_key()


def test_missing_key_gives_instructions(monkeypatch):
    monkeypatch.delenv(ai_config.API_KEY_ENV, raising=False)
    monkeypatch.setattr(ai, "api_key", lambda: "")
    with pytest.raises(DataError, match="setx"):
        ai.generate([{"role": "user", "text": "ciao"}], "sistema")


def test_chat_history_sends_only_what_is_needed():
    att = ai.Attachment("note.txt", "text", "text/plain", text="contenuto")
    history = []
    for i in range(8):
        history += [ai.Turn("user", f"domanda {i}", [att] if i in (5, 7) else [],
                            "schema" if i == 6 else ""), ai.Turn("assistant", f"risposta {i}")]
    contents = ai._contents(history)
    assert len(contents) == ai_config.MAX_HISTORY_MESSAGES and contents[0].role == "user"
    text = " ".join(p.text or "" for c in contents for p in c.parts)
    assert text.count("--- File: note.txt ---") == 1                 # solo l'ultimo allegato
    assert "erano allegati: note.txt" in text and text.count("<contesto_dataset>") == 1


def test_agent_messages_conversion():
    calls = [("aggrega", {"per": ["Regione"]})]
    contents = ai._agent_contents([
        {"role": "user", "text": "domanda"},
        {"role": "model", "raw": None, "text": "", "calls": calls},
        {"role": "tool", "results": [("aggrega", {"righe": 5})], "text": "ora rispondi"}])
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].function_call.name == "aggrega"
    assert contents[2].parts[0].function_response.response == {"righe": 5}
    assert contents[2].parts[1].text == "ora rispondi"


def test_tool_declarations_are_accepted_by_the_sdk():
    from google.genai import types

    for t in ai_tools.declarations():
        types.FunctionDeclaration(name=t["name"], description=t["description"],
                                  parameters_json_schema=t["parameters"])


def fake_client(response=None, error=None):
    def generate_content(**kwargs):
        fake_client.kwargs = kwargs
        if error:
            raise error
        return response
    return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))


def test_generate_parses_calls_and_text(monkeypatch):
    part_call = SimpleNamespace(function_call=SimpleNamespace(name="aggrega", args={"per": ["x"]}),
                                text=None, thought=False)
    part_thought = SimpleNamespace(function_call=None, text="ragionamento", thought=True)
    part_text = SimpleNamespace(function_call=None, text="risposta", thought=False)
    content = SimpleNamespace(parts=[part_thought, part_call, part_text])
    response = SimpleNamespace(prompt_feedback=None, usage_metadata=SimpleNamespace(total_token_count=42),
                               candidates=[SimpleNamespace(content=content, finish_reason=None)])
    monkeypatch.setattr(ai, "_client", lambda: fake_client(response))
    out = ai.generate([{"role": "user", "text": "ciao"}], "sistema", tools=ai_tools.declarations(),
                      allow_tools=False)
    assert out.calls == [("aggrega", {"per": ["x"]})] and out.text == "risposta"
    assert out.tokens == 42 and out.raw is content
    config = fake_client.kwargs["config"]
    assert len(config.tools[0].function_declarations) == len(ai_tools.TOOLS)
    assert config.tool_config.function_calling_config.mode.value == "NONE"
    assert config.automatic_function_calling.disable is True       # gli strumenti li esegue l'app


def test_generate_empty_response(monkeypatch):
    response = SimpleNamespace(prompt_feedback=None, usage_metadata=None, candidates=[])
    monkeypatch.setattr(ai, "_client", lambda: fake_client(response))
    with pytest.raises(DataError, match="vuota"):
        ai.generate([{"role": "user", "text": "ciao"}], "sistema")


@pytest.mark.parametrize("code,message,expected", [
    (400, "API key not valid", "non è valida"),
    (404, "model not found", "non è disponibile"),
    (503, "overloaded", "temporaneamente non disponibile"),
    (403, "forbidden", "non ha accesso"),
])
def test_api_errors_become_instructions(code, message, expected):
    exc = SimpleNamespace(code=code, status="", message=message, details=None)
    assert expected in str(ai._api_error(exc))


def test_daily_quota_is_recognised():
    details = {"error": {"details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                         "quotaValue": "20"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3600s"}]}}
    exc = SimpleNamespace(code=429, status="RESOURCE_EXHAUSTED", message="quota", details=details)
    info = ai._quota_info(exc)
    assert info.daily and info.limit == "20" and info.delay == 3600
    assert "limite giornaliero" in str(ai._api_error(exc))


def test_attachment_limits(tmp_path):
    ok = tmp_path / "dati.csv"
    ok.write_text("a,b\n1,2\n", encoding="utf-8")
    assert ai.load_attachment(str(ok)).kind == "text"
    bad = tmp_path / "programma.exe"
    bad.write_bytes(b"\x00\x01")
    with pytest.raises(DataError):
        ai.load_attachment(str(bad))
    fake_pdf = tmp_path / "falso.pdf"
    fake_pdf.write_text("non sono un pdf")
    with pytest.raises(DataError):
        ai.load_attachment(str(fake_pdf))
