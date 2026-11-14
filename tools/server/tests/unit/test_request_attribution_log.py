import os
import tempfile
import pytest
from utils import *

server = ServerPreset.tinyllama2()


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0

    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


PROMPT = "The quick brown fox jumps over the lazy dog."


def chat(kind=None, session=None, session_name=None, agent=None, role=None, max_tokens=4,
         stream=False, ignore_eos=False):
    headers = {}
    for name, value in (("X-OMP-Kind", kind), ("X-OMP-Session", session),
                        ("X-OMP-Session-Name", session_name), ("X-OMP-Agent", agent),
                        ("X-OMP-Role", role)):
        if value is not None:
            headers[name] = value
    return server.make_request("POST", "/v1/chat/completions", headers=headers, data={
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        "ignore_eos": ignore_eos,
    })


def test_attribution_reaches_the_slot_log():
    global server
    server.start()
    log = LogReader(server.log_path)

    assert chat(kind="title", session="01a14291", session_name="Fix_llamacpp_attribution",
                agent="Main", role="smol").status_code == 200
    out = log.drain()
    assert "kind = 'title'" in out
    assert "session = '01a14291'" in out
    assert "session_name = 'Fix_llamacpp_attribution'" in out
    assert "agent = 'Main'" in out
    assert "role = 'smol'" in out

    # a client that sends no tag is still named on the same line
    assert chat().status_code == 200
    assert "kind = '-', session = '-', session_name = '-', agent = '-', role = '-'" in log.drain()


def test_attribution_is_reduced_to_log_safe_tokens():
    global server
    server.start()
    log = LogReader(server.log_path)

    # anything outside [a-zA-Z0-9_.-] is dropped: a header cannot be trusted to
    # stay inside the one line the slot log writes it on
    assert chat(kind="title; drop *table*", session="a b", session_name="x'; drop table --",
                agent="A;B", role="tiny>&<x").status_code == 200
    out = log.drain()
    assert "kind = 'titledroptable'" in out
    assert "session = 'ab'" in out
    assert "session_name = 'xdroptable--'" in out
    assert "agent = 'AB'" in out
    assert "role = 'tinyx'" in out

    assert chat(session_name="n" * 40).status_code == 200
    out = log.drain()
    assert "session_name = '" + "n" * 32 + "'" in out
    assert "n" * 33 not in out


def test_attribution_is_visible_on_slots_while_processing():
    global server
    server.n_predict = -1
    server.start()

    def generate():
        # stories260K finishes 200 tokens faster than one polling turn, so the
        # request is kept open (ignore_eos) long enough for /slots to be read.
        return chat(kind="recap", session="01a14291", session_name="Handoff",
                    agent="ClientAttribution", role="default", max_tokens=4096,
                    ignore_eos=True)

    def read_attribution():
        for _ in range(400):
            res = server.make_request("GET", "/slots")
            for slot in res.body:
                if slot["is_processing"] and "params" in slot:
                    params = slot["params"]
                    return (params["kind"], params["session"], params["session_name"],
                            params["agent"], params["role"])
            time.sleep(0.01)
        return None

    got, _ = parallel_function_calls([
        (read_attribution, ()),
        (generate, ()),
    ])
    assert got == ("recap", "01a14291", "Handoff", "ClientAttribution", "default")
