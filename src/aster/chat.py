"""Talk to it: a tiny chat server over the native sentence generator.

POST /say {"text": "..."} -> {"reply": "..."} ; the page at / is the UI.
Everything the reply says is formed by word neurons firing along learned
order -- there is no template behind it.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from aster.cognition.plastic import PlasticNetwork
from aster.speech import Speaker

CORPUS_PATH = Path("/tmp/speech-data/corpus.json")

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>talk to it</title>
<style>
 body { font: 18px/1.5 system-ui; max-width: 44rem; margin: 3rem auto; padding: 0 1rem; background:#101418; color:#e8e6e3 }
 h1 { font-size: 1.3rem; color:#8ecae6 }
 #log { border:1px solid #2b3a42; border-radius:10px; padding:1rem; min-height:14rem; background:#16202a }
 .you { color:#ffd166 } .it { color:#8ecae6 }
 form { display:flex; gap:.5rem; margin-top:1rem }
 input { flex:1; font:inherit; padding:.6rem .8rem; border-radius:8px; border:1px solid #2b3a42; background:#0d1216; color:#e8e6e3 }
 button { font:inherit; padding:.6rem 1.1rem; border-radius:8px; border:0; background:#8ecae6; color:#10202a; cursor:pointer }
 small { color:#7c8b95 }
</style></head><body>
<h1>talk to it</h1>
<p><small>every reply is a sentence formed by its word neurons firing along
what it has learned -- no fixed template. teach it: <em>the sun is a star</em>.
ask it: <em>what does print mean?</em> or just talk.</small></p>
<div id="log"></div>
<form id="f"><input id="i" autocomplete="off" placeholder="say something..."><button>send</button></form>
<script>
const log = document.getElementById('log');
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = document.getElementById('i');
  const text = input.value.trim();
  if (!text) return;
  log.innerHTML += `<div class="you">you&gt; ${text}</div>`;
  input.value = '';
  const r = await fetch('/say', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})});
  const data = await r.json();
  log.innerHTML += `<div class="it">it&gt; ${data.reply}</div>`;
  log.scrollTop = log.scrollHeight;
});
document.getElementById('i').focus();
</script></body></html>
"""


def build_speaker() -> Speaker:
    net = PlasticNetwork()
    speaker = Speaker(net=net, warm=True)
    if CORPUS_PATH.exists():
        data = json.loads(CORPUS_PATH.read_text())
        glosses = data["glosses"][:2500]
        sentences = data["sentences"][:12000]
        for word, gloss in glosses:
            net.observe(word, "means", gloss)
        net.observe_corpus([f"{w} means {g}" for w, g in glosses])
        net.observe_corpus(sentences)
        net.tick(100)
    return speaker


class Handler(BaseHTTPRequestHandler):
    speaker: Speaker  # set in main

    def do_GET(self) -> None:
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        reply = self.speaker.respond(str(payload.get("text", "")))
        body = json.dumps({"reply": reply}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    Handler.speaker = build_speaker()
    server = ThreadingHTTPServer(("0.0.0.0", 8080), Handler)
    print("chat server on 0.0.0.0:8080", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
