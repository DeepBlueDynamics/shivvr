"""
Persistent model server - loads vec2text once and handles multiple requests.

Protocol:
- Reads JSON commands from stdin, one per line
- Writes JSON responses to stdout, one per line
- Commands: embed, invert, score, shutdown
"""
import json
import platform
import sys


def _install_resource_stub():
    if platform.system() != "Windows":
        return
    if "resource" in sys.modules:
        return

    class _RUsage:
        ru_utime = 0.0
        ru_stime = 0.0
        ru_maxrss = 0
        ru_ixrss = 0
        ru_idrss = 0
        ru_isrss = 0
        ru_minflt = 0
        ru_majflt = 0
        ru_nswap = 0
        ru_inblock = 0
        ru_oublock = 0
        ru_msgsnd = 0
        ru_msgrcv = 0
        ru_nsignals = 0
        ru_nvcsw = 0
        ru_nivcsw = 0

    class _ResourceStub:
        RLIMIT_AS = 9
        RLIMIT_DATA = 2
        RLIMIT_STACK = 3
        RLIM_INFINITY = -1
        RUSAGE_SELF = 0

        @staticmethod
        def getrusage(_who):
            return _RUsage()

        @staticmethod
        def getrlimit(_resource):
            return (_ResourceStub.RLIM_INFINITY, _ResourceStub.RLIM_INFINITY)

        @staticmethod
        def setrlimit(_resource, _limits):
            return None

    sys.modules["resource"] = _ResourceStub()


def _emit(event):
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def _log(message):
    _emit({"event": "log", "message": message})


class ModelServer:
    def __init__(self, model_name="gtr-base"):
        self.model_name = model_name
        self.corrector = None
        self.tokenizer = None
        self.device = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return

        import torch
        import vec2text
        from transformers import AutoTokenizer

        _log("loading vec2text model (will be cached)...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _log(f"using device: {self.device}")

        self.corrector = vec2text.load_pretrained_corrector(self.model_name)
        if hasattr(self.corrector, "to"):
            self.corrector.to(self.device)
        if hasattr(self.corrector, "model") and hasattr(self.corrector.model, "eval"):
            self.corrector.model.eval()
        if hasattr(self.corrector, "inversion_trainer") and hasattr(self.corrector.inversion_trainer, "model"):
            if hasattr(self.corrector.inversion_trainer.model, "eval"):
                self.corrector.inversion_trainer.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/gtr-t5-base", use_fast=True
        )

        self.loaded = True
        _log("model loaded and cached")

    def embed(self, texts, max_length=32):
        import torch

        if not texts:
            return []

        with torch.inference_mode():
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            embeddings = self.corrector.inversion_trainer.call_embedding_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.detach().cpu().tolist()

    def invert(self, texts, num_steps=20, max_length=32):
        import torch
        import vec2text

        if not texts:
            return [], []

        with torch.inference_mode():
            # Embed original texts
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            embeddings = self.corrector.inversion_trainer.call_embedding_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # Invert embeddings to text
            recon_texts = vec2text.invert_embeddings(
                embeddings=embeddings,
                corrector=self.corrector,
                num_steps=num_steps,
            )

            # Calculate similarity scores
            recon_inputs = self.tokenizer(
                recon_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            recon_embeddings = self.corrector.inversion_trainer.call_embedding_model(
                input_ids=recon_inputs.input_ids,
                attention_mask=recon_inputs.attention_mask,
            )
            recon_embeddings = torch.nn.functional.normalize(recon_embeddings, p=2, dim=1)
            scores = torch.nn.functional.cosine_similarity(
                embeddings, recon_embeddings, dim=1
            ).cpu().tolist()

            return recon_texts, scores

    def invert_embeddings(self, embeddings, num_steps=20, max_length=32):
        import torch
        import vec2text

        if not embeddings:
            return [], []

        with torch.inference_mode():
            tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
            tensor = torch.nn.functional.normalize(tensor, p=2, dim=1)

            recon_texts = vec2text.invert_embeddings(
                embeddings=tensor,
                corrector=self.corrector,
                num_steps=num_steps,
            )

            recon_inputs = self.tokenizer(
                recon_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)
            recon_embeddings = self.corrector.inversion_trainer.call_embedding_model(
                input_ids=recon_inputs.input_ids,
                attention_mask=recon_inputs.attention_mask,
            )
            recon_embeddings = torch.nn.functional.normalize(recon_embeddings, p=2, dim=1)
            scores = torch.nn.functional.cosine_similarity(
                tensor, recon_embeddings, dim=1
            ).cpu().tolist()

            return recon_texts, scores

    def score(self, texts, max_length=32):
        """Score text pairs by cosine similarity of their embeddings."""
        import torch

        if len(texts) < 2:
            return []

        embeddings = self.embed(texts, max_length)
        scores = []
        for i in range(len(embeddings) - 1):
            a = torch.tensor(embeddings[i])
            b = torch.tensor(embeddings[i + 1])
            sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            scores.append(sim)
        return scores


def main():
    _install_resource_stub()

    # Read initial config
    config_line = sys.stdin.readline()
    if not config_line:
        _emit({"event": "error", "message": "no config received"})
        return

    try:
        config = json.loads(config_line)
    except json.JSONDecodeError as e:
        _emit({"event": "error", "message": f"invalid config JSON: {e}"})
        return

    model_name = config.get("model", "gtr-base")
    server = ModelServer(model_name)

    try:
        server.load()
    except Exception as e:
        _emit({"event": "error", "message": f"failed to load model: {e}"})
        return

    _emit({"event": "ready"})

    # Main command loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            _emit({"event": "error", "message": f"invalid command JSON: {e}"})
            continue

        command = cmd.get("command", "")
        request_id = cmd.get("id", None)

        try:
            if command == "shutdown":
                _emit({"event": "shutdown", "id": request_id})
                break

            elif command == "embed":
                texts = cmd.get("texts", [])
                max_length = cmd.get("max_length", 32)
                embeddings = server.embed(texts, max_length)
                _emit({
                    "event": "result",
                    "id": request_id,
                    "command": "embed",
                    "embeddings": embeddings,
                })

            elif command == "invert":
                texts = cmd.get("texts", [])
                num_steps = cmd.get("num_steps", 20)
                max_length = cmd.get("max_length", 32)
                inversions, scores = server.invert(texts, num_steps, max_length)
                _emit({
                    "event": "result",
                    "id": request_id,
                    "command": "invert",
                    "inversions": inversions,
                    "scores": scores,
                })

            elif command == "invert_embeddings":
                embeddings = cmd.get("embeddings", [])
                num_steps = cmd.get("num_steps", 20)
                max_length = cmd.get("max_length", 32)
                inversions, scores = server.invert_embeddings(
                    embeddings, num_steps, max_length
                )
                _emit({
                    "event": "result",
                    "id": request_id,
                    "command": "invert_embeddings",
                    "inversions": inversions,
                    "scores": scores,
                })

            elif command == "score":
                texts = cmd.get("texts", [])
                max_length = cmd.get("max_length", 32)
                scores = server.score(texts, max_length)
                _emit({
                    "event": "result",
                    "id": request_id,
                    "command": "score",
                    "scores": scores,
                })

            elif command == "ping":
                _emit({"event": "pong", "id": request_id})

            else:
                _emit({"event": "error", "id": request_id, "message": f"unknown command: {command}"})

        except Exception as e:
            _emit({"event": "error", "id": request_id, "message": str(e)})

    _log("server shutdown")


if __name__ == "__main__":
    main()
