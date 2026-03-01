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


def main():
    _install_resource_stub()
    import torch
    import vec2text

    payload = json.load(sys.stdin)
    texts = payload.get("texts", [])
    model_name = payload.get("model", "gtr-base")
    num_steps = int(payload.get("num_steps", 20))
    max_length = int(payload.get("max_length", 32))

    if not texts:
        json.dump({"inversions": [], "scores": []}, sys.stdout)
        return

    # Emit loading event
    print(json.dumps({"event": "log", "message": "loading vec2text model..."}), flush=True)

    corrector = vec2text.load_pretrained_corrector(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(corrector, "to"):
        corrector.to(device)
    if hasattr(corrector, "model") and hasattr(corrector.model, "eval"):
        corrector.model.eval()
    if hasattr(corrector, "inversion_trainer") and hasattr(corrector.inversion_trainer, "model"):
        if hasattr(corrector.inversion_trainer.model, "eval"):
            corrector.inversion_trainer.model.eval()

    print(json.dumps({"event": "log", "message": f"inverting {len(texts)} chunks..."}), flush=True)

    tokenizer = corrector.embedder_tokenizer
    with torch.inference_mode():
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        embeddings = corrector.inversion_trainer.call_embedding_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Invert embeddings back to text
        recon_texts = vec2text.invert_embeddings(
            embeddings=embeddings,
            corrector=corrector,
            num_steps=num_steps,
        )

        # Calculate similarity scores
        recon_inputs = tokenizer(
            recon_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        recon_embeddings = corrector.inversion_trainer.call_embedding_model(
            input_ids=recon_inputs.input_ids,
            attention_mask=recon_inputs.attention_mask,
        )
        recon_embeddings = torch.nn.functional.normalize(recon_embeddings, p=2, dim=1)
        scores = torch.nn.functional.cosine_similarity(
            embeddings, recon_embeddings, dim=1
        ).cpu().tolist()

    # Emit results for each chunk
    for i, (orig, recon, score) in enumerate(zip(texts, recon_texts, scores)):
        print(json.dumps({
            "event": "inversion",
            "index": i,
            "original": orig,
            "inverted": recon,
            "score": score,
        }), flush=True)

    print(json.dumps({"event": "done"}), flush=True)


if __name__ == "__main__":
    main()
