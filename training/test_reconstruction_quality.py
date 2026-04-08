"""
Test whether mixed-topic windows have lower reconstruction quality
than coherent single-topic windows.
"""

import argparse
import importlib
import inspect
import platform
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


VEC2TEXT = None


def _install_resource_stub() -> None:
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


def _import_vec2text():
    script_dir = Path(__file__).resolve().parent
    local_pkg = script_dir / "vec2text"
    if local_pkg.exists():
        sys.path = [p for p in sys.path if Path(p).resolve() != script_dir]
        sys.path = [p for p in sys.path if p not in ("", ".")]
    return importlib.import_module("vec2text")


def _get_vec2text():
    global VEC2TEXT
    if VEC2TEXT is None:
        _install_resource_stub()
        VEC2TEXT = _import_vec2text()
    return VEC2TEXT


def _get_device(corrector) -> torch.device:
    if hasattr(corrector, "device"):
        return corrector.device
    if hasattr(corrector, "parameters"):
        try:
            return next(corrector.parameters()).device
        except StopIteration:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_embedder(corrector):
    if hasattr(corrector, "embedder"):
        return corrector.embedder
    if hasattr(corrector, "inverter") and hasattr(corrector.inverter, "embedder"):
        return corrector.inverter.embedder
    raise RuntimeError("Could not find embedder on corrector. Check vec2text API.")


def _mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    summed = torch.sum(last_hidden * mask, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    return summed / counts


def _embed_text(text: str, embedder, device: torch.device, max_length: int) -> torch.Tensor:
    if hasattr(embedder, "tokenizer") and hasattr(embedder, "model"):
        inputs = embedder.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        model = embedder.model
        if hasattr(model, "to"):
            model = model.to(device)

        with torch.inference_mode():
            outputs = model(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embedding = outputs.pooler_output
        else:
            embedding = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
    elif hasattr(embedder, "encode"):
        embedding = embedder.encode([text], normalize_embeddings=True)
        embedding = torch.tensor(embedding, dtype=torch.float32, device=device)
    else:
        raise RuntimeError("Unsupported embedder interface. Check vec2text API.")

    return torch.nn.functional.normalize(embedding, p=2, dim=1)


def _resolve_module(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _call_with_signature(fn, contexts):
    sig = inspect.signature(fn)
    for context in contexts:
        args = []
        kwargs = {}
        ok = True
        for name, param in sig.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            if name in context:
                value = context[name]
            elif param.default is not param.empty:
                continue
            else:
                ok = False
                break

            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD) and param.default is param.empty:
                args.append(value)
            else:
                kwargs[name] = value

        if not ok:
            continue
        return fn(*args, **kwargs)

    return None


def _build_contexts(
    model_name: str,
    device: str,
    inversion_model: str,
    corrector_model: str,
    embedder,
    tokenizer,
    model_for_model_param,
):
    base = {
        "model_name": model_name,
        "model_name_or_path": model_name,
        "model_id": model_name,
        "model_path": model_name,
        "pretrained_model_name_or_path": model_name,
        "embedder": embedder if embedder is not None else model_name,
        "inversion_model": inversion_model,
        "inversion_model_name": inversion_model,
        "hypothesis_model": inversion_model,
        "hypothesis_model_name": inversion_model,
        "corrector_model": corrector_model,
        "corrector_model_name": corrector_model,
        "corrector_name": corrector_model,
        "device": device,
        "torch_device": device,
        "fp16": False,
        "use_fp16": False,
        "dtype": torch.float32,
        "torch_dtype": torch.float32,
    }

    if embedder is not None:
        base["embedder"] = embedder
        base["encoder"] = embedder
        base["sentence_encoder"] = embedder

    if tokenizer is not None:
        base["tokenizer"] = tokenizer

    contexts = []
    ctx = dict(base)
    if model_for_model_param is not None:
        ctx["model"] = model_for_model_param
    contexts.append(ctx)

    if model_for_model_param is not None and model_for_model_param is not model_name:
        ctx_alt = dict(base)
        ctx_alt["model"] = model_name
        contexts.append(ctx_alt)

    return contexts


def _load_embedder(model_name: str):
    vec2text_mod = _get_vec2text()
    modules = [
        vec2text_mod,
        _resolve_module("vec2text.api"),
        _resolve_module("vec2text.models"),
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    errors = []

    for mod in modules:
        if mod is None:
            continue
        for attr in ("load_embedder_and_tokenizer", "load_embedder", "load_encoder_decoder"):
            fn = getattr(mod, attr, None)
            if not callable(fn):
                continue
            contexts = _build_contexts(
                model_name=model_name,
                device=device,
                inversion_model=None,
                corrector_model=None,
                embedder=None,
                tokenizer=None,
                model_for_model_param=model_name,
            )
            try:
                result = _call_with_signature(fn, contexts)
            except Exception as exc:
                errors.append(f"{mod.__name__}.{attr}: {exc}")
                continue
            if result is None:
                continue
            if isinstance(result, tuple):
                if len(result) >= 2:
                    return result[0], result[1]
                return result[0], None
            return result, getattr(result, "tokenizer", None)

    return None, None, errors


def _load_corrector(model_name: str, inversion_model: str, corrector_model: str, embedder, tokenizer):
    vec2text_mod = _get_vec2text()
    candidates = []
    modules = [
        vec2text_mod,
        _resolve_module("vec2text.api"),
        _resolve_module("vec2text.models"),
    ]

    for mod in modules:
        if mod is None:
            continue
        for attr in (
            "load_corrector",
            "load_pretrained_corrector",
            "load_corrector_and_tokenizer",
            "load_model",
            "load",
        ):
            fn = getattr(mod, attr, None)
            if callable(fn):
                candidates.append((mod.__name__, attr, fn))

    errors = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for mod_name, attr, fn in candidates:
        try:
            contexts = _build_contexts(
                model_name=model_name,
                device=device,
                inversion_model=inversion_model,
                corrector_model=corrector_model,
                embedder=embedder,
                tokenizer=tokenizer,
                model_for_model_param=embedder if embedder is not None else model_name,
            )
            result = _call_with_signature(fn, contexts)
            if result is None:
                try:
                    result = fn()
                except Exception as exc:
                    errors.append(f"{mod_name}.{attr}: {exc}")
                    continue
        except Exception as exc:
            errors.append(f"{mod_name}.{attr}: {exc}")
            continue

        if isinstance(result, tuple):
            for item in result:
                if hasattr(item, "embedder"):
                    return item
            return result[0]

        return result

    available = []
    for mod in modules:
        if mod is None:
            continue
        available.extend([name for name in dir(mod) if "load" in name.lower()])

    detail = "\n".join(errors) if errors else "No loader functions succeeded."
    raise RuntimeError(
        "Could not find a vec2text corrector loader.\n"
        f"Available load-like functions: {sorted(set(available))}\n"
        f"Errors: {detail}"
    )


def _get_invert_fn():
    vec2text_mod = _get_vec2text()
    for mod in (vec2text_mod, _resolve_module("vec2text.api"), _resolve_module("vec2text.inversion")):
        if mod is None:
            continue
        fn = getattr(mod, "invert_embeddings", None)
        if callable(fn):
            return fn
    return None


def get_reconstruction_quality(
    text: str,
    corrector,
    num_steps: int = 20,
    max_length: int = 128,
) -> Tuple[str, float]:
    """
    Embed text, invert it, re-embed, measure cosine similarity.
    Returns (reconstructed_text, quality_score)
    """
    device = _get_device(corrector)
    if hasattr(corrector, "inversion_trainer") and hasattr(corrector, "embedder_tokenizer"):
        inputs = corrector.embedder_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.inference_mode():
            original_embedding = corrector.inversion_trainer.call_embedding_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
        original_embedding = torch.nn.functional.normalize(original_embedding, p=2, dim=1)
    else:
        embedder = _get_embedder(corrector)
        original_embedding = _embed_text(text, embedder, device, max_length)

    invert_fn = _get_invert_fn()
    if invert_fn is None:
        raise RuntimeError("vec2text.invert_embeddings not found; API may have changed.")

    reconstructed = invert_fn(
        embeddings=original_embedding,
        corrector=corrector,
        num_steps=num_steps,
    )[0]

    if hasattr(corrector, "inversion_trainer") and hasattr(corrector, "embedder_tokenizer"):
        recon_inputs = corrector.embedder_tokenizer(
            reconstructed,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.inference_mode():
            reconstructed_embedding = corrector.inversion_trainer.call_embedding_model(
                input_ids=recon_inputs.input_ids,
                attention_mask=recon_inputs.attention_mask,
            )
        reconstructed_embedding = torch.nn.functional.normalize(reconstructed_embedding, p=2, dim=1)
    else:
        reconstructed_embedding = _embed_text(reconstructed, embedder, device, max_length)

    quality = torch.nn.functional.cosine_similarity(
        original_embedding,
        reconstructed_embedding,
    ).item()

    return reconstructed, quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gtr-base", help="vec2text model name")
    parser.add_argument(
        "--embedder_model",
        type=str,
        default=None,
        help="Override embedder model name if required by vec2text API",
    )
    parser.add_argument(
        "--inversion_model",
        type=str,
        default=None,
        help="Override inversion model name if required by vec2text API",
    )
    parser.add_argument(
        "--corrector_model",
        type=str,
        default=None,
        help="Override corrector model name if required by vec2text API",
    )
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--truncate", type=int, default=80, help="Max chars to show per line")
    args = parser.parse_args()

    vec2text_mod = _get_vec2text()
    print(f"Using vec2text from: {getattr(vec2text_mod, '__file__', 'unknown')}")
    print(f"Loading pretrained vec2text corrector ({args.model})...")
    inversion_model = args.inversion_model or f"{args.model}-inversion"
    corrector_model = args.corrector_model or f"{args.model}-corrector"

    embedder_model = args.embedder_model or args.model
    embedder, tokenizer, embedder_errors = _load_embedder(embedder_model)
    if embedder is None and embedder_errors:
        print("Warning: embedder loader errors:")
        for err in embedder_errors:
            print(f"  - {err}")

    corrector = _load_corrector(
        args.model,
        inversion_model,
        corrector_model,
        embedder,
        tokenizer,
    )
    device = _get_device(corrector)
    if hasattr(corrector, "to"):
        corrector.to(device)
    if hasattr(corrector, "model") and hasattr(corrector.model, "eval"):
        corrector.model.eval()
    if hasattr(corrector, "inversion_trainer") and hasattr(corrector.inversion_trainer, "model"):
        if hasattr(corrector.inversion_trainer.model, "eval"):
            corrector.inversion_trainer.model.eval()

    test_cases: List[Tuple[str, str]] = [
        ("coherent_bio",
         "The mitochondria is the powerhouse of the cell. It produces ATP through oxidative "
         "phosphorylation. The electron transport chain is located in the inner membrane."),
        ("coherent_finance",
         "The stock market fell sharply on Tuesday. Investors worried about rising interest rates. "
         "The Federal Reserve signaled more rate hikes ahead."),
        ("coherent_tech",
         "Python is a high-level programming language. It supports multiple programming paradigms. "
         "The syntax emphasizes code readability."),
        ("coherent_history",
         "The Roman Empire lasted for centuries. It expanded across Europe and the Mediterranean. "
         "The fall of Rome marked the end of antiquity."),
        ("mixed_bio_finance",
         "The mitochondria produces ATP through cellular respiration. The stock market crashed "
         "yesterday amid recession fears."),
        ("mixed_tech_history",
         "Python uses dynamic typing and garbage collection. The Roman Empire fell in 476 AD when "
         "Germanic tribes invaded."),
        ("mixed_bio_tech",
         "DNA replication occurs during the S phase of the cell cycle. Machine learning models "
         "require large datasets for training."),
        ("mixed_finance_history",
         "Interest rates rose to combat inflation. The French Revolution began in 1789 with the "
         "storming of the Bastille."),
        ("messy_transcript",
         "Uh, so like, the the mitochondria is, you know, basically the powerhouse, um, of the "
         "cell, right?"),
        ("very_short",
         "Cells need energy."),
        ("three_topics",
         "DNA stores genetic information. Stock prices fell sharply. The Roman army was well organized."),
    ]

    print("\n" + "=" * 80)
    print("RECONSTRUCTION QUALITY TEST")
    print("=" * 80)

    results = []

    for name, text in test_cases:
        print(f"\n--- {name} ---")
        truncated_input = (text[: args.truncate] + "...") if len(text) > args.truncate else text
        print(f"Input:  {truncated_input}")

        reconstructed, quality = get_reconstruction_quality(
            text,
            corrector,
            num_steps=args.num_steps,
            max_length=args.max_length,
        )

        truncated_output = (
            reconstructed[: args.truncate] + "..."
            if len(reconstructed) > args.truncate
            else reconstructed
        )
        print(f"Output: {truncated_output}")
        print(f"Quality: {quality:.4f}")

        results.append((name, quality, "mixed" in name))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    def _is_edge(name: str) -> bool:
        lowered = name.lower()
        return any(token in lowered for token in ["edge", "messy", "short", "three"])

    coherent_scores = [
        q for name, q, is_mixed in results if not is_mixed and not _is_edge(name)
    ]
    mixed_scores = [q for name, q, is_mixed in results if is_mixed]

    if coherent_scores:
        print(
            f"\nCoherent windows (n={len(coherent_scores)}): "
            f"{np.mean(coherent_scores):.4f} ± {np.std(coherent_scores):.4f}"
        )
    if mixed_scores:
        print(
            f"Mixed windows (n={len(mixed_scores)}):    "
            f"{np.mean(mixed_scores):.4f} ± {np.std(mixed_scores):.4f}"
        )

    if coherent_scores and mixed_scores:
        diff = np.mean(coherent_scores) - np.mean(mixed_scores)
        print(f"\nDifference: {diff:.4f}")

        if diff > 0.05:
            print(">>> HYPOTHESIS SUPPORTED: Coherent windows reconstruct better")
        elif diff < -0.05:
            print(">>> HYPOTHESIS REJECTED: Mixed windows reconstruct better (?!)")
        else:
            print(">>> INCONCLUSIVE: No significant difference")

    print("\n" + "=" * 80)
    print("ALL RESULTS")
    print("=" * 80)
    print(f"{'Name':<25} {'Quality':>10} {'Type':<10}")
    print("-" * 50)
    for name, quality, is_mixed in sorted(results, key=lambda x: -x[1]):
        type_str = "MIXED" if is_mixed else "coherent"
        print(f"{name:<25} {quality:>10.4f} {type_str:<10}")


if __name__ == "__main__":
    main()
