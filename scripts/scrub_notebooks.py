"""Nettoie les sorties de notebooks : chemins machine et bruit d'avertissements.

Ce qui est retiré :
  - les flux stderr dont le contenu est un avertissement Python ou un chemin absolu ;
  - toute occurrence résiduelle du chemin de la machine dans les autres sorties.

Ce qui est conservé :
  - les images, les résultats d'exécution, les impressions volontaires ;
  - le code, le markdown, l'ordre des cellules, les numéros d'exécution.

Le script est idempotent et rapporte ce qu'il a fait, fichier par fichier.
"""

import json
import re
import sys
from pathlib import Path

HOME = re.compile(r"/Users/[^/\s\"']+(?:/[^\s\"':]+)*")
WARNING = re.compile(
    r"\b(UserWarning|RuntimeWarning|ConvergenceWarning|FutureWarning|DeprecationWarning|warnings\.warn)\b"
)


def is_noise(output: dict) -> bool:
    """Un flux stderr fait de bruit d'avertissement, sans information propre au projet."""
    if output.get("output_type") != "stream" or output.get("name") != "stderr":
        return False
    text = "".join(output.get("text", []))
    return bool(WARNING.search(text) or HOME.search(text))


def scrub_text(value):
    if isinstance(value, str):
        return HOME.sub("<chemin local>", value)
    if isinstance(value, list):
        return [scrub_text(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub_text(v) for k, v in value.items()}
    return value


def scrub(path: Path) -> tuple[int, int, int]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    dropped = images = 0

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        kept = []
        for output in cell.get("outputs", []):
            if is_noise(output):
                dropped += 1
                continue
            if "image/png" in output.get("data", {}):
                images += 1
            kept.append(output)
        cell["outputs"] = kept

    # Les sorties conservées peuvent encore citer un chemin ; les données binaires
    # des images ne contiennent pas de texte et traversent le nettoyage intactes.
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            if "text" in output:
                output["text"] = scrub_text(output["text"])
            data = output.get("data", {})
            for key in list(data):
                if key.startswith("text/"):
                    data[key] = scrub_text(data[key])
            if "traceback" in output:
                output["traceback"] = scrub_text(output["traceback"])

    notebook["metadata"] = scrub_text(notebook.get("metadata", {}))

    serialized = json.dumps(notebook, ensure_ascii=False, indent=1)
    remaining = len(HOME.findall(serialized))
    path.write_text(serialized + "\n", encoding="utf-8")
    return dropped, images, remaining


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    total_dropped = total_images = total_remaining = 0

    for path in sorted(root.glob("*.ipynb")):
        before = path.stat().st_size
        dropped, images, remaining = scrub(path)
        after = path.stat().st_size
        total_dropped += dropped
        total_images += images
        total_remaining += remaining
        print(
            f"{path.name:48s} {before // 1024:5d} ko -> {after // 1024:4d} ko"
            f"  sorties retirées: {dropped:4d}  figures gardées: {images:3d}"
            f"  chemins restants: {remaining}"
        )

    print(
        f"\nTotal : {total_dropped} sorties retirées, {total_images} figures conservées, "
        f"{total_remaining} chemins machine restants."
    )
    return 1 if total_remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())
