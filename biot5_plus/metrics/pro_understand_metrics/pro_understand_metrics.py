import datasets
from datasets.config import importlib_metadata, version
import evaluate
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from Levenshtein import distance as lev
from rdkit import RDLogger

NLTK_VERSION = version.parse(importlib_metadata.version("nltk"))

_CITATION = """\
@article{xxx
}
"""

_DESCRIPTION = """\
Save results.
"""

_KWARGS_DESCRIPTION = """
No args.
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Pro_understand(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
                    }
                ),
                # datasets.Features(
                #     {
                #         "predictions": datasets.Value("string", id="sequence"),
                #         "references": datasets.Value("string", id="sequence"),
                #     }
                # ),
            ],
            codebase_urls=["https://xxx.com"],
            reference_urls=[
                "https://xxx.com"
            ],
        )

    def _compute(self, predictions, references, tsv_path="tmp.tsv"):
        rouge = evaluate.load("rouge")
        references_new = [ref[0] for ref in references]
        inputs = [ref[1] for ref in references]

        rouge_scores = rouge.compute(predictions=predictions, references=references)
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("input\tground truth\toutput\n")
            for it, gt, out in zip(inputs, references_new, predictions):
                f.write(it + "\t" + gt + "\t" + out + "\n")

        return {
            'rougeL': rouge_scores['rougeL'],
        }