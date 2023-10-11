import datasets
from datasets.config import importlib_metadata, version
import evaluate

import numpy as np

from transformers import BertTokenizerFast

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

NLTK_VERSION = version.parse(importlib_metadata.version("nltk"))

_CITATION = """\
@article{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Ji, Heng},
  journal={arXiv preprint arXiv:2204.11817},
  year={2022}
}
"""

_DESCRIPTION = """\
Translation metrics for molecule to text task.
"""

_KWARGS_DESCRIPTION = """
Args:
    text_model: desired language model tokenizer.
    text_trunc_lenght: tokenizer maximum length
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Mol2Text_translation(evaluate.Metric):
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
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=["https://github.com/blender-nlp/MolT5/blob/main/evaluation/text_translation_metrics.py"],
            reference_urls=[
                "https://github.com/blender-nlp/MolT5"
            ],
        )

    def _download_and_prepare(self, dl_manager):
        import nltk

        nltk.download("wordnet")
        if NLTK_VERSION >= version.Version("3.6.5"):
            nltk.download("punkt")
        if NLTK_VERSION >= version.Version("3.6.6"):
            nltk.download("omw-1.4")

    def _compute(self, predictions, references, text_model='allenai/scibert_scivocab_uncased', text_trunc_length=512, tsv_path="tmp.tsv"):
        inputs = [references[i][1] for i in range(len(references))]
        references = [references[i][0] for i in range(len(references))]
        text_tokenizer = BertTokenizerFast.from_pretrained(text_model)

        meteor_scores = []

        refs = []
        preds = []

        for gt, out in zip(references, predictions):

            gt_tokens = text_tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                                padding='max_length')
            gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
            gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
            gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

            out_tokens = text_tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                                padding='max_length')
            out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
            out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
            out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

            refs.append([gt_tokens])
            preds.append(out_tokens)

            mscore = meteor_score([gt_tokens], out_tokens)
            meteor_scores.append(mscore)

        bleu2 = corpus_bleu(refs, preds, weights=(.5,.5))
        bleu4 = corpus_bleu(refs, preds, weights=(.25,.25,.25,.25))

        _meteor_score = np.mean(meteor_scores)

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

        rouge_scores = []

        refs = []
        preds = []

        for gt, out in zip(references, predictions):

            rs = scorer.score(out, gt)
            rouge_scores.append(rs)

        rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
        rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
        rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])

        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("SMILES\tground truth\toutput\n")
            for it, gt, out in zip(inputs, references, predictions):
                f.write(it + "\t" + gt + "\t" + out + "\n")

        return {
            "bleu2": bleu2,
            "bleu4": bleu4,
            "rouge1": rouge_1,
            "rouge2": rouge_2,
            "rougeL": rouge_l,
            "meteor": _meteor_score,
        }