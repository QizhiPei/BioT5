import datasets
from datasets.config import importlib_metadata, version
import evaluate
from rdkit import Chem
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
class Text2Mol_translation(evaluate.Metric):
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
            codebase_urls=["https://xxx.com"],
            reference_urls=[
                "https://xxx.com"
            ],
        )

    def _download_and_prepare(self, dl_manager):
        import nltk

        nltk.download("wordnet")
        if NLTK_VERSION >= version.Version("3.6.5"):
            nltk.download("punkt")
        if NLTK_VERSION >= version.Version("3.6.6"):
            nltk.download("omw-1.4")

    def _compute(self, predictions, references, tsv_path="tmp.tsv", verbose=False):
        inputs = [references[i][1] for i in range(len(references))]
        references = [references[i][0] for i in range(len(references))]
        outputs = []

        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("description\tground truth\toutput\n")
            for it, gt, out in zip(inputs, references, predictions):
                f.write(it + "\t" + gt + "\t" + out + "\n")
                outputs.append((it, gt, out))
        
        RDLogger.DisableLog('rdApp.*')
        bleu_scores = []
        #meteor_scores = []

        references = []
        hypotheses = []

        for i, (smi, gt, out) in enumerate(outputs):

            if i % 100 == 0:
                if verbose:
                    print(i, 'processed.')


            gt_tokens = [c for c in gt]

            out_tokens = [c for c in out]

            references.append([gt_tokens])
            hypotheses.append(out_tokens)

            # mscore = meteor_score([gt], out)
            # meteor_scores.append(mscore)

        # BLEU score
        bleu_score = corpus_bleu(references, hypotheses)
        if verbose: print('BLEU score:', bleu_score)

        # Meteor score
        # _meteor_score = np.mean(meteor_scores)
        # print('Average Meteor score:', _meteor_score)

        rouge_scores = []

        references = []
        hypotheses = []

        levs = []

        num_exact = 0

        bad_mols = 0

        for i, (smi, gt, out) in enumerate(outputs):

            hypotheses.append(out)
            references.append(gt)

            try:
                m_out = Chem.MolFromSmiles(out)
                m_gt = Chem.MolFromSmiles(gt)

                if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt): num_exact += 1
                #if gt == out: num_exact += 1 #old version that didn't standardize strings
            except:
                bad_mols += 1

            

            levs.append(lev(out, gt))


        # Exact matching score
        exact_match_score = num_exact/(i+1)
        if verbose:
            print('Exact Match:')
            print(exact_match_score)

        # Levenshtein score
        levenshtein_score = np.mean(levs)
        if verbose:
            print('Levenshtein:')
            print(levenshtein_score)
            
        validity_score = 1 - bad_mols/len(outputs)
        if verbose:
            print('bad mols:', bad_mols)
            print('validity:', validity_score)

        return {
            'bleu': bleu_score,
            'exact_match': exact_match_score,
            'levenshtein': levenshtein_score,
            'validity': validity_score
        }