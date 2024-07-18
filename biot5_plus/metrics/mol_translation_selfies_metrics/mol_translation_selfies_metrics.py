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
class Mol_translation_selfies(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
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

    def _download_and_prepare(self, dl_manager):
        import nltk

        nltk.download("wordnet")
        if NLTK_VERSION >= version.Version("3.6.5"):
            nltk.download("punkt")
        if NLTK_VERSION >= version.Version("3.6.6"):
            nltk.download("omw-1.4")

    def _compute(self, predictions, references, tsv_path="tmp.tsv", verbose=False):
        def convert_to_canonical_smiles(smiles):
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is not None:
                canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True)
                return canonical_smiles
            else:
                return None
            
        inputs = [references[i][0] for i in range(len(references))]
        references_smi = [references[i][1] for i in range(len(references))]
        references_self = [references[i][2] for i in range(len(references))]
        predictions_smi = [predictions[i][0] for i in range(len(predictions))]
        predictions_self = [predictions[i][1] for i in range(len(predictions))]
        outputs = []
        bad_mols = 0
        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("description\tground truth\toutput\tgt_selfies\toutput_selfies\n")
            for desc, gt_smi, out_smi, gt_self, out_self in zip(inputs, references_smi, predictions_smi, references_self, predictions_self):
                gt_smi = convert_to_canonical_smiles(gt_smi)
                out_smi = convert_to_canonical_smiles(out_smi)
                if out_smi is None:
                    bad_mols += 1
                    continue
                if gt_smi is None: 
                    continue
                f.write(desc + "\t" + gt_smi + "\t" + out_smi + "\t" + gt_self + "\t" + out_self + "\n")
                outputs.append((desc, gt_smi, out_smi, gt_self, out_self))
        
        RDLogger.DisableLog('rdApp.*')
        bleu_scores = []
        #meteor_scores = []

        references_self = []
        hypotheses_self = []

        references_smi = []
        hypotheses_smi = []

        for i, (desc, gt_smi, out_smi, gt_self, out_self) in enumerate(outputs):

            if i % 100 == 0:
                if verbose:
                    print(i, 'processed.')


            gt_self_tokens = [c for c in gt_self]
            out_self_tokens = [c for c in out_self]

            references_self.append([gt_self_tokens])
            hypotheses_self.append(out_self_tokens)

            gt_smi_tokens = [c for c in gt_smi]
            out_smi_tokens = [c for c in out_smi]

            references_smi.append([gt_smi_tokens])
            hypotheses_smi.append(out_smi_tokens)

            # mscore = meteor_score([gt], out)
            # meteor_scores.append(mscore)

        # BLEU score
        bleu_score_smi = corpus_bleu(references_smi, hypotheses_smi)
        bleu_score_self = corpus_bleu(references_self, hypotheses_self)
        if verbose: 
            print('SMILES BLEU score:', bleu_score_smi)
            print('SELFIES BLEU score:', bleu_score_self)

        # Meteor score
        # _meteor_score = np.mean(meteor_scores)
        # print('Average Meteor score:', _meteor_score)

        # rouge_scores = []

        references_self = []
        hypotheses_self = []
        
        references_smi = []
        hypotheses_smi = []

        levs = []

        num_exact = 0

        # bad_mols = 0
        outputs_rdkit_mols = []

        for i, (desc, gt_smi, out_smi, gt_self, out_self) in enumerate(outputs):

            hypotheses_self.append(out_self)
            references_self.append(gt_self)

            hypotheses_smi.append(out_smi)
            references_smi.append(gt_smi)

            try:
                m_out = Chem.MolFromSmiles(out_smi)
                m_gt = Chem.MolFromSmiles(gt_smi)

                if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt): num_exact += 1
                #if gt == out: num_exact += 1 #old version that didn't standardize strings
                outputs_rdkit_mols.append((m_gt, m_out))
            except:
                bad_mols += 1 # no effect as we have already removed bad mols

            
            levs.append(lev(out_smi, gt_smi))


        # Exact matching score
        exact_match_score = num_exact/(i+1)
        if verbose:
            print('Exact Match:')
            print(exact_match_score)

        # Levenshtein score
        levenshtein_score = np.mean(levs)
        if verbose:
            print('SMILES Levenshtein:')
            print(levenshtein_score)
            
        validity_score = 1 - bad_mols/len(outputs)
        if verbose:
            print('bad mols:', bad_mols)
            print('validity:', validity_score)

        MACCS_sims = []
        morgan_sims = []
        RDK_sims = []

        enum_list = outputs_rdkit_mols
        morgan_r = 2

        for i, (gt_m, ot_m) in enumerate(enum_list):

            if i % 100 == 0:
                if verbose: print(i, 'processed.')

            MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity))
            RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity))
            morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m,morgan_r), AllChem.GetMorganFingerprint(ot_m, morgan_r)))

        maccs_sims_score = np.mean(MACCS_sims)
        rdk_sims_score = np.mean(RDK_sims)
        morgan_sims_score = np.mean(morgan_sims)
        if verbose:
            print('Average MACCS Similarity:', maccs_sims_score)
            print('Average RDK Similarity:', rdk_sims_score)
            print('Average Morgan Similarity:', morgan_sims_score)

        return {
            'bleu_smi': bleu_score_smi,
            'bleu_self': bleu_score_self,
            'exact_match': exact_match_score,
            'levenshtein_smi': levenshtein_score,
            'validity': validity_score,
            'maccs_sims': maccs_sims_score,
            'rdk_sims': rdk_sims_score,
            'morgan_sims': morgan_sims_score,
        }