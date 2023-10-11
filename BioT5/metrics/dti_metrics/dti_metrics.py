import datasets
from datasets.config import importlib_metadata, version
import evaluate

import numpy as np
from sklearn import metrics
import random

_CITATION = """\
@article{xxx
}
"""

_DESCRIPTION = """\
Classification metrics for DTI.
"""

_KWARGS_DESCRIPTION = """
No args.
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class DTI_classification(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("float", id="sequence"),
                        "references": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            codebase_urls=["https://xxx.com"],
            reference_urls=[
                "https://xxx.com"
            ],
        )

    def _compute(self, predictions, references, tsv_path="tmp.tsv"):
        y_pred = predictions
        y_true = [1 if ref == 'Yes.' else 0 for ref in references]

        auroc = metrics.roc_auc_score(y_true, y_pred)
        auprc = metrics.average_precision_score(y_true, y_pred)

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        prec, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
        # different with DrugBAN
        precision = tpr / (tpr + fpr + 0.00001)
        f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
        thred_optim = thresholds[5:][np.argmax(f1[5:])]
        y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
        cm1 = metrics.confusion_matrix(y_true, y_pred_s)
        accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
        sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
        specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        precision1 = metrics.precision_score(y_true, y_pred_s)

        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("ground truth\toutput\n")
            for gt, out in zip(y_true, y_pred):
                f.write(str(gt) + "\t" + str(out) + "\n")

        return {
            "accuracy": accuracy,
            "auroc": auroc,
            "auprc": auprc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": np.max(f1[5:]),
            "thred_optim": thred_optim,
            "precision": precision1,
        }
