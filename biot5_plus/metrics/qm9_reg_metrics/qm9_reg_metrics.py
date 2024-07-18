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
class qm9_regression(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("float", id="sequence"),
                        "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
                    }
                ),
            ],
            codebase_urls=["https://xxx.com"],
            reference_urls=[
                "https://xxx.com"
            ],
        )

    def _compute(self, predictions, references, tsv_path="tmp.tsv"):
        # y_pred = predictions
        # y_true = [ref[1] for ref in references]
        # insts = [ref[0] for ref in references]
        predictions_homo = []
        references_homo = []

        predictions_lumo = []
        references_lumo = []

        predictions_gap = []
        references_gap = []

        for pred, ref in zip(predictions, references):
            inst = ref[0]
            gt = ref[1]
            if 'HOMO-LUMO' in inst or ('HOMO' in inst and 'LUMO' in inst):
                predictions_gap.append(pred)
                references_gap.append(float(gt))
            elif 'HOMO' in inst and 'LUMO' not in inst:
                predictions_homo.append(pred)
                references_homo.append(float(gt))
            elif 'LUMO' in inst and 'HOMO' not in inst:
                predictions_lumo.append(pred)
                references_lumo.append(float(gt))
        # Function to calculate RMSE and MAE, returns None if lists are empty
        def calc_metrics(refs, preds):
            if len(refs) > 0 and len(preds) > 0:
                rmse = np.sqrt(metrics.mean_squared_error(refs, preds))
                mae = metrics.mean_absolute_error(refs, preds)
            else:
                rmse, mae = 0.0, 0.0
            return rmse, mae
        # Calculate metrics: RMSE, MAE
        rmse_gap, mae_gap = calc_metrics(references_gap, predictions_gap)
        # rmse_gap = np.sqrt(metrics.mean_squared_error(references_gap, predictions_gap))
        # mae_gap = metrics.mean_absolute_error(references_gap, predictions_gap)
        rmse_homo, mae_homo = calc_metrics(references_homo, predictions_homo)
        # rmse_homo = np.sqrt(metrics.mean_squared_error(references_homo, predictions_homo))
        # mae_homo = metrics.mean_absolute_error(references_homo, predictions_homo)
        rmse_lumo, mae_lumo = calc_metrics(references_lumo, predictions_lumo)
        # rmse_lumo = np.sqrt(metrics.mean_squared_error(references_lumo, predictions_lumo))
        # mae_lumo = metrics.mean_absolute_error(references_lumo, predictions_lumo)
        # with open(tsv_path, "w", encoding="utf-8") as f:
        #     f.write("ground truth\toutput\n")
        #     for gt, out in zip(y_true, y_pred):
        #         f.write(str(gt) + "\t" + str(out) + "\n")
        references_all = references_homo + references_lumo + references_gap
        predictions_all = predictions_homo + predictions_lumo + predictions_gap
        rmse_all, mae_all = calc_metrics(references_all, predictions_all)
        # rmse_all = np.sqrt(metrics.mean_squared_error(references_all, predictions_all))
        # mae_all = metrics.mean_absolute_error(references_all, predictions_all)

        return {
            "rmse_homo": rmse_homo,
            "mae_homo": mae_homo,
            "rmse_lumo": rmse_lumo,
            "mae_lumo": mae_lumo,
            "rmse_gap": rmse_gap,
            "mae_gap": mae_gap,
            "rmse_all": rmse_all,
            "mae_all": mae_all
        }
