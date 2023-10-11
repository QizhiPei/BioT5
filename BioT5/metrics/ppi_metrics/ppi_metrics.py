import datasets
from datasets.config import importlib_metadata, version
import evaluate

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef,accuracy_score, precision_recall_curve, precision_score,recall_score,f1_score,cohen_kappa_score,auc
from sklearn.metrics import matthews_corrcoef,accuracy_score, precision_score,recall_score


_CITATION = """\
@article{xxx
}
"""

_DESCRIPTION = """\
Classification metrics for PPI.
"""

_KWARGS_DESCRIPTION = """
No args.
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class PPI_classification(evaluate.Metric):
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

        fpr, tpr, _= roc_curve(y_true, y_pred)
        auc1 = roc_auc_score(y_true, y_pred)
        
        for i in range(0,len(y_pred)):
            if(y_pred[i]>0.5):
                y_pred[i]=1
            else:
                y_pred[i]=0
                
        cm1=confusion_matrix(y_true,y_pred)
        acc1 = accuracy_score(y_true, y_pred, sample_weight=None)
        spec1= (cm1[0,0])/(cm1[0,0]+cm1[0,1])
        sens1 = recall_score(y_true, y_pred, sample_weight=None)
        prec1=precision_score(y_true, y_pred, sample_weight=None)
        f11=f1_score(y_true, y_pred)
        ka1=cohen_kappa_score(y_true, y_pred)

        # sensitivity1.append(sens1)
        # specificity1.append(spec1)
        # accuracy1.append(acc1)
        # precision1.append(prec1)
        # F11.append(f11)
        # kappa1.append(ka1)

        coef=matthews_corrcoef(y_true, y_pred, sample_weight=None)
        # m_coef.append(coef)
        precision12, recall12, thresholds_AUPR = precision_recall_curve(y_true,y_pred)
        AUPR = auc(recall12, precision12)
        # aupr_list.append(AUPR)


        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("ground truth\toutput\n")
            for gt, out in zip(y_true, y_pred):
                f.write(str(gt) + "\t" + str(out) + "\n")

        return {
            "accuracy": acc1,
            "specificity": spec1,
            "sensitivity": sens1,
            "precision": prec1,
            "mcc": coef,
            "auroc": auc1,
            "f1": f11, 
            "auprc": AUPR,
            "kappa": ka1,
        }
