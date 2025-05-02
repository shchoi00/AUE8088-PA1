from torchmetrics import Metric
import torch

EPS = 1e-8


class MyF1Score(Metric):
    def __init__(self, num_classes: int, average: str):
        super().__init__()
        self.num_classes = num_classes
        self.average = average
        self.add_state(
            "conf_mat",
            default=torch.zeros(num_classes, num_classes, dtype=torch.int64),
            dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds: logits or probabilities (B x C)
        preds = preds.argmax(dim=1)
        preds = preds.view(-1)
        target = target.view(-1)
        if preds.shape != target.shape:
            raise Exception(f"Tensors have different shapes. preds: {preds.shape}, target: {target.shape}")
        # Update confusion matrix
        for t, p in zip(target, preds):
            self.conf_mat[t.long(), p.long()] += 1

    def compute(self) -> torch.Tensor:
        # Compute per-class F1 (one-vs-rest for each class)
        conf = self.conf_mat.float()
        TP = torch.diag(conf)
        FP = conf.sum(dim=0) - TP
        FN = conf.sum(dim=1) - TP

        f1 = 2 * TP / (2 * TP + FP + FN + EPS)

        if self.average == "macro":
            return f1.mean()
        elif self.average == "micro":
            total_TP = TP.sum()
            total_FP = FP.sum()
            total_FN = FN.sum()
            return 2 * total_TP / (2 * total_TP + total_FP + total_FN + EPS)
        elif self.average == "weighted":
            support = conf.sum(dim=1)
            weights = support / (support.sum() + EPS)
            return (f1 * weights).sum()
        elif self.average == "none" or self.average is None:
            return f1
        else:
            raise ValueError(f"Unknown average type: {self.average}")



class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target:torch.Tensor):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = preds.max(dim=1)[1]
        
        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            raise Exception(f"Tensors have different shapes. preds: {preds.shape}, target: {target.shape}")


        # [TODO] Cound the number of correct prediction
        correct = preds.eq(target)
        correct = torch.sum(correct)
        
        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total.float()
