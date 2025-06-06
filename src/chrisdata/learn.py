from typing import Union

from pydantic import BaseModel


class F1(BaseModel):
    n_correct: int = 0
    n_pos_gold: int = 0
    n_pos_pred: int = 0

    def __str__(self):
        return f"F1={self.f1:.4f}, Prec={self.prec:.4f}, Rec={self.rec:.4f}, #correct={self.n_correct}, #pos_gold={self.n_pos_gold}, #pos_pred={self.n_pos_pred}"

    def __add__(self, other: "F1") -> "F1":
        if isinstance(other, F1):
            return F1(
                n_correct=self.n_correct + other.n_correct,
                n_pos_gold=self.n_pos_gold + other.n_pos_gold,
                n_pos_pred=self.n_pos_pred + other.n_pos_pred,
            )
        return NotImplemented

    def __eq__(self, other: object) -> bool:
        if isinstance(other, F1):
            return self.f1 == other.f1
        return NotImplemented

    def __lt__(self, other: "F1") -> bool:
        if isinstance(other, F1):
            return self.f1 < other.f1
        return NotImplemented

    @property
    def prec(self):
        if self.n_pos_pred == 0:
            return 1.0 if self.n_pos_gold == 0 else 0.0
        return self.n_correct / self.n_pos_pred

    @property
    def rec(self):
        if self.n_pos_gold == 0:
            return 1.0 if self.n_pos_pred == 0 else 0.0
        return self.n_correct / self.n_pos_gold

    @property
    def f1(self):
        if self.n_pos_gold == 0 and self.n_pos_pred == 0:
            return 1.0
        return 2 * self.prec * self.rec / (self.prec + self.rec + 1e-10)


class Sum(BaseModel):
    count: int = 0
    sum: float = 0.0

    def __str__(self):
        return f"Avg={self.avg:.1f}, Sum={self.sum:.0f}, Count={self.count}"

    def __add__(self, other: Union[int, float, "Sum"]) -> "Sum":
        if isinstance(other, (int, float)):
            return Sum(
                count=self.count + 1,
                sum=self.sum + other
            )
        elif isinstance(other, Sum):
            return Sum(
                count=self.count + other.count,
                sum=self.sum + other.sum
            )
        else:
            return NotImplemented

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0


class RegressionSample(BaseModel):
    sentence1: str
    sentence2: str
    label: float
    id: str
