from typing import Union, Optional

from pydantic import BaseModel


class F1(BaseModel):
    n_correct: Optional[int] = None
    n_pos_gold: Optional[int] = None
    n_pos_pred: Optional[int] = None

    def __str__(self):
        return f"F1={self.f1:.4f}, Prec={self.prec:.4f}, Rec={self.rec:.4f}, #correct={self.n_correct}, #pos_gold={self.n_pos_gold}, #pos_pred={self.n_pos_pred}"

    def __add__(self, other: "F1") -> "F1":
        if isinstance(other, F1):
            # Handle None values - treat None as 0 for addition
            self_correct = self.n_correct if self.n_correct is not None else 0
            self_gold = self.n_pos_gold if self.n_pos_gold is not None else 0
            self_pred = self.n_pos_pred if self.n_pos_pred is not None else 0
            
            other_correct = other.n_correct if other.n_correct is not None else 0
            other_gold = other.n_pos_gold if other.n_pos_gold is not None else 0
            other_pred = other.n_pos_pred if other.n_pos_pred is not None else 0
            
            return F1(
                n_correct=self_correct + other_correct,
                n_pos_gold=self_gold + other_gold,
                n_pos_pred=self_pred + other_pred,
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
    def valid(self) -> bool:
        return self.n_correct is not None and self.n_pos_gold is not None and self.n_pos_pred is not None

    @property
    def prec(self):
        if not self.valid:
            return 0.0
        if self.n_pos_pred == 0:
            return 1.0 if self.n_pos_gold == 0 else 0.0
        return self.n_correct / self.n_pos_pred

    @property
    def rec(self):
        if not self.valid:
            return 0.0
        if self.n_pos_gold == 0:
            return 1.0 if self.n_pos_pred == 0 else 0.0
        return self.n_correct / self.n_pos_gold

    @property
    def f1(self):
        if not self.valid:
            return 0.0
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
