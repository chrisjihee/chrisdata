from pydantic import BaseModel


class F1(BaseModel):
    n_correct: int = 0
    n_pos_gold: int = 0
    n_pos_pred: int = 0

    def __str__(self):
        return f"F1={self.f1:.4f}, Prec={self.prec:.4f}, Rec={self.rec:.4f}, #correct={self.n_correct}, #pos_gold={self.n_pos_gold}, #pos_pred={self.n_pos_pred}"

    def __add__(self, other: "F1") -> "F1":
        if not isinstance(other, F1):
            return NotImplemented
        return F1(
            n_correct=self.n_correct + other.n_correct,
            n_pos_gold=self.n_pos_gold + other.n_pos_gold,
            n_pos_pred=self.n_pos_pred + other.n_pos_pred,
        )

    @property
    def prec(self):
        return self.n_correct / (self.n_pos_pred + 1e-10)

    @property
    def rec(self):
        return self.n_correct / (self.n_pos_gold + 1e-10)

    @property
    def f1(self):
        return 2 * self.prec * self.rec / (self.prec + self.rec + 1e-10)


class RegressionSample(BaseModel):
    sentence1: str
    sentence2: str
    label: float
    idx: int
