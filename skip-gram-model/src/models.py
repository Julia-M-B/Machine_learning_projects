from sqlalchemy import Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property

Base = declarative_base()

class PositiveSample(Base):
    """
    Tabela przechowuje pary tokenów center-context
    """
    __tablename__ = "positive_samples"
    id = Column(Integer, primary_key=True)
    center_token = Column(Integer)
    context_token = Column(Integer)

class TrainingExample(Base):
    """
    Tabela przechowuje gotowe przykłady uczące dla modelu skip gram
    """
    __tablename__ = "training_examples"
    id = Column(Integer, primary_key=True)

    target = Column(Integer)  # target token

    positive = Column(Integer)  # positive context token

    negative1 = Column(Integer)  # negative context tokens
    negative2 = Column(Integer)
    negative3 = Column(Integer)
    negative4 = Column(Integer)
    negative5 = Column(Integer)


    @hybrid_property
    def negative_tokens(self):
        return self.negative1, self.negative2, self.negative3, self.negative4, self.negative5

    @negative_tokens.setter
    def negative_tokens(self, values):
        self.negative1, self.negative2, self.negative3, self.negative4, self.negative5 = values

    @hybrid_property
    def positive_pair(self):
        return self.target, self.positive

    @positive_pair.setter
    def positive_pair(self, values):
        self.target, self.positive = values

    @hybrid_property
    def negative_pairs(self):
        return (
            (self.target, self.negative1),
            (self.target, self.negative2),
            (self.target, self.negative3),
            (self.target, self.negative4),
            (self.target, self.negative5),
                )
