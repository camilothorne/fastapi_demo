from fastapi import FastAPI
from bert_inference import ChemClassifier
from pydantic import BaseModel


app = FastAPI()
chem = ChemClassifier()


class Token(BaseModel):
    tokens: list[str]
    table:  str


class Pred(BaseModel):
    prob: float
    label: str
    input: str


# Home Page
@app.get("/")
def read_root():
    return {"FastAPI Demo v1.0": {"Functions":["/classify", "/tokenize"]}}


# Classify
@app.get("/classify/{input_sentence}")
async def read_item(input_sentence: str) -> Pred:
    result = chem.get_sentence_class(input_sentence)
    return result 


# Tokenize
@app.get("/tokenize/{input_sentence}")
async def read_item(input_sentence: str) -> Token:
    toks, tab  = chem.get_sentence_encoding(input_sentence)
    return Token(tokens=toks, table=tab) 