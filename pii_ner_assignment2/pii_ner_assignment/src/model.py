from transformers import AutoConfig, AutoModelForTokenClassification
from labels import LABEL2ID, ID2LABEL


def create_model(model_name: str):
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=0.2,          # more regularization
        attention_probs_dropout_prob=0.1, # field used by BERT/DeBERTa
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
    )
    return model
