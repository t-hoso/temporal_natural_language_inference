class Setting(object):
    def __init__(self):
        self.__set_names()

    def __set_names(self):
        self.MODEL_NAME_FFN = "ffn"
        self.MODEL_NAME_BERT = "bert"
        self.MODEL_NAME_EXPLAIN = "explain"
        self.MODEL_NAME_SIAMESE = "siamese"
        self.MODEL_NAME_KNOWLEDGE_ONLY = "knowledge_only"
        self.MODEL_NAME_EXPLAIN_KNOWLEDGE = "explain_knowledge"
        self.MODEL_NAME_SIAMESE_KNOWLEDGE = "siamese_knowledge"
        self.MODEL_NAME_KNOWLEDGE_ONLY_SUBTRACTION = "knowledge_only_subtraction"
        self.MODEL_NAME_KNOWLEDGE_ONLY_RELU = "knowledge_only_relu"
        self.MODEL_NAME_kNOWLEDGE_ONLY_SUBTRACTION_RELU = "knowledge_only_subtraction_relu"
        self.MODEL_NAME_KNOWLEDGE_ONLY_COMBINE_RELU = "knowledge_only_combine_relu"
        self.MODEL_NAME_TRANSE_EXPLAIN = "transe_explain"
        self.MODEL_NAME_EXPLAIN_BERT = "explain_bert"
        self.MODEL_NAME_KNOWLEDGE_EXPLAIN_BERT = "knowledge_explain_bert"

        self.LOSS_FN_CROSS_ENTROPY_LOSS = "cross_entropy_loss"
        self.LOSS_FN_SELF_EXPLAIN_LOSS = "self_explain_loss"

        self.OPTIMIZER_ADAM = "Adam"
        self.OPTIMIZER_ADAMW = "AdamW"