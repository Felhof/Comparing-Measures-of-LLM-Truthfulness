import pandas as pd

from LieDetection.lllm.questions_loaders import QuestionsLoader


SIMPLE_TQA_QUESTION_PREFIX = "You will be given a question and an answer. Answer 'True' if the answer correctly answers the question and 'False' if it does not.\n"
SIMPLE_TQA_ANSWER_PREFIX = "True/False:"

class SimpleTQA(QuestionsLoader):
    question_prefix: str = SIMPLE_TQA_QUESTION_PREFIX
    answer_prefix: str = SIMPLE_TQA_ANSWER_PREFIX

    binary_questions = True
    default_processed_filename = "simple_tqa"

    def load_from_raw(self):
        simple_tqa = pd.read_csv("data/simple_tqa.csv")
        preprocessed_questions = simple_tqa[["question", "answer"]]
        return preprocessed_questions