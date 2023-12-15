import pandas as pd

from LieDetection.lllm.questions_loaders import QuestionsLoader


SIMPLE_FACTS_QUESTION_PREFIX = "Is the following statement true or false? Answer only 'True' or 'False'.\nStatement:"
SIMPLE_FACTS_ANSWER_PREFIX = "True/False:"

class SimpleFacts(QuestionsLoader):
    question_prefix: str = SIMPLE_FACTS_QUESTION_PREFIX
    answer_prefix: str = SIMPLE_FACTS_ANSWER_PREFIX

    binary_questions = True
    default_processed_filename = "simple_facts"

    def load_from_raw(self):
        facts_data = pd.read_csv("RepresentationEngineering/data/facts/facts_true_false.csv")

        simple_facts = pd.DataFrame({
            "question": facts_data["statement"]
        })
        simple_facts["answer"] = facts_data["label"].apply(
            lambda label: "True" if label else "False"
        )
        return simple_facts