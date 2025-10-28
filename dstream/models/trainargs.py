from dataclasses import dataclass


@dataclass
class TrainingArguments:
    output_dir="./output"
    report_to="none"


 