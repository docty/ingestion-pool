from datasets import DatasetDict, Dataset 
import pandas as pd
from dstream.preprocess.utils import setLogging
 
logger = setLogging()

class HuggingFaceDataset:
    @staticmethod
    def convert_input_label(example):
        feature_columns = [k for k in example.keys() if k != "target"]
        return {"input": [example[f] for f in feature_columns], "label": example["target"]}

    def to_huggingface_dataset(self, X_train, X_test, y_train, y_test) -> DatasetDict:
        logger.info("Converting data to Hugging Face Dataset format...")
        try:
            train_df = pd.concat([X_train, y_train.rename("target")], axis=1)
            test_df = pd.concat([X_test, y_test.rename("target")], axis=1)

            dataset_dict = DatasetDict({
                "train": Dataset.from_pandas(train_df, preserve_index=False),
                "test": Dataset.from_pandas(test_df, preserve_index=False),
            })

            new_data = dataset_dict.map(
                self.convert_input_label,
                remove_columns=dataset_dict['train'].features,
            )
            logger.info("Conversion to Hugging Face Dataset complete.")
            return new_data
        except Exception as e:
            logger.error(f"Error during Hugging Face conversion: {e}")
            raise