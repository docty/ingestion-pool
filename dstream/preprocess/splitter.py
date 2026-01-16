from dstream.preprocess.base import  IDataSplitter
from sklearn.model_selection import train_test_split
from dstream.utils.logged import setLogging
 
logger = setLogging().getLogger("Splitter")

class SimpleDataSplitter(IDataSplitter):
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):
        logger.info("Splitting dataset into train and test sets...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            logger.info(f"Data split completed: Train={len(X_train)}, Test={len(X_test)}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error during data splitting: {e}")
            raise