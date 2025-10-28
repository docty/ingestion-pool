from sklearn.datasets import load_diabetes, load_iris

def regression_dataset():
  df = load_diabetes(as_frame=True)
  return df.frame


def classification_dataset():
  df = load_iris(as_frame=True)
  return df.frame 