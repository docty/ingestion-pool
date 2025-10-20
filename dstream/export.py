import numpy as np
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_model_to_onnx(model, X_train, filename="model.onnx", opset=12):
    """
    Converts and exports a trained scikit-learn model to ONNX format.
    """
    initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=opset)
    with open(filename, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"Model exported to {filename}")


def predict_with_onnx(filename, X_sample):
    """
    Loads an ONNX model and predicts the output for a given input sample.
    """
    sess = rt.InferenceSession(filename, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    sample = X_sample.astype(np.float32).reshape(1, -1)
    pred_onx = sess.run([label_name], {input_name: sample})[0]
    return int(pred_onx[0])