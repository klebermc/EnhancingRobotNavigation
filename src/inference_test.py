import onnxruntime as ort
import numpy as np
import torch

obs = np.zeros((1,9)).astype(np.float32)
ort_sess = ort.InferenceSession('my_sac_actor.onnx')
outputs = ort_sess.run(None, {'input':obs})
print(outputs[0][0][0])