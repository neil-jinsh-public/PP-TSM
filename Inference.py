import os
import paddle
import argparse
import subprocess
import time

import paddle.nn.functional as F
from Module import ResNetTweaksTSM, ppTSMHead, Recognizer2D
import warnings
from Preprocessing import preprocessing

warnings.filterwarnings("ignore")

# ========================= Model Configs ==========================
parser = argparse.ArgumentParser(description="Model Configs")
parser.add_argument('--depth', type=str, default=50)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--in_channels', type=int, default=2048)
parser.add_argument('--drop_ratio', type=float, default=0.5)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--ls_eps', type=float, default=0.1)
obj = parser.parse_args()


# =========================== Inference ============================
def inference(video_file_path):
    model_file = 'model/inference_para.pdparams'

    pptsm = ResNetTweaksTSM(pretrained=None, depth=obj.depth)
    head = ppTSMHead(num_classes=obj.num_classes,
                     in_channels=obj.in_channels,
                     drop_ratio=obj.drop_ratio,
                     std=obj.std,
                     ls_eps=obj.ls_eps)

    model = Recognizer2D(backbone=pptsm, head=head)
    model.eval()
    state_dicts = paddle.load(model_file)
    model.set_state_dict(state_dicts)

    # video prepocessing
    data = preprocessing(video_file_path)
    outputs = model.infer_step(data)
    scores = F.softmax(outputs)
    class_id = paddle.argmax(scores, axis=-1)
    pred = class_id.numpy()[0]

    return pred

# =========================== Del video ============================


# ============================== Main ===============================
def main():

    path_temp = os.getcwd()
    cmd = 'cd /D {} && python Video_save.py'.format(path_temp)
    obj = subprocess.Popen(cmd, shell=True,
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    obj.communicate()

    path = 'Video'
    file_name_list = os.listdir(path)
    try:
        video_file_path = os.path.join(path, file_name_list[-1])
    except IndexError as reason:
        print('Error is {}'.format(reason))
    else:
        flag = inference(video_file_path)
        if flag:
            print("Stagnant water")
        else:
            print("No Stagnant water")


if __name__ == '__main__':
    main()

