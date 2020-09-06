import random
import numpy as np
# from sklearn
# from sklearn.metrics import mean_squared_error


# TODO
# 1. keypoint간 difference 계산 (순서대로 나열되도록. 18C2=153개)
# 2. difference를 이용해 객체 간 RMSE 계산
# 3. 뽑히지 않은 keypoint에 대해서는 difference를 0으로 처리할 필요가 있음.

def get_rmse_btw_detected_objs(clustering_info):
    outlier = np.array([0.0, 0.0])
    detected_num = len(clustering_info)
    
    """Ignore undetected points"""
    for i, each_array in enumerate(clustering_info):
        outlier_pos = np.where(each_array == outlier)

        for j, another_array in enumerate(clustering_info):
            if j == i:
                break
            else:
                another_array[outlier_pos] = 0.0

    """Calculate difference between points in the array"""
    each_array_diff_feature = []
    for i, each_array in enumerate(clustering_info):
        diff_feature_tmp = []
        for j, each_row in enumerate(each_array):
            for k, another_row in enumerate(each_array):
                if not k > j:
                    continue
                dist = np.linalg.norm(each_row - another_row)
                diff_feature_tmp.append(dist)
        each_array_diff_feature.append(np.array(diff_feature_tmp))


    """RMSE of each vectors"""
    rmse_dict = {}
    for i, each_array in enumerate(each_array_diff_feature):
        for j, another_array in enumerate(each_array_diff_feature):
            if not j > i:
                continue
            rmse_val = np.sqrt(np.mean((each_array - another_array) ** 2))
            rmse_dict[f'{i}_{j}'] = rmse_val

    # print(rmse_dict)

    return rmse_dict, each_array_diff_feature

                

if __name__ == "__main__":
    # clustering_info 내 element 순서
    # ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "neck"]
    a = np.array([[0.32492905855178833, 0.501892626285553], [0.33310380578041077, 0.4880438446998596], [0.3172435760498047, 0.4852469861507416], [0.34462684392929077, 0.4728955626487732], [0.30753883719444275, 0.4686611294746399], [0.3610004782676697, 0.5396445989608765], [0.2846807837486267, 0.5242716670036316], [0.3714495599269867, 0.6368532776832581], [0.27145498991012573, 0.6120590567588806], [0.37712714076042175, 0.7311155200004578], [0.2755839228630066, 0.6737926602363586], [0.35210663080215454, 0.7016264796257019], [0.30599188804626465, 0.6999101042747498], [0.35864901542663574, 0.8164647817611694], [0.30819612741470337, 0.8178834915161133], [0.3533784747123718, 0.9173151254653931], [0.3239937722682953, 0.9195994734764099], [0.3232930600643158, 0.5320881605148315]])
    b = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.5448254942893982, 0.5491952300071716], [0.5169099569320679, 0.5717753171920776], [0.5317429900169373, 0.5782879590988159], [0.5602002739906311, 0.6473382115364075], [0.5552582144737244, 0.6471742391586304], [0.5898007154464722, 0.7076377868652344], [0.5884667038917542, 0.7101715207099915], [0.49416595697402954, 0.6712039709091187], [0.4950810968875885, 0.6748852729797363], [0.5244903564453125, 0.7593855261802673], [0.5438936948776245, 0.7431986927986145], [0.4927375018596649, 0.8380841612815857], [0.509762167930603, 0.8592588305473328], [0.5249126553535461, 0.5750530958175659]])

    # c = np.random.rand(18, 2)


    clustering_info = [a, b]

    get_rmse_btw_detected_objs(clustering_info, 224, 224)