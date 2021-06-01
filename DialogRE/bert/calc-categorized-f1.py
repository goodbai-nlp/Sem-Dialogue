# coding:utf-8
def evaluate(devp, data):
    index = 0
    correct_sys, all_sys = 0, 0
    correct_gt = 0

    for i in range(len(data)):
        for j in range(len(data[i][1])):        # K 
            for id in data[i][1][j]["rid"]:
                if id != 36:
                    correct_gt += 1
                    if id in devp[index]:
                        correct_sys += 1
            for id in devp[index]:
                if id != 36:
                    all_sys += 1
            index += 1

    precision = correct_sys / all_sys if all_sys != 0 else 1
    recall = correct_sys / correct_gt if correct_gt != 0 else 0
    f_1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    return precision, recall, f_1

def evaluate_new(devp, ref):
    index = 0
    correct_sys, all_sys = 0, 0
    correct_gt = 0
    assert len(devp) == len(ref)
    for i in range(len(data)):
        for id in data[i]:
           if id != 36:
                correct_gt += 1
                if id in devp[index]:
                    correct_sys += 1
        for id in devp[i]:
            if id != 36:
                all_sys += 1

    precision = correct_sys / all_sys if all_sys != 0 else 1
    recall = correct_sys / correct_gt if correct_gt != 0 else 0
    f_1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    return precision, recall, f_1
