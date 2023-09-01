import numpy as np


def prefilter_masks(
        masks,
        scores,
        labels,
        weights,
        thr
):
    new_masks = dict()

    for model in range(len(masks)):
        if len(masks[model]) != len(scores[model]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(masks[model]), len(scores[model])))
            exit()
        if len(masks[model]) != len(labels[model]):
            print('Error. Length of boxes arrays not equal to length of labels array: {} != {}'.format(len(masks[model]), len(labels[model])))
            exit()
        for ins in range(len(masks[model])):
            score = scores[model][ins]
            if score < thr:
                continue
            label = int(labels[model][ins])
            mask = masks[model][ins]
            b = [int(label), float(score) * weights[model], weights[model], model, mask]
            if label not in new_masks:
                new_masks[label] = []
            new_masks[label].append(b)

    # for k in new_masks:
    #     # current_masks = np.array(new_masks[k])
    #     new_masks[k] = new_masks[k][new_masks[k][:, 1].argsort()[::-1]]
    
    return new_masks



def weighted_masks_fusion(
        masks_list,
        scores_list,
        labels_list,
        weights=None,
        iou_thr=0.7,
        skip_mask_thr=0.0,
        conf_type='avg',
        allows_overflow=False
):
    if weights is None:
        weights = np.ones(len(masks_list))
    if len(weights) != len(masks_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(masks_list)))
        exit()
    weights = np.array(weights)

    if conf_type not in ['avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg']:
        print('Unknown conf_type: {}. Must be "avg", "max" or "box_and_model_avg", or "absent_model_aware_avg"'.format(conf_type))
        exit()
    
    filter_masks = prefilter_masks(masks_list, scores_list, labels_list, weights, skip_mask_thr)
    if len(filter_masks) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))



if __name__=='__main__':
    np.random.seed(0)
    masks_list = np.random.randint(0, 2, (2, 5, 3, 3), dtype='int')
    scores_list = np.random.random((2, 5))
    labels_list = np.random.randint(0, 2, (2, 5), dtype='int')
    weights = [0.5, 2.0]
    iou_thr = 0.7
    skip_mask_thr = 0.8
    masks, scores, labels = weighted_masks_fusion(masks_list, scores_list, labels_list, weights, iou_thr, skip_mask_thr)
