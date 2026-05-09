def compute_scale_pos_weight(y_train):
    return (
        y_train.value_counts()[0] /
        y_train.value_counts()[1]
    )