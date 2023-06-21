def combine_loss(loss_dict):
    loss_dict["loss"] = 0.20 * loss_dict["loss_color"] + \
        0.80 * loss_dict["loss_depth"]  # depth is very important
    return loss_dict
