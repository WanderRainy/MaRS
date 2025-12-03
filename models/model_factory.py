import torch.nn as nn
from models.MaRS import build_mars

def get_model(model_str, norm_pix_loss=True):
    # NOTE: we might first check the model_str
    if model_str in globals().keys():
        model = globals()[model_str](norm_pix_loss=norm_pix_loss)
    elif model_str in ['mars_base', 'mars_large']:
        if model_str == 'mars_base':
            model = build_mars(model_type='base', contrast_type='cross')
        elif model_str == 'mars_large':
            model = build_mars(model_type='large', contrast_type='cross')
    else:
        raise KeyError(f"Model `{model_str}` is not suuported.")
    return model


