import pretrained_microscopy_models as pmm
import segmentation_models_pytorch as smp
from torch.hub import load_state_dict_from_url

for encoder in [
    "efficientnet-b4",
    "inceptionv4",
    "resnet50",
    "se_resnext101_32x4d",
    "senet154",
    "vgg16_bn",
]:
    try:
        smp.encoders.get_encoder(encoder, weights="imagenet")
    except Exception as e:
        print(f"Skipping imagenet weights for {encoder}: {e}")

    for pretrained_weights in ["micronet", "image-micronet"]:
        url = pmm.util.get_pretrained_microscopynet_url(encoder, pretrained_weights)
        try:
            load_state_dict_from_url(url, map_location="cpu")
        except Exception as e:
            print(f"Skipping {pretrained_weights} weights for {encoder}: {e}")
