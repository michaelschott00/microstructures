import segmentation_models_pytorch as smp
from torch.hub import load_state_dict_from_url


def get_pretrained_microscopynet_url(
    encoder, encoder_weights, version=1.1, self_supervision=""
):
    """
    Get the url to download the specified pretrained encoder.

    Copied from https://github.com/nasa/pretrained-microscopy-models/blob/9b7c4abc1321e81eca7a68d548e5371676fa74fa/pretrained_microscopy_models/util.py#L27 to avoid version conflicts with smp.

    Args:
        encoder (str): pretrained encoder model name (e.g. resnet50)
        encoder_weights (str): pretraining dataset, either 'micronet' or
            'imagenet-micronet' with the latter indicating the encoder
            was first pretrained on imagenet and then finetuned on microscopynet
        version (float): model version to use, defaults to latest.
            Current options are 1.0 or 1.1.
        self_supervision (str): self-supervision method used. If self-supervision
            was not used set to '' (which is default).

    Returns:
        str: url to download the pretrained model
    """

    # there is an error with the name for resnext101_32x8d so catch and return
    # (currently there is only version 1.0 for this model so don't need to check version.)
    if encoder == "resnext101_32x8d":
        return "https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/resnext101_pretrained_microscopynet_v1.0.pth.tar"

    # only resnet50/micronet has version 1.1 so I'm not going to overcomplicate this right now.
    if encoder != "resnet50" or encoder_weights != "micronet":
        version = 1.0

    # setup self-supervision
    if self_supervision != "":
        version = 1.0
        self_supervision = "_" + self_supervision

    # correct for name change for URL
    if encoder_weights == "micronet":
        encoder_weights = "microscopynet"
    elif encoder_weights == "image-micronet":
        encoder_weights = "imagenet-microscopynet"
    else:
        raise ValueError("encoder_weights must be 'micronet' or 'image-micronet'")

    # get url
    url_base = (
        "https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/"
    )
    url_end = "_v%s.pth.tar" % str(version)
    return (
        url_base + f"{encoder}{self_supervision}_pretrained_{encoder_weights}" + url_end
    )


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
        url = get_pretrained_microscopynet_url(encoder, pretrained_weights)
        try:
            load_state_dict_from_url(url, map_location="cpu")
        except Exception as e:
            print(f"Skipping {pretrained_weights} weights for {encoder}: {e}")
