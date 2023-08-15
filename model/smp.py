import segmentation_models_pytorch as smp
import torch


class SMP:
    def __init__(self):
        self.in_channels = 3
        self.classes = 1

    def load_models(self, model: str):
        model = model.lower()
        if model == "unetplusplus":
            load_model = smp.UnetPlusPlus(
                encoder_name="efficientnet-b5",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=self.in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.classes,  # model output channels (number of classes in your dataset)
            )
        if model == "unet":
            load_model = smp.Unet(
                encoder_name="efficientnet-b5",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=self.in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.classes,  # model output channels (number of classes in your dataset)
            )
        if model == "pspnet":
            load_model = smp.PSPNet(
                encoder_name="efficientnet-b5",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=self.in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.classes,  # model output channels (number of classes in your dataset)
            )
        load_model = (
            load_model.to("cuda") if torch.cuda.is_available() else load_model.to("cpu")
        )
        return load_model
