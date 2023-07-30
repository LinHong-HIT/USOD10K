from .Transformer_depth import Transformer
from .Transformer import token_Transformer
from .DAM_module import *
from .Decoder_Dconv import Decoder


class ImageDepthNet(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet, self).__init__()
        # Cross modality fusion
        self.DAM1 = DAM_module(6)
        self.DAM2 = CA_SA_Enhance(128)
        self.DAM3 = CA_SA_Enhance(128)
        # Encoder
        self.rgb_backbone = T2t_vit_t_14(pretrained=True, args=args)
        self.depth_backbone = T2t_vit_t_14(pretrained=True, args=args)

        self.transformer = Transformer(embed_dim=384, depth=12, num_heads=6, mlp_ratio=3.)

        # Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder()


    def forward(self, image_Input, depth_Input):
        B, _, _, _ = image_Input.shape

        feature_map1 = self.rgb_backbone(image_Input, layer_flag=1)
        dep_layer3_vit, _, _, dep_layer1, dep_layer2, dep_layer3 = self.depth_backbone(depth_Input)

        img_cmf1 = self.DAM1(feature_map1, dep_layer1)
        img_layer_cat1 = feature_map1 + img_cmf1
        feature_map2, rgb_fea_1_4 = self.rgb_backbone(img_layer_cat1, layer_flag=2)

        img_cmf2 = self.DAM2(feature_map2, dep_layer2)
        img_layer_cat2 = feature_map2 + img_cmf2
        feature_map3, rgb_fea_1_8 = self.rgb_backbone(img_layer_cat2, layer_flag=3)

        img_cmf3 = self.DAM3(feature_map3, dep_layer3)
        img_layer_cat3 = feature_map3 + img_cmf3

        img_layer3_vit = self.rgb_backbone(img_layer_cat3, image_Input, layer_flag=4)

        rgb_fea_1_16, depth_fea_1_16 = self.transformer(img_layer3_vit, dep_layer3_vit)

        rgb_fea_1_16 = rgb_fea_1_16.transpose(1, 2).reshape(B, 384, 14, 14)
        depth_fea_1_16 = depth_fea_1_16.transpose(1, 2).reshape(B, 384, 14, 14)

        outputs = self.decoder.forward(rgb_fea_1_16, depth_fea_1_16,  feature_map3, feature_map2, feature_map1)


        return outputs
