import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet50
from torchvision.models import resnet101

class MultiHeadAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super(MultiHeadAttentionModule, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        
        assert self.head_dim * num_heads == out_channels, "out_channels must be divisible by num_heads"
        
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.shape
        
        # Split the embedding dimension to multiple heads
        query = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        key = self.key_conv(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        value = self.value_conv(x).view(batch_size, self.num_heads, self.head_dim, height * width)
        
        # Transpose to get dimensions batch_size * num_heads * height*width * head_dim
        query, key = query.transpose(2, 3), key.transpose(2, 3)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = self.softmax(scores)
        
        
        # Apply attention weights to values
        
        weighted_value = torch.matmul(attention, value.transpose(2, 3))
        weighted_value = weighted_value.transpose(2, 3).contiguous()
        weighted_value = weighted_value.view(batch_size, -1, height, width)
        
        return weighted_value

class GlobalMaxPool2d(nn.Module):
    def forward(self, x):
        return F.adaptive_max_pool2d(x, (1, 1))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dropout_prob=0.0):
        super(ResidualBlock, self).__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU()
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.match_channels = in_channels != out_channels
        if self.match_channels:
            self.adjust_channels = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        identity = x
        if self.match_channels:
            identity = self.adjust_channels(identity)
        
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out += identity
        return out


# class ProgressiveFeatureLearning(nn.Module):
#     def __init__(self):
#         super(ProgressiveFeatureLearning, self).__init__()
#         # 增加层的宽度
#         self.layer1 = nn.Sequential(
#             DepthwiseSeparableConv(2048, 1280, kernel_size=3, padding=1), # 从1024增加到1280
#             nn.BatchNorm2d(1280),
#             nn.ELU(),
#             nn.Dropout(p=0.6)
#         )
#         self.layer2 = nn.Sequential(
#             DepthwiseSeparableConv(1280, 768, kernel_size=3, padding=1), # 从512增加到768
#             nn.BatchNorm2d(768),
#             nn.ELU(),
#             nn.Dropout(p=0.5)
#         )
#         # 增加一个额外的深度层
#         self.layer3 = nn.Sequential(
#             DepthwiseSeparableConv(768, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ELU(),
#             nn.Dropout(p=0.4)
#         )
#         self.layer4 = nn.Sequential( # 新增加的层
#             DepthwiseSeparableConv(512, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ELU(),
#             nn.Dropout(p=0.3)
#         )
        
#         self.layer5 = nn.Sequential(
#         DepthwiseSeparableConv(256, 128, kernel_size=3, padding=1),
#         nn.BatchNorm2d(128),
#         nn.ELU(),
#         nn.Dropout(p=0.3)
#         )

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x) # 确保新加层参与前向传播
#         x = self.layer5(x)
#         return x

class ProgressiveFeatureLearning(nn.Module):
    def __init__(self):
        super(ProgressiveFeatureLearning, self).__init__()
        # Increase layer width and add residual connections with dropout
        self.layer1 = ResidualBlock(2048, 1280, kernel_size=3, padding=1, dropout_prob=0.6)
        self.layer2 = ResidualBlock(1280, 768, kernel_size=3, padding=1, dropout_prob=0.5)
        self.layer3 = ResidualBlock(768, 512, kernel_size=3, padding=1, dropout_prob=0.4)
        self.layer4 = ResidualBlock(512, 256, kernel_size=3, padding=1, dropout_prob=0.3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class DeepProgressiveAttentionNetwork(nn.Module):
    def __init__(self, num_classes=101):
        super(DeepProgressiveAttentionNetwork, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Progressive Feature Learning branch remains unchanged.
        self.progressive_feature_learning = ProgressiveFeatureLearning()
        self.regional_feature_enhancement = MultiHeadAttentionModule(256, 256, num_heads=8)
        
        # Global Feature Learning branch - additional classifier
        self.global_max_pool = GlobalMaxPool2d()
        self.classifier_a = nn.Linear(2048, num_classes)  # Assuming the backbone's output has 2048 channels
        
        # Regional Feature Enhancement branch - additional classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier_b = nn.Linear(256, num_classes)
        
        # Final classifier after feature concatenation
        self.classifier_final = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x):
        backbone_features = self.features(x)
        
        # Global Feature Learning branch
        global_features = self.global_max_pool(backbone_features)
        global_features_flat = global_features.view(global_features.size(0), -1)
        global_class_scores = self.classifier_a(global_features_flat)
        
        # Progressive Feature Learning and Regional Feature Enhancement branches
        progressive_features = self.progressive_feature_learning(backbone_features)
        regional_features = self.regional_feature_enhancement(progressive_features)
        regional_features_flat = self.global_avg_pool(regional_features)
        regional_features_flat = regional_features_flat.view(regional_features.size(0), -1)
        regional_class_scores = self.classifier_b(regional_features_flat)
        
        # Feature concatenation and final classification
        combined_class_scores = torch.cat((global_class_scores, regional_class_scores), dim=1)
        out = self.classifier_final(combined_class_scores)
        return out, global_class_scores, regional_class_scores  # Return individual scores for KL divergence loss calculation


