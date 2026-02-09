import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import numpy as np
import torchmetrics
import pytorch_lightning as pl
from tensorboardX import SummaryWriter
from typing import Tuple, Union
from torch.nn import functional as F
import copy
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from ConfusionMatrix import display_cls_confusion_matrix
from sklearn.metrics import classification_report
from esresnet import ESResNeXtFBSP
#baseline frames16
lr = 0.0002
_tokenizer = _Tokenizer()
#tensorboard --logdir /home/et23-maixj/mxj/JMPF_pooling/runs --port=9009
 # 初始化PromptPool
prompt_length = 64   # 提示的长度
pool_size = 32
top_k = 5
freq = [0] * pool_size
pltname = 'unimod_39k_v_v1'
# 加入alpha L1正则化
class PromptFrequencyTable(nn.Module):
    def __init__(self, num_prompts):
        super(PromptFrequencyTable, self).__init__()
        # 初始化频率表，每个提示的初始使用频率为1
        self.frequency = torch.ones(num_prompts) / pool_size
    
    def update_frequency(self, selected_indices):
        # 为选中的提示增加使用频率
        self.frequency[selected_indices] += 1
                
    def get_least_used_prompt(self):
        # 获取最少被使用的提示的索引
        least_used_index = torch.argmin(self.frequency).item()
        return least_used_index

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        # 定义卷积层，用于提取音频特征
        # 输入形状：[32, 1, 110250] (batch_size, channels, length)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15, stride=5, padding=1) # 输出形状：[32, 16, 55125]
        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=5, padding=1) # 输出形状：[32, 32, 27562]
        self.conv3 = nn.Conv1d(32, 64, kernel_size=15, stride=5, padding=1) # 输出形状：[32, 64, 13781]
        self.conv4 = nn.Conv1d(64, 128, kernel_size=15, stride=5, padding=1) # 输出形状：[32, 128, 6890]
        self.conv5 = nn.Conv1d(128, 256, kernel_size=15, stride=5, padding=1) # 输出形状：[32, 256, 3445]
        self.conv6 = nn.Conv1d(256, 512, kernel_size=15, stride=5, padding=1) # 输出形状：[32, 512, 1723]
        
        # 定义全连接层，用于降维
        self.fc = nn.Sequential(nn.Linear(512 * 5, 512 * 3), nn.Linear(512 * 3, 768))  # 全连接层, 输入形状：[32, 512 * 5], 输出形状：[32, 768]
        
        # 定义Layer Normalization
        self.ln = nn.LayerNorm(768)  # Layer Normalization, 形状：[32, 768]

    def forward(self, x):
        # 卷积层提取特征
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        # 将特征图展平
        x = x.view(x.size(0), -1)  # 展平后的形状：[32, 512*1723]
        
        # 通过全连接层降维
        x = self.fc(x)  # 全连接层输出形状：[32, 768]
        
        # 应用Layer Normalization
        x = self.ln(x)  # Layer Normalization后的形状：[32, 768]
        
        return x

class PromptPool(nn.Module):
    """
    提示池类，用于从提示池中选择与输入最相似的提示。

    属性:
        pool_size: int - 提示池中的提示数量。
        prompt_length: int - 每个提示的长度。
        embed_dim: int - 嵌入向量的维度。
        top_k: int - 选择的最相似提示的数量。
    """
    def __init__(self, pool_size, prompt_length, top_k,  classnames, clip_model):
        super(PromptPool, self).__init__()

        n_cls = len(classnames)
        dtype = clip_model.dtype
        embed_dim = clip_model.ln_final.weight.shape[0]

        self.pool_size = pool_size
        self.prompt_length = int(prompt_length / 2)
        self.nctx = int(prompt_length / 2)
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.prompt_pool = nn.Parameter(nn.init.normal_(torch.randn(pool_size, self.prompt_length, embed_dim), std=0.02))
        self.prompt_key = nn.Parameter(nn.init.normal_(torch.randn(pool_size, embed_dim, dtype=dtype), std=0.02))
        self.transformerClip = TransformerClip(width=768, layers=12, heads=8, )
        # prompt_length = n_ctx
        # ctx_dim = embed_dim
        self.prompt_table = PromptFrequencyTable(pool_size)
        # self-attention
        # fc_in_dim = 64 # original fc layer's in dimention 512

        self.alpha = nn.Sequential(nn.Linear(self.embed_dim, 1),nn.Sigmoid())

        ctx_vectors = torch.empty(self.nctx, embed_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * prompt_length)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {prompt_length}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + prompt_length :, :])  # CLS, EOS

        self.n_cls = n_cls

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        # 归一化频率并计算倒数作为惩罚因子
        self.penalty_factors = torch.log(self.prompt_table.frequency.max() / self.prompt_table.frequency + 1)  #添加小常数
        self.pencoding = PositionalEncoding()


    def forward(self, x_embed, trainflag):
        # x_embed为输入向量

        # 计算输入嵌入向量的L2标准化值
        x_embed = torch.mean(x_embed,dim=0)
        x_embed_norm = x_embed / torch.norm(x_embed, p=2, dim=-1, keepdim=True)

        # 计算提示键的L2标准化值
        prompt_key_norm = self.prompt_key / torch.norm(self.prompt_key, p=2, dim=-1, keepdim=True)

        # 计算输入与每个提示键的相似度
        sim = torch.matmul(x_embed_norm, prompt_key_norm.T)


        if trainflag:
            # 调整相似度分数，以偏好不常用的提示
            sim = sim * self.penalty_factors.to(sim.device)
        # print(self.penalty_factors)

            
        top_k_indices = torch.topk(sim, self.top_k, dim=-1).indices
        # 选择top_k个最相似的提示
        if self.top_k != 1:
            selected_prompts = torch.index_select(self.prompt_pool, 0, top_k_indices.view(-1))
            attention_weights = self.alpha(selected_prompts.view(-1, self.embed_dim)).view(-1, self.prompt_length)

            attention_weights = attention_weights.unsqueeze(-1)

            out = attention_weights * selected_prompts
            selected_prompts = out.sum(dim=0).unsqueeze(0).expand(self.n_cls, -1, -1)

            for i in top_k_indices:
                freq[i.item()] = freq[i.item()] + 1
                self.prompt_table.update_frequency(i.item())
        else:
            selected_prompts = torch.index_select(self.prompt_pool, 0, top_k_indices.view(-1))
            freq[top_k_indices.item()] = freq[top_k_indices.item()] + 1
            self.prompt_table.update_frequency(top_k_indices.item()) #去掉频率
        
        prefix = self.token_prefix
        suffix = self.token_suffix

        # selected_prompts = selected_prompts.expand(self.n_cls, -1, -1)          # (batch, n_ctx, ctx_dim)

        ctx = self.ctx                     # (n_ctx, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)

        
        ctx = ctx.expand(self.n_cls, -1, -1)          # (batch, n_ctx, ctx_dim)

        # selected_prompts = selected_prompts + ctx  #双流融合

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                selected_prompts,
                ctx,
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts

    def on_save_checkpoint(self, checkpoint):
        # 将my_variable添加到检查点字典
        checkpoint['my_variable'] = self.penalty_factors
        checkpoint['prompt_key'] = self.prompt_key
        checkpoint['prompt_pool'] = self.prompt_pool
        return checkpoint
    
    def on_load_checkpoint(self, checkpoint):
        # 从检查点字典中获取my_variable
        self.penalty_factors = checkpoint.get('my_variable')  # 使用默认值作为备选
        self.prompt_key = checkpoint.get('prompt_key')  # 使用默认值作为备选
        self.prompt_pool = checkpoint.get('prompt_pool')  # 使用默认值作为备选

    def _mean_pooling_for_similarity_visual(self, visual_output, valid_list,):
        video_mask_un = valid_list.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4, residual_ratio=0.2):
        super(Adapter, self).__init__()
        self.residual_ratio = residual_ratio
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        a = self.fc(x)
        x = self.residual_ratio * a + (1 - self.residual_ratio) * x
        return x

def make_classifier_head(classifier_head,
                         clip_encoder,
                         bias=False):

    if clip_encoder == 'ViT-B/32':
        in_features = 512
    elif clip_encoder == 'RN50':
        in_features = 1024
    elif clip_encoder == 'ViT-L/14':
        in_features = 768

    num_classes = 7

    linear_head = nn.Linear(in_features, num_classes, bias=bias)

    if classifier_head == 'linear':
        head = linear_head
    elif classifier_head == 'adapter':
        adapter = Adapter(in_features, residual_ratio=0.2)
        head = nn.Sequential(
            adapter,
            linear_head
        )
    else:
        raise ValueError(f"Invalid head: {classifier_head}")
    return head, num_classes, in_features

clip_name = 'ViT-L/14'
if clip_name == 'ViT-B/32':
    fea_size = 512
elif clip_name == 'ViT-L/14':
    fea_size = 768

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class CapEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class ResidualAttentionBlockClip(nn.Module): 
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head) #PyTorch提供的nn.MultiheadAttention类，这个类可以自动根据输入的序列计算QKV三个子空间的向量，
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, para_tuple: tuple):
        # x: torch.Tensor, attn_mask: torch.Tensor
        # print(para_tuple)
        x, attn_mask = para_tuple
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)

class sResidualAttentionBlockClip(nn.Module): 
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head) #PyTorch提供的nn.MultiheadAttention类，这个类可以自动根据输入的序列计算QKV三个子空间的向量，
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head
        self.shareq = nn.Parameter(torch.ones(4,d_model))

    def attention(self, q: torch.Tensor, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        return self.attn(q, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, para_tuple: tuple):
        # x: torch.Tensor, attn_mask: torch.Tensor
        # print(para_tuple)
        x, attn_mask = para_tuple
        x = x + self.attention(torch.unsqueeze(self.shareq,dim=1).repeat(1,x.shape[1],1), self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)

class LogitHead(nn.Module):
    def __init__(self, head, logit_scale=float(np.log(1 / 0.07))):
        super().__init__()
        self.head = head
        self.logit_scale = logit_scale
        
        # Not learnable for simplicity
        self.logit_scale = torch.FloatTensor([logit_scale])
        # Learnable
        # self.logit_scale = torch.nn.Parameter(torch.ones([]) * logit_scale)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        x = x.to(torch.float32)
        x = self.head(x)
        x = x * self.logit_scale.exp().to(x)
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

class TransformerClip(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlockClip(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        return self.resblocks((x, attn_mask))[0]

class sTransformerClip(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[sResidualAttentionBlockClip(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.resblocks((x, attn_mask))[0]

class encoder_image(nn.Module):
    def __init__(self):
        super().__init__()
        vision_width = 768
        vision_heads = vision_width // 64
        vision_patch_size = 32

        image_resolution = 224
        embed_dim = 512

        self.visual = VisionTransformer(
            input_resolution = image_resolution,
            patch_size = vision_patch_size,
            width = vision_width,
            layers = 12,
            heads = vision_heads,
            output_dim = embed_dim
        )
    
    def encode_image(self, image):
        return self.visual(image.type(torch.float16))
    
    def forward(self, x: torch.Tensor):
        x = self.encode_image(x)
        return x

class PositionalEncoding(nn.Module):
    # 初始化函数，需要指定嵌入维度和最大序列长度
    def __init__(self):
        super().__init__()
        self.frame_position_embeddings = nn.Embedding(16,768)
    def forward(self, x):
        return self.frame_position_embeddings(x)

class TextTensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor, eot_indices):
        super(TextTensorDataset, self).__init__()
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        self.eot_indices = eot_indices
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index], self.eot_indices[index]

    def __len__(self):
        return self.input_tensor.size(0)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
class Expo(nn.Module):
    def __init__(self):
        super(Expo, self).__init__()
        self.fc1 = nn.Linear(768, 768 * 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(768 * 2, 768 * 2)
        self.fc3 = nn.Linear(768 * 2, 768)  # 输出维度与输入维度相同
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class QKVResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_11 = LayerNorm(d_model)
        self.ln_12 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def attention(self, q: torch.Tensor, kv: torch.Tensor,):
        return self.attn(q, kv, kv, need_weights=False)[0]

    def forward(self, para_tuple: tuple):
        # x: torch.Tensor, attn_mask: torch.Tensor
        # print(para_tuple)
        q, kv = para_tuple
        x = kv + self.attention(self.ln_11(q), self.ln_12(kv))
        x = x + self.mlp(self.ln_2(x))
        return (q, x)

class QKVTransformerClip(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[QKVResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        return self.resblocks((q, kv))[0]

class JMPF(pl.LightningModule):
    def __init__(self):
        super().__init__()
        classnames = {'anger','disgust','fear','happiness','neutral','sadness','surprise','contempt','anxiety','helplessness','disappointment'}
        self.confusion_matrix = np.zeros([7, 7])
        transformer_width = fea_size
        transformer_heads = transformer_width // 64
        transformer_layers = 12

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=len(classnames))
        
        self.transformerClip = TransformerClip(width=768, layers=transformer_layers,
                                        heads=8, )
        model, preprocess = clip.load(clip_name,device = self.device)
        model = model.eval()

        smodel, preprocess = clip.load('ViT-B/32',device = self.device)
        smodel = smodel.eval()
        self.pencoding = PositionalEncoding()
        self.cpencoding = PositionalEncoding()
        self.image_encoder = model.visual
        self.image_encoder_s = model.visual
        self.token_embedding = smodel.token_embedding
        n_fft = 2048
        hop_length = 561
        win_length = 1654
        window = 'blackmanharris'
        normalized = True,
        onesided = True,
        spec_height = -1
        spec_width = -1
        apply_attention: bool = True
        multilabel: bool = True
        pretrained: Union[bool, str] = True
        embed_dim = 512
        logit = 4.60517

        self.cap_encoder = CapEncoder(smodel)
        # self.audio_encoder = ESResNeXtFBSP(
        #     n_fft=n_fft,#FFT的窗口大小
        #     hop_length=hop_length,#FFT的跳跃长度
        #     win_length=win_length,#FFT每个窗口的实际长度
        #     window=window,#FFT使用的窗函数
        #     normalized=normalized,#对频谱图进行归一化处理
        #     onesided=onesided,#是否只保留频谱图的一半
        #     spec_height=spec_height,#频谱图的高度
        #     spec_width=spec_width,#频谱图的宽度，也就是时间维度的大小
        #     num_classes=embed_dim,#音频编码器输出的特征向量的维度，也就是类别数
        #     apply_attention=apply_attention,#是否在音频编码器中使用注意力机制
        #     pretrained=False,
        # )
        self.audio_encoder = AudioEncoder()

        self.lstm =  nn.LSTM(input_size=transformer_width, hidden_size=transformer_width, num_layers=1, bidirectional=False)

        # self.text_encoder = TextEncoder(smodel)
        # self.prompt_learner = PromptLearner(classnames, smodel)

        self.prompt_pool = PromptPool(pool_size=pool_size, prompt_length=prompt_length, top_k=top_k, classnames=classnames, clip_model=model)
        
        self.text_encoder = TextEncoder(model)
        self.tokenized_prompts = self.prompt_pool.tokenized_prompts


        self.seq_transformerClip = TransformerClip(width=768, layers=12,
                                        heads=8)
        
        # self.feate =  nn.Linear(768, 768)
        # self.atransTo768 = nn.Linear(512, 768)
        self.ctransTo768 = nn.Linear(512, 768)

        # self.atransTo768 = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(512, 1024)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(1024, 768))
        # ]))
        # self.ctransTo768 = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(512, 1024)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(1024, 768))
        # ]))

        self.writer = SummaryWriter(log_dir='/home/et23-maixj/mxj/JMPF_pooling/runs')

        labels = [0,1,2,3,4,5,6]
        labels = torch.tensor(labels)

        head, num_classes, in_features = make_classifier_head(
                classifier_head = "adapter",
                clip_encoder = 'ViT-L/14',
            )
        
        # self.logit_head = LogitHead(
        #                 head,
        #                 logit_scale = 4.60517,
        #             )

        del_key = []
        preweig = model.state_dict()
        
        #删掉多余层

        for key in preweig.keys():
            if 'visual' not in key:
                del_key.append(key)

        for key in del_key:
            del preweig[key]
        
        # #全体改名字
        # finaweig = copy.deepcopy(preweig)
        self.logit_scale = model.logit_scale
        # self.transshape = nn.Linear(768, 512)
        # for key in preweig.keys():
        #     if 'resblocks' in key:
        #         key_change = key.replace('resblocks.','resblocks')
        #         finaweig[key_change] = copy.deepcopy(finaweig[key])
        #         del finaweig[key]
        
        #加载
        # self.image_encoder.load_state_dict(finaweig, strict=False)
        self.image_encoder.load_state_dict(preweig, strict=False)
        self.image_encoder_s.load_state_dict(preweig, strict=False)
        self.logit_scale = model.logit_scale
        
        # name_not_to_update = ["model","smodel"]
        # for name, param in self.named_parameters():
        #     param.requires_grad_(True)
        #     for targ in name_not_to_update:
        #         if targ in name:
        #             param.requires_grad_(False)
        for name, param in self.named_parameters():
            param.requires_grad_(False)
            if 'logit_head' in name:
                param.requires_grad_(True)
            elif 'pencoding' in name:
                param.requires_grad_(True)
            elif 'transformerClip' in name:
                param.requires_grad_(True)
            elif 'seq_transformerClip' in name:
                param.requires_grad_(True)
            elif 'chag' in name:
                param.requires_grad_(True)
            elif 'cpencoding' in name:
                param.requires_grad_(True)
            elif 'prompt_pool' in name:
                param.requires_grad_(True)
            elif 'feate' in name:
                param.requires_grad_(True)
            elif 'transshape' in name:
                param.requires_grad_(True)
            elif 'lstm' in name:
                param.requires_grad_(True)
            elif 'atransTo768' in name:
                param.requires_grad_(True)
            elif 'ctransTo768' in name:
                param.requires_grad_(True)
            elif 'audio_encoder' in name:
                param.requires_grad_(True)


        # 遍历模型中的所有参数
        # print("not train param:")
        # for name, param in self.named_parameters():
        #     if not param.requires_grad:
        #         print(name)

    @property
    def learning_rate(self):
        # 获取优化器
        optimizer = self.optimizers()
        # 返回第一个参数组的学习率
        return optimizer.param_groups[0]["lr"]

    def parse_batch_train(self, batch):
        f_frames, o_frames, label, valid_list = batch
        return f_frames, o_frames, label, valid_list
    
    def _mean_pooling_for_similarity_visual(self, visual_output, valid_list,):
        video_mask_un = valid_list.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    
    def forward(self,batch):
        audio = None
        cap = None
        f_frames = None
        o_frames = None
        label = None

        audio_encoder = self.audio_encoder
        cap_encoder = self.cap_encoder

        f_frames, o_frames, label, valid_list = self.parse_batch_train(batch)

        cap = None
        audio = None

        imageface_feature_flag = 1
        imagescene_feature_flag = 0
        cap_feature_flag = 0
        audio_feature_flag = 0

        encoder_image = self.image_encoder
        b, timestep, channel, h, w = f_frames.shape
        f_frames = f_frames.view(b*timestep,channel,h,w)
        video_features = encoder_image(f_frames.type(self.dtype))
        bs_pair = valid_list.size(0)
        #12*16 512
        video_features = video_features.view(bs_pair, timestep, video_features.size(-1))
        
        if o_frames != None:
            imagescene_feature_flag = 1
            b, timestep, channel, h, w = o_frames.shape
            o_frames = o_frames.view(b*timestep,channel,h,w)
            o_video_features = self.image_encoder_s(o_frames.type(self.dtype))
            bs_pair = valid_list.size(0)
            
            #12*16 512
            o_video_features = o_video_features.view(bs_pair, timestep, o_video_features.size(-1))
        else:
            o_video_features = torch.zeros_like(video_features)

        ##
        if cap is not None and label is not None:
            cap_feature_flag = 1
            cap_t = []
            for cap_i in cap:
                cap_i = cap_i.replace("_", " ")
                prompt_i = clip.tokenize(cap_i)
                cap_t.append(prompt_i)

            cap_tokenized_prompts = torch.cat([p for p in cap_t]).to(self.logit_scale.device) # 8,77

            with torch.no_grad():
                cap_embedding = self.token_embedding(cap_tokenized_prompts).type(self.dtype)
            
            # prompt = clip.tokenize(cap)
            # with torch.no_grad():
            #       embedding = self.model.token_embedding(prompt.to('cpu')).type(self.model.dtype)
            cap_features = cap_encoder(cap_embedding,cap_tokenized_prompts)
            # cap_features = cap_features / cap_features.norm(dim=-1, keepdim=True)
            cap_features = self.ctransTo768(cap_features)
        else:
            cap_features = torch.zeros(b,768)

        if audio is not None and label is not None:
            audio_feature = audio_encoder(audio)
            audio_feature_flag = 1
            # audio_feature = self.atransTo768(audio_feature)
        else:
            audio_feature = torch.zeros(b,768)
        
        

        if cap_features != None and video_features !=None:
            cap_features = cap_features.to(video_features.device)

        if audio_feature != None and video_features !=None:
            audio_feature = audio_feature.to(video_features.device)

        ##
        ###########
            

        f_features = torch.split(video_features, 1, dim=1)
        o_features = torch.split(o_video_features, 1, dim=1)
        
        count = len(f_features)
        f_features_n = []
        o_features_n = []

        for i in range(count): 
            f_temp = f_features[i].squeeze(1)
            o_temp = o_features[i].squeeze(1)

            in_i = torch.stack([f_temp, o_temp], dim=0).to(f_temp.device) ## type, b, fea 向右传播
            ty, b, f = in_i.shape
            output, (h, c) = self.lstm(in_i)
            ou_i = output + in_i
            oup = torch.split(ou_i, 1, dim=0)

            f_features_n.append(oup[0].squeeze(0))
            o_features_n.append(oup[1].squeeze(0))
        
        f_features = torch.stack(f_features_n,dim = 1)
        o_features = torch.stack(o_features_n,dim = 1)

        ###########

        visual_output = f_features
        o_features = torch.mean(o_features,dim = 1)
        

        visual_output_original = visual_output
        seq_length = visual_output.size(1)#获取序列长度L
        position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
        #把position_ids扩展成和visual_output相同的批次大小N
        position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
        
        frame_position_embeddings = self.pencoding(position_ids)

        #获取位置编码向量，存储在frame_position_embeddings中
        visual_output = visual_output + frame_position_embeddings
        #把位置编码向量和视频帧特征向量相加
        extended_video_mask = (1.0 - valid_list.unsqueeze(1)) * -1000000.0
        #指示哪些视频帧是有效的，哪些是无效的
        extended_video_mask = extended_video_mask.expand(-1, valid_list.size(1), -1)
        visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND 调整visual_output的维度顺序
        visual_output = self.transformerClip(visual_output, extended_video_mask) #16,32,768 32,16,16
        #对输入数据进行自注意力（self-attention）计算，并输出新的特征向量
        visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
        visual_output = visual_output + visual_output_original

        visual_output = self._mean_pooling_for_similarity_visual(visual_output, valid_list)

        visual_output = visual_output.unsqueeze(0)
        o_features = o_features.unsqueeze(0)
        cap_features = cap_features.unsqueeze(0)
        audio_feature = audio_feature.unsqueeze(0)

        all_feature = torch.stack([visual_output.squeeze(0), o_features.squeeze(0), cap_features.squeeze(0), audio_feature.squeeze(0)], dim=0)


        valid = [imageface_feature_flag,imagescene_feature_flag,cap_feature_flag,audio_feature_flag]
        clip_position_embeddings = self.cpencoding(torch.arange(4, dtype=torch.long, device=all_feature.device).unsqueeze(0).expand(all_feature.shape[1], -1)).permute(1, 0, 2)

        valid = torch.tensor(valid)
        valid = valid.repeat(b,1).to(all_feature.device)
        mask = (1.0 - valid.unsqueeze(1)) * -1000000.0
        mask = mask.expand(-1, valid.size(1), -1).to(all_feature.device)

        all_feature = self.seq_transformerClip(all_feature + clip_position_embeddings,mask)
        #4,32,768  32,4,4 #16,32,768 32,16,16
        all_feature = all_feature.permute(1, 0, 2) 
        all_feature = self._mean_pooling_for_similarity_visual(all_feature, valid)

        all_feature = all_feature / all_feature.norm(dim=-1, keepdim=True)

        # logits = self.logit_head(all_feature)
        # loss = None

        # # 创建一些模拟的输入嵌入向量
        # input_embed = torch.randn(3, embed_dim)  # 假设有3个输入向量


        prompts = self.prompt_pool(all_feature, self.training)
        
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        # logits = self.logit_head(all_feature)
        # logits = logit_scale * self.transshape(all_feature) @ text_features.t()
        logits = logit_scale * all_feature @ text_features.t()
        loss = None

        # L1 正则化系数
        lambda_l = 0.01

        # 计算 alpha 层的 γ 缩放因子的 L1 正则化项
        l1_reg = lambda_l * self.prompt_pool.alpha[0].weight.abs().sum()
        
        if label != None:
            loss = F.cross_entropy(logits, label) + l1_reg

        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, label)
        return loss, preds, acc

    def training_step(self, batch, batch_nb):
        # loss = self.forward(batch)
        loss, preds, acc = self.forward(batch)
        self.log('train_loss', loss, prog_bar=True,sync_dist=True)
        self.log('train_acc', acc, prog_bar=True, sync_dist=True)
        self.log("learning_rate", self.learning_rate, prog_bar=True, sync_dist=True)
        self.writer.add_scalar("learning_rate", self.learning_rate, self.global_step)
        self.writer.add_scalar("train_loss", loss, self.global_step)
        self.writer.add_scalar("train_acc", acc, self.global_step)
        # print(freq)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # loss = self.forward(batch)
        loss, preds, acc = self.forward(batch)
        self.log('val_loss', loss, prog_bar=True,sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        self.writer.add_scalar("val_loss", loss, self.global_step)
        self.writer.add_scalar("val_acc", acc, self.global_step)
        self.writer.add_histogram("freq", np.array(freq), self.global_step)
        # print(freq)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
        self.parameters(), lr=lr, weight_decay=0.000001, betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, threshold=0.001)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def on_test_epoch_end(self) -> None:
        print(self.confusion_matrix)
        labels_7 = ['anger','disgust','fear','happiness','neutral','sadness','surprise']

        test_number_7 = [1487, 467, 431, 1473, 1393, 638, 1958]
        name_7 = '39K (7 classes)'
        display_cls_confusion_matrix(self.confusion_matrix,labels_7,test_number_7 , name_7, pltname)
        print(freq)
        return super().on_test_epoch_end()

    def test_step(self, batch, batch_idx):
        f_frames, o_frames, label, valid_list = self.parse_batch_train(batch)
        loss, preds, acc = self.forward(batch)
        # loss = self.forward(batch)
        y_pred = preds.cpu().numpy()
        y_true = label.cpu().numpy()

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=True)
        # 初始化WAR和UAR
        war_0, war_1, war_2, war_3, war_4, war_5, war_6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if '0' in report:
            war_0 = report['0']['recall']
        if '1' in report:
            war_1 = report['1']['recall']
        if '2' in report:
            war_2 = report['2']['recall']
        if '3' in report:
            war_3 = report['3']['recall']
        if '4' in report:
            war_4 = report['4']['recall']
        if '5' in report:
            war_5 = report['5']['recall']
        if '6' in report:
            war_6 = report['6']['recall']
        
        uar = report['macro avg']['recall']
        war = report['accuracy']

        self.log('test_war_0', war_0, prog_bar=True,sync_dist=True)
        self.log('test_war_1', war_1, prog_bar=True,sync_dist=True)
        self.log('test_war_2', war_2, prog_bar=True,sync_dist=True)
        self.log('test_war_3', war_3, prog_bar=True,sync_dist=True)
        self.log('test_war_4', war_4, prog_bar=True,sync_dist=True)
        self.log('test_war_5', war_5, prog_bar=True,sync_dist=True)
        self.log('test_war_6', war_6, prog_bar=True,sync_dist=True)

        batch_size = len(y_pred)
        for i in range(batch_size):
            self.confusion_matrix[y_true[i],y_pred[i]] += 1
        
        self.log('test_loss', loss, prog_bar=True,sync_dist=True)
        self.log('test_war', war, prog_bar=True, sync_dist=True)
        self.log('test_uar', uar, prog_bar=True, sync_dist=True)





