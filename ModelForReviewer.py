import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchmetrics
from tensorboardX import SummaryWriter
from typing import Tuple, Union
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from ConfusionMatrix import display_cls_confusion_matrix, display_cls_confusion_matrix_6
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Initialize settings
lr = 0.0002
tokenizer = _Tokenizer()

# Initialize Prompt Pool
prompt_length = 64
pool_size = 32
top_k = 5
freq = [0] * pool_size
pltname = 'unimod_mafw_missmode'

# Add alpha L1 regularization
class PromptFrequencyTable(nn.Module):
    def __init__(self, num_prompts):
        super(PromptFrequencyTable, self).__init__()
        self.frequency = torch.ones(num_prompts) / pool_size
    
    def update_frequency(self, selected_indices):
        self.frequency[selected_indices] += 1
                
    def get_least_used_prompt(self):
        least_used_index = torch.argmin(self.frequency).item()
        return least_used_index

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15, stride=5, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=5, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=15, stride=5, padding=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=15, stride=5, padding=1)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=15, stride=5, padding=1)
        self.conv6 = nn.Conv1d(256, 512, kernel_size=15, stride=5, padding=1)
        
        self.fc = nn.Sequential(nn.Linear(512 * 5, 512 * 3), nn.Linear(512 * 3, 768))
        self.ln = nn.LayerNorm(768)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.ln(x)
        
        return x

class PromptPool(nn.Module):
    def __init__(self, pool_size, prompt_length, top_k, classnames, clip_model):
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
        self.transformerClip = TransformerClip(width=768, layers=12, heads=8)
        self.prompt_table = PromptFrequencyTable(pool_size)
        self.alpha = nn.Sequential(nn.Linear(self.embed_dim, 1), nn.Sigmoid())

        ctx_vectors = torch.empty(self.nctx, embed_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * prompt_length)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {prompt_length}")

        self.ctx = nn.Parameter(ctx_vectors)

        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + prompt_length:, :])

        self.n_cls = n_cls
        self.tokenized_prompts = tokenized_prompts

        self.penalty_factors = torch.log(self.prompt_table.frequency.max() / self.prompt_table.frequency + 1)
        self.pencoding = PositionalEncoding()

    def forward(self, x_embed, train_flag):
        x_embed = torch.mean(x_embed, dim=0)
        x_embed_norm = x_embed / torch.norm(x_embed, p=2, dim=-1, keepdim=True)
        prompt_key_norm = self.prompt_key / torch.norm(self.prompt_key, p=2, dim=-1, keepdim=True)
        sim = torch.matmul(x_embed_norm, prompt_key_norm.T)

        if train_flag:
            sim = sim * self.penalty_factors.to(sim.device)

        top_k_indices = torch.topk(sim, self.top_k, dim=-1).indices
        if self.top_k != 1:
            selected_prompts = torch.index_select(self.prompt_pool, 0, top_k_indices.view(-1))
            attention_weights = self.alpha(selected_prompts.view(-1, self.embed_dim)).view(-1, self.prompt_length)
            attention_weights = attention_weights.unsqueeze(-1)
            out = attention_weights * selected_prompts
            selected_prompts = out.sum(dim=0).unsqueeze(0).expand(self.n_cls, -1, -1)

            for i in top_k_indices:
                freq[i.item()] += 1
                self.prompt_table.update_frequency(i.item())
        else:
            selected_prompts = torch.index_select(self.prompt_pool, 0, top_k_indices.view(-1))
            freq[top_k_indices.item()] += 1
            self.prompt_table.update_frequency(top_k_indices.item())
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = torch.cat([prefix, selected_prompts, ctx, suffix], dim=1)

        return prompts, self.prompt_pool, self.prompt_key

    def on_save_checkpoint(self, checkpoint):
        checkpoint['penalty_factors'] = self.penalty_factors
        checkpoint['prompt_key'] = self.prompt_key
        checkpoint['prompt_pool'] = self.prompt_pool
        return checkpoint
    
    def on_load_checkpoint(self, checkpoint):
        self.penalty_factors = checkpoint.get('penalty_factors')
        self.prompt_key = checkpoint.get('prompt_key')
        self.prompt_pool = checkpoint.get('prompt_pool')

    def mean_pooling_for_similarity_visual(self, visual_output, valid_list):
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

def make_classifier_head(classifier_head, clip_encoder, bias=False):
    if clip_encoder == 'ViT-B/32':
        in_features = 512
    elif clip_encoder == 'RN50':
        in_features = 1024
    elif clip_encoder == 'ViT-L/14':
        in_features = 768

    num_classes = 11
    linear_head = nn.Linear(in_features, num_classes, bias=bias)

    if classifier_head == 'linear':
        head = linear_head
    elif classifier_head == 'adapter':
        adapter = Adapter(in_features, residual_ratio=0.2)
        head = nn.Sequential(adapter, linear_head)
    else:
        raise ValueError(f"Invalid head: {classifier_head}")

    return head, num_classes, in_features

clip_name = 'ViT-L/14'
if clip_name == 'ViT-B/32':
    fea_size = 512
elif clip_name == 'ViT-L/14':
    fea_size = 768

class LayerNorm(nn.LayerNorm):
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
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
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
        self.attn = nn.MultiheadAttention(d_model, n_head)
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
        x, attn_mask = para_tuple
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)

class SharedResidualAttentionBlockClip(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head
        self.shared_q = nn.Parameter(torch.ones(4, d_model))

    def attention(self, q: torch.Tensor, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        return self.attn(q, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, para_tuple: tuple):
        x, attn_mask = para_tuple
        x = x + self.attention(torch.unsqueeze(self.shared_q, dim=1).repeat(1, x.shape[1], 1), self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)

class LogitHead(nn.Module):
    def __init__(self, head, logit_scale=float(np.log(1 / 0.07))):
        super().__init__()
        self.head = head
        self.logit_scale = torch.FloatTensor([logit_scale])

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
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
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

class SharedTransformerClip(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[SharedResidualAttentionBlockClip(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        return self.resblocks((x, attn_mask))[0]

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vision_width = 768
        vision_heads = vision_width // 64
        vision_patch_size = 32
        image_resolution = 224
        embed_dim = 512

        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=12,
            heads=vision_heads,
            output_dim=embed_dim
        )
    
    def encode_image(self, image):
        return self.visual(image.type(torch.float16))
    
    def forward(self, x: torch.Tensor):
        x = self.encode_image(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_position_embeddings = nn.Embedding(16, 768)
    
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
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class Expo(nn.Module):
    def __init__(self):
        super(Expo, self).__init__()
        self.fc1 = nn.Linear(768, 768 * 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(768 * 2, 768 * 2)
        self.fc3 = nn.Linear(768 * 2, 768)

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

    def attention(self, q: torch.Tensor, kv: torch.Tensor):
        return self.attn(q, kv, kv, need_weights=False)[0]

    def forward(self, para_tuple: tuple):
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
        classnames = {'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'contempt', 'anxiety', 'helplessness', 'disappointment'}
        self.confusion_matrix = np.zeros([11, 11])
        transformer_width = fea_size
        transformer_heads = transformer_width // 64
        transformer_layers = 12

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=len(classnames))
        self.transformer_clip = TransformerClip(width=768, layers=transformer_layers, heads=8)
        model, preprocess = clip.load(clip_name, device=self.device)
        model = model.eval()

        self.pencoding = PositionalEncoding()
        self.cpencoding = PositionalEncoding()
        self.image_encoder = model.visual
        self.image_encoder_s = model.visual
        self.token_embedding = model.token_embedding
        embed_dim = 512

        self.cap_encoder = CapEncoder(model)
        self.audio_encoder = AudioEncoder()
        self.lstm = nn.LSTM(input_size=transformer_width, hidden_size=transformer_width, num_layers=1, bidirectional=False)
        self.prompt_pool = PromptPool(pool_size=pool_size, prompt_length=prompt_length, top_k=top_k, classnames=classnames, clip_model=model)
        self.text_encoder = TextEncoder(model)
        self.tokenized_prompts = self.prompt_pool.tokenized_prompts
        self.seq_transformer_clip = TransformerClip(width=768, layers=12, heads=8)
        self.ctrans_to_768 = nn.Linear(512, 768)
        self.writer = SummaryWriter(log_dir='runs')
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6])

        head, num_classes, in_features = make_classifier_head(classifier_head="adapter", clip_encoder='ViT-L/14')

        pre_weights = model.state_dict()
        del_keys = [key for key in pre_weights.keys() if 'visual' not in key]

        for key in del_keys:
            del pre_weights[key]

        self.image_encoder.load_state_dict(pre_weights, strict=False)
        self.image_encoder_s.load_state_dict(pre_weights, strict=False)
        self.logit_scale = model.logit_scale

        for name, param in self.named_parameters():
            param.requires_grad_(False)
            if 'logit_head' in name:
                param.requires_grad_(True)
            elif 'pencoding' in name:
                param.requires_grad_(True)
            elif 'transformer_clip' in name:
                param.requires_grad_(True)
            elif 'seq_transformer_clip' in name:
                param.requires_grad_(True)
            elif 'ctrans_to_768' in name:
                param.requires_grad_(True)
            elif 'audio_encoder' in name:
                param.requires_grad_(True)

    @property
    def learning_rate(self):
        optimizer = self.optimizers()
        return optimizer.param_groups[0]["lr"]

    def parse_batch_train(self, batch):
        f_frames, o_frames, cap, label, audio, valid_list = batch
        return f_frames, o_frames, cap, label, audio, valid_list

    def mean_pooling_for_similarity_visual(self, visual_output, valid_list):
        video_mask_un = valid_list.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def forward(self, batch, batch_idx):
        f_frames, o_frames, cap, label, audio, valid_list = self.parse_batch_train(batch)

        cap_feature_flag = 0
        imageface_feature_flag = 0
        imagescene_feature_flag = 0
        audio_feature_flag = 0

        if cap is not None:
            cap_feature_flag = 1
            cap_t = [clip.tokenize(cap_i.replace("_", " ")) for cap_i in cap]
            cap_tokenized_prompts = torch.cat(cap_t).to(self.logit_scale.device)

            with torch.no_grad():
                cap_embedding = self.token_embedding(cap_tokenized_prompts).type(self.dtype)

            cap_features = self.cap_encoder(cap_embedding, cap_tokenized_prompts)
            cap_features = self.ctrans_to_768(cap_features)
        else:
            cap_features = torch.zeros(len(cap), 768).to(self.device)

        if f_frames is not None:
            imageface_feature_flag = 1
            b, timestep, channel, h, w = f_frames.shape
            f_frames = f_frames.view(b * timestep, channel, h, w)
            video_features = self.image_encoder(f_frames.type(self.dtype))
            video_features = video_features.view(len(valid_list), timestep, video_features.size(-1))
        else:
            video_features = torch.zeros(len(valid_list), 16, 768).to(self.device)

        if o_frames is not None:
            imagescene_feature_flag = 1
            b, timestep, channel, h, w = o_frames.shape
            o_frames = o_frames.view(b * timestep, channel, h, w)
            o_video_features = self.image_encoder_s(o_frames.type(self.dtype))
            o_video_features = o_video_features.view(len(valid_list), timestep, o_video_features.size(-1))
        else:
            o_video_features = torch.zeros_like(video_features)

        if audio is not None:
            audio_feature_flag = 1
            audio_feature = self.audio_encoder(audio)
        else:
            audio_feature = torch.zeros(len(cap), 768).to(self.device)

        f_features = torch.split(video_features, 1, dim=1)
        o_features = torch.split(o_video_features, 1, dim=1)
        count = len(f_features)
        f_features_n = []
        o_features_n = []

        for i in range(count):
            f_temp = f_features[i].squeeze(1)
            o_temp = o_features[i].squeeze(1)
            in_i = torch.stack([f_temp, o_temp], dim=0).to(f_temp.device)
            output, (h, c) = self.lstm(in_i)
            ou_i = output + in_i
            oup = torch.split(ou_i, 1, dim=0)
            f_features_n.append(oup[0].squeeze(0))
            o_features_n.append(oup[1].squeeze(0))

        f_features = torch.stack(f_features_n, dim=1)
        o_features = torch.mean(torch.stack(o_features_n, dim=1), dim=1)

        visual_output = f_features
        visual_output_original = visual_output
        seq_length = visual_output.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
        position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
        frame_position_embeddings = self.pencoding(position_ids)
        visual_output = visual_output + frame_position_embeddings
        extended_video_mask = (1.0 - valid_list.unsqueeze(1)) * -1000000.0
        extended_video_mask = extended_video_mask.expand(-1, valid_list.size(1), -1)
        visual_output = visual_output.permute(1, 0, 2)
        visual_output = self.transformer_clip(visual_output, extended_video_mask)
        visual_output = visual_output.permute(1, 0, 2)
        visual_output = visual_output + visual_output_original
        visual_output = self.mean_pooling_for_similarity_visual(visual_output, valid_list)

        all_features = torch.stack([
            visual_output.unsqueeze(0),
            o_features.unsqueeze(0),
            cap_features.unsqueeze(0),
            audio_feature.unsqueeze(0)
        ], dim=0)

        valid = [imageface_feature_flag, imagescene_feature_flag, cap_feature_flag, audio_feature_flag]
        valid = torch.tensor(valid).repeat(len(valid_list), 1).to(self.device)
        mask = (1.0 - valid.unsqueeze(1)) * -1000000.0
        mask = mask.expand(-1, valid.size(1), -1).to(self.device)
        all_features = self.seq_transformer_clip(all_features + self.cpencoding(torch.arange(4, dtype=torch.long, device=all_features.device).unsqueeze(0).expand(all_features.shape[1], -1)).permute(1, 0, 2), mask)
        all_features = self.mean_pooling_for_similarity_visual(all_features.permute(1, 0, 2), valid)
        all_features = all_features / all_features.norm(dim=-1, keepdim=True)

        prompts, prompt_pool, prompt_key = self.prompt_pool(all_features, self.training)
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * all_features @ text_features.t()
        l1_reg = 0.01 * self.prompt_pool.alpha[0].weight.abs().sum()

        loss = F.cross_entropy(logits, label) + l1_reg if label is not None else None
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, label)

        return loss, preds, acc

    def training_step(self, batch, batch_nb):
        loss, preds, acc = self.forward(batch, batch_nb)
        self.log("learning_rate", self.learning_rate, prog_bar=True, sync_dist=True)
        self.writer.add_scalar("learning_rate", self.learning_rate, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, acc = self.forward(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        self.writer.add_scalar("val_loss", loss, self.global_step)
        self.writer.add_scalar("val_acc", acc, self.global_step)
        self.writer.add_histogram("freq", np.array(freq), self.global_step)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.000001, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, threshold=0.001)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def on_test_epoch_end(self):
        print(self.confusion_matrix)
        labels_11 = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'contempt', 'anxiety', 'helplessness', 'disappointment']
        test_number_11 = [1487, 467, 431, 1473, 1393, 638, 1958, 1473, 1393, 638, 1958]
        name_11 = 'MAFW (11 classes)'
        display_cls_confusion_matrix(self.confusion_matrix, labels_11, test_number_11, name_11, pltname)
        display_cls_confusion_matrix_6(self.confusion_matrix, labels_11, test_number_11, name_11, pltname)
        print(freq)

        tsne = TSNE(n_components=2, random_state=42, perplexity=6)
        tsne_results3 = tsne.fit_transform(self.f3.cpu())
        tsne_results4 = tsne.fit_transform(self.f4.cpu())

        plt.figure(figsize=(10, 6))
        plt.scatter(tsne_results3[:, 0], tsne_results3[:, 1], c=self.f3l.cpu())
        plt.colorbar()
        plt.savefig("r3.png")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.scatter(tsne_results4[:, 0], tsne_results4[:, 1], c=self.f4l.cpu())
        plt.colorbar()
        plt.savefig("r4.png")
        plt.show()

        tsne = TSNE(n_components=2, random_state=42)
        tsne_results2 = tsne.fit_transform(self.f2.cpu())
        tsne_results_orf = tsne.fit_transform(self.orf.cpu())

        plt.figure(figsize=(10, 6))
        plt.scatter(tsne_results2[:, 0], tsne_results2[:, 1], c=self.labels.cpu())
        plt.colorbar()
        plt.savefig("r2.png")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.scatter(tsne_results_orf[:, 0], tsne_results_orf[:, 1], c=self.labels.cpu())
        plt.colorbar()
        plt.savefig("orf.png")
        plt.show()

    def test_step(self, batch, batch_idx):
        f_frames, o_frames, cap, label, audio, valid_list = self.parse_batch_train(batch)
        loss, preds, acc = self.forward(batch, batch_idx)
        

        y_pred = preds.cpu().numpy()
        y_true = label.cpu().numpy()
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=True)

        war_0, war_1, war_2, war_3, war_4, war_5, war_6 = [report[str(i)]['recall'] if str(i) in report else 0.0 for i in range(7)]
        uar = report['macro avg']['recall']
        war = report['accuracy']

        self.log('test_war_0', war_0, prog_bar=True, sync_dist=True)
        self.log('test_war_1', war_1, prog_bar=True, sync_dist=True)
        self.log('test_war_2', war_2, prog_bar=True, sync_dist=True)
        self.log('test_war_3', war_3, prog_bar=True, sync_dist=True)
        self.log('test_war_4', war_4, prog_bar=True, sync_dist=True)
        self.log('test_war_5', war_5, prog_bar=True, sync_dist=True)
        self.log('test_war_6', war_6, prog_bar=True, sync_dist=True)

        for i in range(len(y_pred)):
            self.confusion_matrix[y_true[i], y_pred[i]] += 1

        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        self.log('test_war', war, prog_bar=True, sync_dist=True)
        self.log('test_uar', uar, prog_bar=True, sync_dist=True)

