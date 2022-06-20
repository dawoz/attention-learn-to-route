from unicodedata import name
from nets.graph_encoder import GraphAttentionEncoder, GraphAttentionEncoderCustom
from transformers import BertModel, BertConfig, BigBirdConfig, BigBirdModel, AlbertConfig, AlbertModel
from options import get_options
import torch.nn as nn
from nets.graph_encoder import MultiHeadAttentionLayer

if __name__ == '__main__':
    opts = get_options()  
    attention_model = nn.Sequential(*(
            MultiHeadAttentionLayer(8, opts.embedding_dim, 512, opts.normalization)
            for _ in range(opts.n_encode_layers)
            ))

    print(attention_model, f'####################### {sum(p.numel() for p in attention_model.parameters() if p.requires_grad)} ############################\n\n')

    for model_class, config_class in [(BertModel, BertConfig)]:
                                        #(BigBirdModel, BigBirdConfig),
                                        #(AlbertModel, AlbertConfig)]:
        model = model_class(
                config=config_class(
                    num_attention_heads=8,
                    hidden_size=opts.embedding_dim,
                    intermediate_size=512,
                    num_hidden_layers=opts.n_encode_layers
                )
            )

        print(model, f'######################## {sum(p.numel() for p in model.parameters() if p.requires_grad)} ###########################\n\n')
