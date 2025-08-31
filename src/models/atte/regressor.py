# Modules -----------------------------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn

# External functions and utilities ----------------------------------------------------------------------------------------#

# Custom functions --------------------------------------------------------------------------------------------------------#
from src.models.atte.layers.tokenizer import QuantileFeatureTokenizer,  QuantileCategoricalEmbedding


# Custom attention-base regressor -----------------------------------------------------------------------------------------#
class ATTE(nn.Module):
    """
    ________________________________________________________________________________________________________________________
    ATTE: Astronomical Transformer for Tabular Estimation
    ________________________________________________________________________________________________________________________
    Description: 

    ________________________________________________________________________________________________________________________
    """
    def __init__(self):
        super(ATTE, self).__init__()

        self.qft = QuantileFeatureTokenizer()

    def forward(self, x_continous, x_categorical, target, ):


    def set_cat_embbedings(self, X:torch.Tensor, y:torch.Tensor, dict_classes, embedding_dim:int ) -> None:
        
        self.qce = QuantileCategoricalEmbedding(dict_classes, embedding_dim).fit(X, y, self.device)
    
    def set_con_embbedings(self, X:torch.Tensor, y:torch.Tensor, cont_emb_dim:int) -> None:

        self.qft = QuantileFeatureTokenizer(X, embedding_dim= cont_emb_dim, n_quantiles = 1000)
