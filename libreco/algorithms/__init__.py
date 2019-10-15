from .user_KNN import userKNN
from .item_KNN import itemKNN
from .SVD import SVD
from .SVDpp import SVDpp
from .Als import Als
from .FM import FmPure, FmFeat
from .superSVD import superSVD
from .NCF import Ncf
from .wide_deep import WideDeep, WideDeepEstimator
from .DeepFM import DeepFmPure, DeepFmFeat
from .BPR import Bpr
from .YouTubeRec import YouTubeRec
try:
    from .superSVD_cy import superSVD_cy
    from .superSVD_cys import superSVD_cys
except ImportError:
    pass
