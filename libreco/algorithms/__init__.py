from .user_cf import userCF
from .item_cf import itemCF
from .svd import SVD
from .svdpp import SVDpp
from .als import Als
from .fm import FmPure, FmFeat
from .ncf import Ncf
from .wide_deep import WideDeep, WideDeepEstimator
from .deepfm import DeepFmPure, DeepFmFeat
from .bpr import Bpr
from .din import Din
from .youtube import YouTubeRec
try:
    from .superSVD_cy import superSVD_cy
    from .superSVD_cys import superSVD_cys
except ImportError:
    pass
