import os

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

from utils.lang_maps import ALL_TARGETS, HR_MAP

root = os.path.join('token_frequency', 'by_language')

with open(os.path.join('token_frequency', 'measures.tsv'), 'w+', encoding='utf-8') as f:
    
    f.write('lang\tcosine\trho_p\trho_p_p\trho_s\trho_s_p\tT_k\tT_k_p\n')

    for lang in ALL_TARGETS:
        v1 = np.load(
            os.path.join(root, f'{lang}.npy')
        )
        
        try:
            v2 = np.load(
                os.path.join(
                    root,
                    f'{"_".join(HR_MAP[lang])}.npy'
                )
            )
        except:
            continue
        
        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        rho_p, rho_p_p = pearsonr(v1, v2)
        rho_s, rho_s_p = spearmanr(v1, v2)
        T_k, T_k_p = kendalltau(v1, v2)
        
        f.write(f'{lang}\t{cosine}\t{rho_p}\t{rho_p_p}\t{rho_s}\t{rho_s_p}\t{T_k}\t{T_k_p}\n')