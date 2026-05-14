#!/usr/bin/env python3
"""
投影结构有效性验证脚本

实验A（定量）：k近邻地图IoU一致性
  - 验证t-SNE投影近邻场景在道路结构（map mask IoU）上是否更相似
  - 对比投影近邻 vs 随机配对的平均IoU

实验B（定性）：Cluster代表场景可视化
  - 从每个cluster中选取最靠近中心的N个场景
  - 并排展示其多通道地图mask，供论文使用

运行方式（在远程服务器上）：
  python validate_map_structure.py \
    --pkl  /path/to/scene_raster_trainval_val_150scenes.pkl \
    --proj /path/to/projections.csv \
    --out  /path/to/output_dir \
    --method C_raw_tsne \
    --k 10
"""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_val_masks(pkl_path: str) -> dict:
    """
    加载 val pkl，返回 {scene_token: mask_array} 字典。
    mask_array 形状: (H, W, C)，值域 [0,1]，C=3 (divider/drivable/ped)
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    channel_names = ['divider', 'drivable_area', 'ped_crossing']
    masks = {}
    for token, scene in data['scenes'].items():
        ch_list = []
        for ch in channel_names:
            m = scene['masks'].get(ch)
            if m is None:
                m = np.zeros_like(next(iter(scene['masks'].values())))
            ch_list.append((m > 0.5).astype(np.float32))
        # (H, W, C)
        masks[token] = np.stack(ch_list, axis=-1)

    print(f"加载 {len(masks)} 个场景的地图mask, shape示例: {next(iter(masks.values())).shape}")
    return masks


def load_projection(csv_path: str, method: str) -> pd.DataFrame:
    """返回 val 场景的投影坐标 DataFrame，列: scene_token, scene_name, cluster, dim1, dim2"""
    df = pd.read_csv(csv_path)
    df = df[df['method'] == method].copy().reset_index(drop=True)
    print(f"投影方法 {method}: {len(df)} 个场景")
    return df


# ---------------------------------------------------------------------------
# 实验A：定量 - k近邻 map IoU 一致性
# ---------------------------------------------------------------------------

def compute_pairwise_iou(masks: np.ndarray) -> np.ndarray:
    """
    批量计算 N 个 mask 之间的 pairwise IoU（多通道平均）。
    masks: (N, H*W*C) 已展平的二值向量
    返回: (N, N) IoU 矩阵
    """
    N = masks.shape[0]
    iou_matrix = np.zeros((N, N), dtype=np.float32)

    for i in tqdm(range(N), desc="计算 pairwise IoU"):
        intersection = (masks[i] * masks[i:]).sum(axis=1)   # (N-i,)
        union = ((masks[i] + masks[i:]) > 0).sum(axis=1)    # (N-i,)
        iou = intersection / (union + 1e-6)
        iou_matrix[i, i:] = iou
        iou_matrix[i:, i] = iou

    return iou_matrix


def experiment_a(masks_dict: dict, proj_df: pd.DataFrame,
                 k: int = 10, n_random: int = 3000) -> dict:
    """
    验证：投影k近邻的平均map IoU 是否显著高于随机配对。
    """
    # 对齐 scene_token
    common_tokens = [t for t in proj_df['scene_token'] if t in masks_dict]
    proj_sub = proj_df[proj_df['scene_token'].isin(common_tokens)].copy()
    proj_sub = proj_sub.set_index('scene_token').loc[common_tokens].reset_index()
    print(f"\n[实验A] 参与场景数: {len(common_tokens)}, k={k}")

    # 展平 mask → (N, H*W*C)
    mask_arrays = np.stack(
        [masks_dict[t].flatten() for t in proj_sub['scene_token']], axis=0
    )
    coords_2d = proj_sub[['dim1', 'dim2']].values
    N = len(common_tokens)

    # 计算 pairwise IoU
    iou_matrix = compute_pairwise_iou(mask_arrays)
    np.fill_diagonal(iou_matrix, np.nan)

    # 投影 k-NN 近邻的平均 IoU
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords_2d)
    _, indices = nbrs.kneighbors(coords_2d)
    indices = indices[:, 1:]  # 去掉自身

    neighbor_ious = []
    for i in range(N):
        neighbor_ious.append(iou_matrix[i, indices[i]].mean())
    neighbor_ious = np.array(neighbor_ious)

    # 随机配对的平均 IoU（baseline）
    rng = np.random.RandomState(42)
    random_ious = []
    for _ in range(n_random):
        idx = rng.choice(N, k, replace=False)
        # 避免自身
        sample_iou = [iou_matrix[i, idx[idx != i]].mean()
                      for i in range(min(10, N))]
        random_ious.append(np.nanmean(sample_iou))
    random_ious = np.array(random_ious)

    # 全局随机基线（直接取 iou_matrix 上三角均值）
    triu_vals = iou_matrix[np.triu_indices(N, k=1)]
    global_mean_iou = float(np.nanmean(triu_vals))

    # t检验
    t_stat, p_val = stats.ttest_1samp(neighbor_ious, global_mean_iou)

    results = {
        'n_scenes': N,
        'k': k,
        'neighbor_mean_iou': float(neighbor_ious.mean()),
        'neighbor_std_iou': float(neighbor_ious.std()),
        'global_mean_iou': global_mean_iou,
        'improvement_pct': float((neighbor_ious.mean() - global_mean_iou) / global_mean_iou * 100),
        't_statistic': float(t_stat),
        'p_value': float(p_val),
    }

    print(f"  投影近邻平均IoU: {results['neighbor_mean_iou']:.4f} ± {results['neighbor_std_iou']:.4f}")
    print(f"  全局随机基线IoU: {results['global_mean_iou']:.4f}")
    print(f"  IoU提升: +{results['improvement_pct']:.1f}%")
    print(f"  t={results['t_statistic']:.3f}, p={results['p_value']:.4e}")

    # 不同 k 下的结果
    print("\n  不同 k 下的近邻IoU:")
    for kk in [5, 10, 15, 20]:
        nbrs_k = NearestNeighbors(n_neighbors=kk + 1).fit(coords_2d)
        _, idx_k = nbrs_k.kneighbors(coords_2d)
        idx_k = idx_k[:, 1:]
        ious_k = np.array([iou_matrix[i, idx_k[i]].mean() for i in range(N)])
        t_k, p_k = stats.ttest_1samp(ious_k, global_mean_iou)
        sig = '***' if p_k < 0.001 else ('**' if p_k < 0.01 else ('*' if p_k < 0.05 else 'ns'))
        print(f"    k={kk}: mean IoU={ious_k.mean():.4f}, +{(ious_k.mean()-global_mean_iou)/global_mean_iou*100:.1f}%, p={p_k:.4e} {sig}")

    return results, iou_matrix, proj_sub


# ---------------------------------------------------------------------------
# 实验B：定性 - Cluster 代表场景可视化
# ---------------------------------------------------------------------------

def experiment_b(masks_dict: dict, proj_sub: pd.DataFrame,
                 n_per_cluster: int = 3, output_dir: Path = None):
    """
    每个cluster取最靠近中心的 n_per_cluster 个场景，
    生成多通道mask拼图，输出为 cluster_gallery.pdf/png。
    """
    clusters = sorted(proj_sub['cluster'].unique())
    channel_names = ['Divider', 'Drivable Area', 'Ped. Crossing']
    channel_colors = [
        plt.cm.Blues,
        plt.cm.Greens,
        plt.cm.Oranges,
    ]

    print(f"\n[实验B] 生成 {len(clusters)} 个cluster的代表场景图，每cluster取 {n_per_cluster} 个")

    # 每个cluster取最靠近中心的场景
    selected = {}
    for c in clusters:
        sub = proj_sub[proj_sub['cluster'] == c].copy()
        cx, cy = sub['dim1'].mean(), sub['dim2'].mean()
        sub['dist'] = np.sqrt((sub['dim1'] - cx)**2 + (sub['dim2'] - cy)**2)
        sub = sub.sort_values('dist')
        valid = sub[sub['scene_token'].isin(masks_dict)]
        selected[c] = valid['scene_token'].iloc[:n_per_cluster].tolist()
        print(f"  Cluster {c} (n={len(sub)}): 选取 {selected[c]}")

    # 绘图: rows=clusters, cols=n_per_cluster, 每格显示合并后的3通道RGB图
    n_clusters = len(clusters)
    fig, axes = plt.subplots(
        n_clusters, n_per_cluster,
        figsize=(n_per_cluster * 3, n_clusters * 3),
        squeeze=False
    )

    for row, c in enumerate(clusters):
        for col, token in enumerate(selected[c]):
            ax = axes[row][col]
            mask = masks_dict[token]  # (H, W, 3)

            # 合成RGB：divider=蓝, drivable=绿, ped=橙
            rgb = np.zeros((*mask.shape[:2], 3), dtype=np.float32)
            rgb[..., 2] = mask[..., 0]          # divider → 蓝
            rgb[..., 1] = mask[..., 1]          # drivable → 绿
            rgb[..., 0] = mask[..., 2] * 0.8   # ped → 红（橙近似）

            ax.imshow(rgb, origin='upper')
            ax.axis('off')

            # 仅第一列标注 cluster id
            if col == 0:
                ax.set_ylabel(f'Cluster {c}\n(n={len(proj_sub[proj_sub["cluster"]==c])})',
                              fontsize=9, rotation=0, labelpad=55, va='center')
            # 仅第一行标注列序号
            if row == 0:
                ax.set_title(f'Rep. {col+1}', fontsize=9)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0, 0, 1), label='Divider'),
        Patch(facecolor=(0, 1, 0), label='Drivable Area'),
        Patch(facecolor=(0.8, 0, 0), label='Ped. Crossing'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Representative Scenes per Road-Structure Cluster\n(t-SNE projection, 3-channel HD map)',
                 fontsize=11, y=1.01)
    plt.tight_layout()

    out_png = output_dir / 'cluster_gallery.png'
    out_pdf = output_dir / 'cluster_gallery.pdf'
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"  保存到: {out_png}")
    print(f"          {out_pdf}")

    # 同时生成每个cluster的单独大图（方便论文裁剪）
    for c in clusters:
        fig2, axes2 = plt.subplots(1, n_per_cluster,
                                   figsize=(n_per_cluster * 3.5, 3.5))
        if n_per_cluster == 1:
            axes2 = [axes2]
        for col, token in enumerate(selected[c]):
            mask = masks_dict[token]
            rgb = np.zeros((*mask.shape[:2], 3), dtype=np.float32)
            rgb[..., 2] = mask[..., 0]
            rgb[..., 1] = mask[..., 1]
            rgb[..., 0] = mask[..., 2] * 0.8
            axes2[col].imshow(rgb, origin='upper')
            axes2[col].axis('off')
            axes2[col].set_title(f'Rep. {col+1}', fontsize=10)
        plt.suptitle(f'Cluster {c}  (n={len(proj_sub[proj_sub["cluster"]==c])})',
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(output_dir / f'cluster_{c}_gallery.png', dpi=200, bbox_inches='tight')
        plt.close()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='验证道路结构投影有效性')
    parser.add_argument('--pkl',    required=True,
                        help='val pkl 路径: scene_raster_trainval_val_150scenes.pkl')
    parser.add_argument('--proj',   required=True,
                        help='projections.csv 路径')
    parser.add_argument('--out',    required=True,
                        help='输出目录')
    parser.add_argument('--method', default='C_raw_tsne',
                        choices=['A_pca_tsne', 'B_pca_umap', 'C_raw_tsne', 'D_raw_umap', 'E_pca'],
                        help='投影方法')
    parser.add_argument('--k',      type=int, default=10,
                        help='近邻数')
    parser.add_argument('--n_per_cluster', type=int, default=3,
                        help='每个cluster展示的代表场景数')
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    masks_dict = load_val_masks(args.pkl)
    proj_df    = load_projection(args.proj, args.method)

    # 实验A
    results_a, iou_matrix, proj_sub = experiment_a(
        masks_dict, proj_df, k=args.k
    )

    # 实验B
    experiment_b(masks_dict, proj_sub,
                 n_per_cluster=args.n_per_cluster,
                 output_dir=output_dir)

    # 保存实验A结果
    out_json = output_dir / 'validation_results.json'
    with open(out_json, 'w') as f:
        json.dump(results_a, f, indent=2)
    print(f"\n实验A结果已保存: {out_json}")

    # 保存 iou_matrix (可供后续分析)
    np.save(output_dir / 'pairwise_iou_matrix.npy', iou_matrix)
    print(f"pairwise IoU 矩阵已保存: {output_dir / 'pairwise_iou_matrix.npy'}")


if __name__ == '__main__':
    main()
