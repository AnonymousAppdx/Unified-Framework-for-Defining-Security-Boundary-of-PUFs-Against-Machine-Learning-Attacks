import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import seaborn as sns
from collections import Counter

def generate_Q_batch(k, bs, device, dtype):
    return torch.randn(k, bs, device=device, dtype=dtype)

def generate_PUFs(k, N, M_count, batch_size, device, dtype, model, l,ff_params):
    print("Step 1: Generating PUF signatures")
    M = (torch.randint(0, 2, (M_count, k), device=device) * 2 - 1).to(dtype)
    signatures = []
    Q_all = []
    num_batches = (N + batch_size - 1) // batch_size

    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc=f"Generating R with model={model}"):
            Q_list = [generate_Q_batch(k, batch_size, device, dtype) for _ in range(l)]
            Q_all.append(torch.stack([q.cpu() for q in Q_list], dim=0))
            R_batch = compute_PUF_response(Q_list, M, model,ff_params)
            signatures.append(R_batch.to(torch.int8).cpu())

    Q_all = torch.cat(Q_all, dim=2)  # (l, k, N)
    return torch.cat(signatures, dim=1), M, Q_all

import torch

def compute_PUF_response(Q_list, M, model="xor-1", ff_params=None):
    if model.startswith("xor") and "ff" in model or model == "ff":
        l = len(Q_list)
        assert ff_params is not None and l == len(Q_list)
        a, b = ff_params[0]

        M_eval, k = M.shape
        _, N = Q_list[0].shape
        R_paths = []

        # 分 batch 避免爆显存
        batch_size = 128       # 控制 N 的 batch
        M_chunk_size = 128      # 控制 M 的 chunk 大小
        with torch.no_grad():
            for Q_batch in Q_list:  # 每条 FF 路径
                w1 = Q_batch[:a + 1, :]     # (a+1, N)
                w2 = Q_batch[:b + 1, :]     # (b+1, N)
                w2_n = Q_batch[b + 1:, :] if b + 1 < k else None

                R_i = torch.empty((M_eval, N), device=M.device)

                for m_start in range(0, M_eval, M_chunk_size):
                    m_end = min(m_start + M_chunk_size, M_eval)
                    M_chunk = M[m_start:m_end]  # (M_chunk, k)

                    # Step 1: 计算 parity
                    parity = M_chunk
                    for i in reversed(range(k)):
                        parity[:, i] = torch.prod(M_chunk[:, i:], dim=1)
                    f1 = parity[:, :a + 1]  # (M_chunk, a+1)

                    R_i_chunk = torch.empty((m_end - m_start, N), device=M.device)

                    for start in range(0, N, batch_size):
                        end = min(start + batch_size, N)
                        B = end - start

                        w1_batch = w1[:, start:end]  # (a+1, B)
                        w2_batch = w2[:, start:end]  # (b+1, B)
                        if w2_n is not None:
                            w2_n_batch = w2_n[:, start:end]  # (k-b-1, B)

                        r1_sign = torch.sign(f1 @ w1_batch).T  # (B, M_chunk)

                        # C_mod: (B, M_chunk, k)
                        C_mod = M_chunk.unsqueeze(0).expand(B, -1, -1).clone()
                        C_mod[:, :, b] = r1_sign

                        parity2 = C_mod
                        for i in reversed(range(k)):
                            parity2[:, :, i] = torch.prod(C_mod[:, :, i:], dim=2)

                        f2 = parity2[:, :, :b + 1]  # (B, M_chunk, b+1)
                        f2_n = parity2[:, :, b + 1:] if b + 1 < k else None

                        d2 = torch.einsum('bmk,kb->bm', f2, w2_batch)
                        if f2_n is not None:
                            d2_n = torch.einsum('bmk,kb->bm', f2_n, w2_n_batch)
                        else:
                            d2_n = 0

                        total_d = d2 + d2_n  # (B, M_chunk)
                        ff_sign = torch.sign(total_d).T  # (M_chunk, B)
                        R_i_chunk[:, start:end] = ff_sign

                    R_i[m_start:m_end] = R_i_chunk
                del C_mod, parity2, f2, f2_n, total_d, ff_sign
                torch.cuda.empty_cache()  # 强制清理 GPU 缓存
                R_paths.append(R_i)
        acc = R_paths[0]
        for i in range(1, len(R_paths)):
            acc *= R_paths[i]
        return acc  # (M, N)


    elif model.startswith("xor"):
        l = len(Q_list)
        M_count, k = M.shape
        _, N = Q_list[0].shape
        batch_size = 8192  # 可调大小，避免 OOM

        R_out = torch.empty((M_count, N), device=Q_list[0].device)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            R_list = []

            for i in range(l):
                Q_i_batch = Q_list[i][:, start:end]  # shape: (k, batch)
                R_i = torch.sign(M @ Q_i_batch)      # shape: (M, batch)
                R_list.append(R_i)

            R_stack = torch.stack(R_list)            # shape: (l, M, batch)
            R_prod = torch.prod(R_stack, dim=0)      # shape: (M, batch)
            R_out[:, start:end] = R_prod

        return R_out  # shape: (M, N)

    else:
        raise NotImplementedError(f"PUF model '{model}' not supported.")






def parse_model_args(args):
    model = args.model
    if model.startswith("xor-ff"):
        l = int(model.split("-")[2])
    elif model.startswith("xor"):
        l = int(model.split("-")[1])
    else:
        l = 1
    return model, l

def select_PUFs(R_full, device, plot=True):
    print("Step 2: Finding most frequent PUF response pattern")
    with torch.no_grad():
        R_full = R_full.T.contiguous().to(device)
        unique_patterns, inverse, counts = torch.unique(R_full, return_inverse=True, return_counts=True, dim=0)
        max_count, max_idx = torch.max(counts, dim=0)
        dominant_pattern_indices = (inverse == max_idx).nonzero(as_tuple=False).squeeze()

        print(f"Max repetition count: {max_count.item()}")
        print(f"Indices with max repetition: {dominant_pattern_indices.numel()}")

        if plot:
            # 将 group size 的频数分布可视化
            count_vals = counts.cpu().numpy()
            count_freq = Counter(count_vals)

            sizes = sorted(count_freq.keys())
            freqs = [count_freq[s] for s in sizes]

            plt.figure(figsize=(10, 5))
            sns.barplot(x=sizes, y=freqs, color='skyblue')
            plt.xlabel("Group Size (number of PUFs sharing the same response)")
            plt.ylabel("Frequency (number of such groups)")
            plt.title("Distribution of Response Pattern Group Sizes")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('__mid.png')

    return dominant_pattern_indices, max_count.item()

def evaluate_PUFs(Q_all, M_p, dominant_indices, device, dtype, model, ff_params=None):
    print("Step 3: Evaluating bias on all samples and repeated samples")

    with torch.no_grad():
        # 所有样本
        Q_all_device = Q_all.to(device)
        Q_list_all = [Q_all_device[i] for i in range(Q_all_device.shape[0])]
        R_all = compute_PUF_response(Q_list_all, M_p, model, ff_params)
        print(f'R_all shape {R_all.shape}')
        # R_all = (1-R_all)//2
        bias_all = torch.mean(R_all, dim=1)
        abs_bias_all = torch.mean(torch.abs(bias_all))
        print(f"[All] Bias shape: {bias_all.shape}")
        print(f"[All] Absolute bias: {abs_bias_all.item():.4f}")

        # 重复样本
        dominant_indices = dominant_indices.cpu()
        # Q_dup = Q_all[:, :, dominant_indices].to(device)
        # Q_list_dup = [Q_dup[i] for i in range(Q_dup.shape[0])]
        # R_dup = compute_PUF_response(Q_list_dup, M_p, model, ff_params)
        R_dup = R_all[:, dominant_indices]
        # R_dup = (1-R_dup)//2
        print(f'R_dup shape {R_dup.shape}')
        # total = R_dup.numel()
        num_positive = int((R_dup > 0).sum().item())
        bias_dup = torch.mean(R_dup, dim=1)
        abs_bias_dup = torch.mean(torch.abs(bias_dup))
        print(f"[Dominant] Bias shape: {bias_dup.shape}")
        print(f"[Dominant] Absolute bias: {abs_bias_dup.item():.4f}")
        print(f"[Dominant] Number of positive samples: {num_positive}")
        # print(f'[Dominant] Total samples: {total}')
    return bias_all, bias_dup, num_positive


def plot_bias(bias_all, bias_dom, num_positive, model, k, N, M_count, M_eval, num_dominant, foldername="bias_plots"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    if bias_all is not None and bias_dom is not None:
        bias_all_np = bias_all.detach().cpu().numpy()
        bias_dom_np = bias_dom.detach().cpu().numpy()
        abs_bias_all = float(torch.mean(torch.abs(bias_all)))
        abs_bias_dom = float(torch.mean(torch.abs(bias_dom)))

        # 设置风格
        sns.set_theme(style="whitegrid")
        colors = sns.color_palette("muted")

        plt.figure(figsize=(8, 4.5))  # 更适合论文版面
        plt.plot(bias_all_np, label='Without Condition', color=colors[0], linewidth=1.5, alpha=0.8)
        plt.plot(bias_dom_np, label='With Condition', color=colors[1], linewidth=1.5, alpha=0.8)
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)

        plt.xlabel('Challenge Index', fontsize=14, fontweight='bold')
        plt.ylabel('Average Bias', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12, loc='best', frameon=False)

        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.minorticks_on()

        # 更简洁但仍然有信息量的标题
        plt.title(f"{model.upper()} ($k={k}$, $N={N}$, $M={M_count}/{M_eval}$)", fontsize=14)

        plt.tight_layout()

        if not os.path.exists(foldername):
            os.makedirs(foldername)

        # 保存为高清PNG和PDF
        filename_base = f"bias_{model}_k{k}_N{N}_M{M_count}_Mp{M_eval}"
        savepath_png = os.path.join(foldername, filename_base + ".png")
        savepath_pdf = os.path.join(foldername, filename_base + ".pdf")

        plt.savefig(savepath_png, dpi=600)
        plt.savefig(savepath_pdf, dpi=600)
        plt.close()

        print(f"[Saved] Bias plots to {savepath_png} and {savepath_pdf}")


import csv
import os

# 保存 bias 统计值
def save_bias_csv(abs_bias_all, abs_bias_dom, model, k, N, M_count, M_eval, num_dominant, output_file="bias_summary.csv", foldername="bias_plots"):
    header = ["model", "k", "N", "M_count", "M_eval", "abs_bias_all", "abs_bias_dom", "num_samples"]
    row = [model, k, N, M_count, M_eval, f"{abs_bias_all:.6f}", f"{abs_bias_dom:.6f}", num_dominant]
    savepath = os.path.join(foldername, output_file)
    file_exists = os.path.isfile(savepath)
    with open(savepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
    print(f"[Saved] Bias summary to {savepath}")


def main():
    parser = argparse.ArgumentParser(description="PUF Simulation Pipeline")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--N", type=int, default=1000000)
    parser.add_argument("--M_count", type=int, default=10)
    parser.add_argument("--M_eval", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--model", type=str, default="xor-1", help="PUF model: xor-l or ff")
    parser.add_argument("--exp_folder", type=str, default="bias_exp", help="Folder name to store")
    parser.add_argument("--ff_a", type=int, default=20)
    parser.add_argument("--ff_b", type=int, default=50)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    exp_folder = args.exp_folder
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    model, l = parse_model_args(args)
    ff_params = [(args.ff_a, args.ff_b)] if "ff" in model else None

    R_full, M, Q_all = generate_PUFs(args.k, args.N, args.M_count, args.batch_size, device, dtype, model, l,ff_params)
    print(f"Generated Q_all with shape: {Q_all.shape} and R_full with shape: {R_full.shape}")
    dominant_indices, account = select_PUFs(R_full, device)
    print(f'dominant_indices shape {dominant_indices.shape}')
    M_p = (torch.randint(0, 2, (args.M_eval, args.k), device=device) * 2 - 1).to(dtype)
    bias_all, bias_dom, num_positive = evaluate_PUFs(Q_all, M_p, dominant_indices, device, dtype, model, ff_params)
    # print(f"Pr={num_positive/account}")
    print(f'bias dom shape {bias_dom.shape}')
    plot_bias(bias_all, bias_dom, num_positive, model, args.k, args.N, args.M_count, args.M_eval, num_dominant=account, foldername=exp_folder)

    # 保存统计数据
    save_bias_csv(
        abs_bias_all=torch.mean(torch.abs(bias_all)).item(),
        abs_bias_dom=torch.mean(torch.abs(bias_dom)).item(),
        model=model,
        k=args.k,
        N=args.N,
        M_count=args.M_count,
        M_eval=args.M_eval,
        num_dominant=account,
        foldername=exp_folder
    )

if __name__ == "__main__":
    main()