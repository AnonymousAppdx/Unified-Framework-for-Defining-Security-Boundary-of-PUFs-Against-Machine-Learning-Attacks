import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 读取数据
df = pd.read_csv('bias_xor64_detailed_2/bias_summary.csv')

output_folder = 'bias_xor64_detailed_2'
os.makedirs(output_folder, exist_ok=True)

# model_order = sorted(df['k'].unique())
model_order = sorted(df['model'].unique(), key=lambda x: (x.startswith("xor-ff"), int(x.split("-")[-1])))
sns.set_theme(style="whitegrid")
colors = sns.color_palette("muted", n_colors=len(model_order))

plt.figure(figsize=(8, 5))  # 比例适合论文版面

for idx, model in enumerate(model_order):
    sub_df = df[df['model'] == model]
    plt.plot(
        sub_df['M_count'],
        0.5 * sub_df['abs_bias_dom'],
        marker='o',
        label=f"$model={model}$",
        linewidth=1.5,
        markersize=5,
        color=colors[idx]
    )

plt.xlabel(r'$M$ (Number of PUFs Selected)', fontsize=14, fontweight='bold')
plt.ylabel(r'$0.5 \times$ Absolute Dominant Bias', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.minorticks_on()
plt.tight_layout()

# —— 核心变化：图例放到图外 ——
plt.legend(
    fontsize=10,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5),
    borderaxespad=0.,
    frameon=False
)

# 调整子图区域，留出图例位置
plt.subplots_adjust(right=0.75)

savepath_png = os.path.join(output_folder, 'better_abs_bias_xor1_detailed_flow_adjusted.png')
savepath_pdf = os.path.join(output_folder, 'better_abs_bias_xor1_detailed_flow_adjusted.pdf')
plt.savefig(savepath_png, dpi=600, bbox_inches='tight')
plt.savefig(savepath_pdf, dpi=600, bbox_inches='tight')
plt.close()

print(f"[Saved] Plot to {savepath_png} and {savepath_pdf}")
