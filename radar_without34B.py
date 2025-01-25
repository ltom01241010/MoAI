import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Model list
models = ['R2T2-MoAI-7B', 'MoAI-7B', 'Cambrian1-7B', 'LLaVA-NeXT-34B', 'Mini-Gemini-HD-8B']
metrics = ['MMBench', 'MME-P', 'SQA-IMG', 'AI2D', 'TextVQA', 'GQA', 'CVBench-2D', 'CVBench-3D']

# Values array
values = np.array([
    [85.2, 1785.5, 88.3, 85.0, 73.5, 77.0, 77.9, 69.2],  # R2T2-MoAI-7B
    [79.3, 1714.0, 83.5, 78.6, 67.8, 70.2, 71.2, 59.3],  # MoAI-7B
    # [68.8, 1567.4, 68.4, 67.3, 65.8, 63.4, 60.2, 57.5],  # ShareGPT4V-7B
    [75.7, 1610.4, 79.3, 73.6, 72.8, 64.3, 72.5, 71.8], # Cambrian1-13B
    [79.3, 1633.2, 81.8, 74.9, 69.5, 67.1, 73.0, 74.8], # LLaVA-NeXT-34B
    # [70.0, 1575.0, 73.5, 70.0, 67.1, 65.4, 62.7, 65.7],  # LLaVA-NeXT-13B
    [72.7, 1606.0, 75.1, 73.5, 70.2, 64.5, 62.2, 63.0]  # Mini-Gemini-HD-8B
])

# Define custom ranges for each metric
metric_ranges = {
    'MMBench':    {'min': 60, 'max': 90, 'step': 6},
    'MME-P':      {'min': 1430, 'max': 1830, 'step': 80},
    'SQA-IMG':    {'min': 62, 'max': 92, 'step': 6},
    'AI2D':       {'min': 62, 'max': 87, 'step': 6},
    'TextVQA':    {'min': 52, 'max': 77, 'step': 5},
    'GQA':        {'min': 55, 'max': 80, 'step': 5},
    'CVBench-2D': {'min': 55, 'max': 80, 'step': 5},
    'CVBench-3D': {'min': 52, 'max': 72, 'step': 4}
}

plt.rcParams['font.family'] = 'Arial'
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)

def normalize_value(value, metric):
    min_val = metric_ranges[metric]['min']
    max_val = metric_ranges[metric]['max']
    return (value - min_val) / (max_val - min_val) * 100

normalized_values = np.zeros_like(values)
for i, metric in enumerate(metrics):
    normalized_values[:, i] = normalize_value(values[:, i], metric)

normalized_values = np.concatenate((normalized_values, normalized_values[:, [0]]), axis=1)
angles = np.concatenate((angles, [angles[0]]))

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='polar')
ax.set_facecolor('#e5e5e5')
fig.patch.set_facecolor('#e5e5e5')

colors = ['#FF9999', '#FFB366', '#4682B4', '#3CB371', '#40E0D0']

for i, model in enumerate(models):
    ax.plot(angles, normalized_values[i], '-', linewidth=2, label=model, color=colors[i])
    ax.fill(angles, normalized_values[i], alpha=0.25, color=colors[i])
    
    for j, value in enumerate(values[i]):
        if j < len(metrics):
            angle = angles[j]
            radius = normalized_values[i][j]
            label_radius = radius + 5
            
            if i in [0, 1]:  # Show values for R2T2-MoAI-7B and MoAI-7B on outside
                label = f'{value:.1f}'
                ax.text(angle, label_radius, label,
                    ha='center', va='center',
                    color='black',
                    fontsize=18,
                    weight='bold')
            elif i == 2:  # Show values for ShareGPT4V-7B on inside
                label = f'{value:.1f}'
                ax.text(angle, radius - 5, label,
                    ha='center', va='center',
                    color='black',
                    fontsize=18,
                    weight='bold')

# Add improvement arrows between MoAI-7B and R2T2-MoAI-7B
moai_idx = 1
r2t2_idx = 0

for j in range(len(metrics)):
    start_angle = angles[j]
    start_radius = normalized_values[moai_idx][j]
    end_radius = normalized_values[r2t2_idx][j]
    
    if end_radius - start_radius > 5:
        ax.annotate('',
                   xy=(start_angle, end_radius),
                   xytext=(start_angle, start_radius),
                   arrowprops=dict(arrowstyle='-|>',
                                 color='#FF0000',
                                 lw=3,
                                 alpha=0.9,
                                 mutation_scale=20))

improvement_arrow = mlines.Line2D([], [], color='#FF0000', marker='>',
                                linestyle='-', linewidth=4,
                                markersize=10, label='Improvement by R2-T2')

ax.set_xticks(angles[:-1])
label_padding = 1.07

for label, angle in zip(metrics, angles[:-1]):
    if angle == 0:
        ha, va = 'left', 'center'
    elif 0 < angle < np.pi/2:
        ha, va = 'left', 'bottom'
    elif angle == np.pi/2:
        ha, va = 'center', 'bottom'
    elif np.pi/2 < angle < np.pi:
        ha, va = 'right', 'bottom'
    elif angle == np.pi:
        ha, va = 'right', 'center'
    elif np.pi < angle < 3*np.pi/2:
        ha, va = 'right', 'top'
    elif angle == 3*np.pi/2:
        ha, va = 'center', 'top'
    else:
        ha, va = 'left', 'top'
    
    ax.text(angle, label_padding * ax.get_rmax(), label,
            ha=ha, va=va,
            fontsize=22,
            weight='bold')

ax.set_xticklabels([])
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
ax.tick_params(axis='y', labelsize=12)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels([])

handles, labels = ax.get_legend_handles_labels()
handles.append(improvement_arrow)

# Move legend to top
plt.legend(handles=handles, 
          loc='upper center',
          bbox_to_anchor=(0.5, 1.10),
          fontsize=16, 
          frameon=False,
          ncol=6)

plt.tight_layout()
plt.savefig('model_comparison_with_arrows.png', dpi=300, bbox_inches='tight')
plt.close()
