import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 设置颜色和标签
color = ['#D45951', '#B78D12', '#126E82', '#207F4C', '#3CDBD3']
labels = ['[1.0, 0.9)', '[0.9, 0.8)', '[0.8, 0.7)', '[0.7, 0.6)', '[0.6, 0.0]']

# 创建颜色标签补丁
patches = [mpatches.Patch(color=color[i], label=labels[i]) for i in range(len(color))]

def save_bar_plot(go_id, go_score, go_color, pro_png_path):
    plt.figure(figsize=(30, 25), clear=True)
    plt.title("Predicted GO terms in barplot", fontsize=50, fontname="Arial")
    plt.xlabel("Confidence score", fontname="Arial", fontsize=36)
    plt.barh(range(len(go_id)), go_score, color=go_color)
    plt.yticks(range(len(go_id)), go_id, fontname="Arial", fontsize=24)
    plt.xticks(fontsize=36, fontname="Arial")
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.95])
    ax.legend(handles=patches, bbox_to_anchor=(0.95, 0.98), ncol=3, prop={'size': 20, 'family': 'Arial'})
    plt.savefig(pro_png_path, bbox_inches='tight', dpi=75)
    plt.clf()
    plt.close()

def save_bubble_plot(go_id, go_depth, go_score, go_color, depth_png_path):
    plt.figure(figsize=(25, 25), clear=True)
    plt.title("Depth of predicted GO terms in bubble plot", fontsize=50, fontname="Arial")
    plt.scatter(go_depth, range(len(go_depth)), color=go_color, s=[int(1500 * s) for s in go_score])
    plt.xlabel("Depth", fontsize=36, fontname="Arial")
    plt.yticks(range(len(go_depth)), go_id, fontname="Arial", fontsize=24)
    plt.xticks(fontsize=36, fontname="Arial")
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.95])
    ax.legend(handles=patches, bbox_to_anchor=(0.5, 0.99), ncol=3, prop={'size': 20, 'family': 'Arial'})
    plt.savefig(depth_png_path, bbox_inches='tight', dpi=75)
    plt.clf()
    plt.close()

def create_plots(data_dir, task_id, go_id, go_score, go_color, go_depth, i, go_num, ns):
    pro_png = f'pro{i}_{go_num}{ns}.png'
    depth_png = f'depth{i}_{go_num}{ns}.png'

    if go_num > 30:
        pro_png = f'pro{i}_30{ns}.png'
        depth_png = f'depth{i}_30{ns}.png'

    pro_png_path = os.path.join(data_dir, task_id, pro_png)
    depth_png_path = os.path.join(data_dir, task_id, depth_png)

    if not os.path.exists(pro_png_path):
        save_bar_plot(go_id, go_score, go_color, pro_png_path)
    
    if not os.path.exists(depth_png_path):
        save_bubble_plot(go_id, go_depth, go_score, go_color, depth_png_path)
        
data_dir = './'
task_id = './'
go_id = ['GO:0008150', 'GO:0009987', 'GO:0003674']
go_score = [0.95, 0.85, 0.65]
go_color = ['#D45951', '#B78D12', '#126E82']
go_depth = [3, 2, 1]
i = 1
go_num = len(go_id)
ns = ''

create_plots(data_dir, task_id, go_id, go_score, go_color, go_depth, i, go_num, ns)
