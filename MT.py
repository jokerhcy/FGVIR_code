from matplotlib import pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = 'arial'
font = dict(family='arial', style='normal', weight='normal', color='black', size=10)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def get_results(datas, num):
    #############num为表格中数据开始的行数################

    results = {}
    for i in range(num - 1, num + 2):
        ifo = list(datas.loc[i])
        results[ifo[0]] = ifo[1:]
    return results


def graph(datas, sub_num, x, subtitle):
    ############x为画图的横坐标、sub_num为子图顺序#############

    plt.subplot(1, 1, sub_num)
    #plt.plot(x, datas['Ours'], "o-", color='r', label="Ours", lw=1.8, ms=8)
    #plt.plot(x, datas['Finetune'], "s-", label="Finetune", lw=1.5, ms=4)
    #plt.plot(x, datas['EWC'], "v-", label="EWC", lw=1.5, ms=4)
    #plt.plot(x, datas['LwF'], "p-", label="LwF", lw=1.5, ms=4)
    #plt.plot(x, datas['iCaRL(NCM)'], "v-", label="iCaRL(NCM)", lw=1.5, ms=4)
    #plt.plot(x, datas['iCaRL-CNN'], "p--", label="iCaRL", lw=1.5, ms=4)
    
    plt.plot(x, datas['joint'], "-", label="joint training", lw=1, ms=1)
    plt.plot(x, datas['joint-OGM'], "-", label="joint training w/ OGM", lw=1, ms=1)
    #plt.plot(x, datas['PASS - 3X'], "v--", label="PASS-3x", lw=1.5, ms=1)

    
    plt.plot(x, datas['independent'], "--", label="independent training", lw=1, ms=1)
    #plt.plot(x, datas['PASS'], "v--", label="PASS", lw=1.5, ms=1)
    #plt.plot(x, datas['FeTrIL'], "p-", label="FeTrIL", lw=1.5, ms=4)
    #plt.plot(x, datas['FOSTER'], "p--", label="FOSTER", lw=1.5, ms=4)
    #plt.plot(x, datas['SSRE-F'], "p-", label="SSRE-F", lw=1.5, ms=4)
    #plt.grid(color='gray', linestyle='--', linewidth=0.5, axis='both')
    plt.xlabel('Number of activities', fontdict=font, fontsize=12)
    plt.ylabel('Top-1 Accuracy(%)', fontdict=font, fontsize=12)
    plt.xticks(x)
    plt.title(subtitle, fontdict=dict(family='arial', style='normal', weight='normal', color='black', size=10))


"""    plt.subplot(1,3,2)
    plt.plot(x, datas['LwF1'], "o-", label="LwF")
    plt.plot(x, datas['差异性特征蒸馏+LwF1'], "s-", label="差异性特征蒸馏+LwF")
    plt.plot(x, datas['微调1'], "v-", label="微调")
    plt.plot(x, datas['差异性特征蒸馏+微调1'], "p-", label="差异性特征蒸馏+微调")
    plt.xlabel('行为类别数量')
    plt.ylabel('准确率(%)')
    plt.grid()
    plt.title("(b)")

    plt.subplot(1,3,3)
    plt.plot(x, datas['LwF2'], "o-", label="LwF")
    plt.plot(x, datas['差异性特征蒸馏+LwF2'], "s-", label="差异性特征蒸馏+LwF")
    plt.plot(x, datas['微调2'], "v-", label="微调")
    plt.plot(x, datas['差异性特征蒸馏+微调2'], "p-", label="差异性特征蒸馏+微调")
    plt.xlabel('行为类别数量')
    plt.ylabel('准确率(%)')
    plt.grid()
    plt.title("(c)")"""


def remove_nan(data):
    data_new = {}
    for key, value in data.items():
        data_new[key] = list(a for a in value if a == a)
    return data_new


#df = pd.read_excel('forgetting_curve.xlsx', header=None, sheet_name='Sheet1')
df = pd.read_excel('PASS_F_con.xlsx', header=None, sheet_name='Sheet1')
data1 = get_results(df, 2)
"""data2 = get_results(df, 17)
data3 = get_results(df, 32)
data2 = remove_nan(data2)
data3 = remove_nan(data3)"""

fig = plt.figure(figsize=(5, 4.5))
#graph(data1, 1, x=[16, 18, 20, 22, 24, 26, 28, 30, 32], subtitle="RGB+Flow+Acc+Gyro (16+8*2)")
#graph(data1, 1, x=[16, 18, 20, 22, 24, 26, 28, 30, 32], subtitle="Performance drop")
graph(data1, 1, x=[16, 18, 20, 22, 24, 26, 28, 30, 32], subtitle="")
plt.tight_layout()
plt.subplots_adjust(top=0.95, wspace=0.6)
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, ncol=1, frameon=True,loc='upper right',bbox_to_anchor=(0.97,0.95),fontsize=12)
plt.savefig('./mask_curve.png', dpi=600)
