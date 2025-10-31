import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
# from PIL import Image

def extract_best_mse(input_file):
    epochs = []
    best_mse_values = []

    # 打开并读取txt文件
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # 正则表达式提取每个epoch和对应的best_mse
    pattern = r"epoch:(\d+):best_mse,best_ci:\s*([\d.]+),\s*([\d.-]+)"
    matches = re.findall(pattern, content)

    # 提取数据
    for match in matches:
        epoch = int(match[0])
        best_mse = float(match[1])
        epochs.append(epoch)
        best_mse_values.append(best_mse)



    return epochs, best_mse_values

def plot_best_mse(epochs, best_mse_values):
    # 绘制图表
    plt.figure(figsize=(20, 10))
    plt.plot(epochs, best_mse_values,linestyle='-', color='#54beaa', label="KIBA")

    plt.xlabel("Epochs")
    plt.ylabel("Best_MSE")
    plt.xticks([100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000],fontsize=12)
    plt.yticks(fontsize=16)

    plt.legend()
    plt.show()

# 使用示例：读取数据并绘图
if __name__ == '__main__':
    input_file = "result/output1.txt"  # 修改为您的文件路径
    epochs, best_mse_values = extract_best_mse(input_file)
    plot_best_mse(epochs, best_mse_values)


