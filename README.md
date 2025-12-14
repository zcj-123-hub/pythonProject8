import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import os
import seaborn as sns
from torch.optim.lr_scheduler import StepLR
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# 设置中文字体（修复中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 使用黑体和备用字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 创建数据目录
os.makedirs('./results', exist_ok=True)

print("正在加载Fashion-MNIST数据集...")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 数据增强（仅用于训练）
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载Fashion-MNIST数据集
train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

# 划分验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 类别名称
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"测试集大小: {len(test_dataset)}")


# 评估函数
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'targets': all_targets
    }


# 训练函数
def train_model(model, train_loader, val_loader, optimizer, criterion,
                scheduler=None, epochs=10, device='cpu'):
    model.train()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        avg_train_accuracy = 100 * correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        if scheduler:
            scheduler.step()

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, '
              f'Train Acc: {avg_train_accuracy:.2f}%, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.2f}%')

    return train_losses, train_accuracies, val_losses, val_accuracies


# 创新CNN模型
class InnovativeCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(InnovativeCNN, self).__init__()

        # 初始卷积层
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 中间卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 高级分类器模型
class AdvancedClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(AdvancedClassifier, self).__init__()

        # 卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# 运行单个实验
def run_experiment(model_name, model_class, optimizer_name, optimizer_class,
                   lr=0.001, epochs=15, device='cpu'):
    print(f"\n{'=' * 60}")
    print(f"实验: {model_name} + {optimizer_name}优化器")
    print(f"{'=' * 60}")

    # 创建模型
    model = model_class().to(device)

    # 定义优化器
    if optimizer_name == 'SGD':
        optimizer = optimizer_class(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == 'AdamW':
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = optimizer_class(model.parameters(), lr=lr)

    # 学习率调度器
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, optimizer, criterion,
        scheduler, epochs, device
    )

    # 在测试集上评估
    test_results = evaluate_model(model, test_loader, device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'model_name': model_name,
        'optimizer_name': optimizer_name,
        'model': model,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'test_results': test_results,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'best_val_acc': max(val_accuracies) if val_accuracies else 0
    }


# 绘制单个实验的训练曲线
def plot_training_curves(experiment_results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    epochs = range(1, len(experiment_results['train_losses']) + 1)
    axes[0].plot(epochs, experiment_results['train_losses'], 'b-', label='训练损失', linewidth=2)
    axes[0].plot(epochs, experiment_results['val_losses'], 'r-', label='验证损失', linewidth=2)
    axes[0].set_xlabel('轮次', fontsize=12)
    axes[0].set_ylabel('损失', fontsize=12)
    axes[0].set_title(f'{experiment_results["model_name"]} - {experiment_results["optimizer_name"]}', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(epochs, experiment_results['train_accuracies'], 'b-', label='训练准确率', linewidth=2)
    axes[1].plot(epochs, experiment_results['val_accuracies'], 'r-', label='验证准确率', linewidth=2)
    axes[1].set_xlabel('轮次', fontsize=12)
    axes[1].set_ylabel('准确率 (%)', fontsize=12)
    axes[1].set_title(f'{experiment_results["model_name"]} - {experiment_results["optimizer_name"]}', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# 绘制所有实验的对比图
def plot_comprehensive_comparison(all_results):
    # 准备数据
    models = []
    optimizers = []
    accuracies = []
    f1_scores = []
    best_val_accs = []

    for result in all_results:
        models.append(result['model_name'])
        optimizers.append(result['optimizer_name'])
        accuracies.append(result['test_results']['accuracy'])
        f1_scores.append(result['test_results']['f1'])
        best_val_accs.append(result['best_val_acc'])

    # 创建DataFrame用于绘图
    df = pd.DataFrame({
        'Model': models,
        'Optimizer': optimizers,
        'Test Accuracy': accuracies,
        'F1 Score': f1_scores,
        'Best Val Accuracy': best_val_accs
    })

    # 1. 测试准确率对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1.1 模型+优化器组合的准确率对比
    unique_combinations = [f"{m}\n({o})" for m, o in zip(models, optimizers)]
    axes[0, 0].bar(range(len(unique_combinations)), accuracies, color=['steelblue', 'lightcoral', 'lightgreen'] * 2)
    axes[0, 0].set_xlabel('模型+优化器组合', fontsize=12)
    axes[0, 0].set_ylabel('测试准确率', fontsize=12)
    axes[0, 0].set_title('不同模型+优化器组合的测试准确率对比', fontsize=14)
    axes[0, 0].set_xticks(range(len(unique_combinations)))
    axes[0, 0].set_xticklabels(unique_combinations, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

    # 1.2 不同优化器在创新CNN上的表现
    cnn_results = [r for r in all_results if r['model_name'] == '创新CNN']
    cnn_opt_names = [r['optimizer_name'] for r in cnn_results]
    cnn_acc = [r['test_results']['accuracy'] for r in cnn_results]

    axes[0, 1].bar(cnn_opt_names, cnn_acc, color=['steelblue', 'lightcoral', 'lightgreen'])
    axes[0, 1].set_xlabel('优化器类型', fontsize=12)
    axes[0, 1].set_ylabel('测试准确率', fontsize=12)
    axes[0, 1].set_title('优化器在创新CNN上的表现对比', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(cnn_acc):
        axes[0, 1].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

    # 1.3 不同优化器在高级分类器上的表现
    adv_results = [r for r in all_results if r['model_name'] == '高级分类器']
    adv_opt_names = [r['optimizer_name'] for r in adv_results]
    adv_acc = [r['test_results']['accuracy'] for r in adv_results]

    axes[1, 0].bar(adv_opt_names, adv_acc, color=['steelblue', 'lightcoral', 'lightgreen'])
    axes[1, 0].set_xlabel('优化器类型', fontsize=12)
    axes[1, 0].set_ylabel('测试准确率', fontsize=12)
    axes[1, 0].set_title('优化器在高级分类器上的表现对比', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    for i, v in enumerate(adv_acc):
        axes[1, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

    # 1.4 最佳验证准确率对比
    x = np.arange(len(all_results))
    width = 0.35

    axes[1, 1].bar(x - width / 2, [r['best_val_acc'] for r in all_results], width, label='最佳验证准确率', alpha=0.8)
    axes[1, 1].bar(x + width / 2, [r['test_results']['accuracy'] * 100 for r in all_results], width,
                   label='测试准确率', alpha=0.8)
    axes[1, 1].set_xlabel('模型+优化器组合', fontsize=12)
    axes[1, 1].set_ylabel('准确率 (%)', fontsize=12)
    axes[1, 1].set_title('验证集vs测试集准确率对比', fontsize=14)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f"{m}\n({o})" for m, o in zip(models, optimizers)],
                               rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('模型与优化器对比实验 - 综合性能分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./results/comprehensive_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 训练过程对比图（损失曲线）
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for idx, result in enumerate(all_results):
        row = idx // 3
        col = idx % 3

        epochs = range(1, len(result['train_losses']) + 1)
        axes[row, col].plot(epochs, result['train_losses'], 'b-', label='训练损失', linewidth=2)
        axes[row, col].plot(epochs, result['val_losses'], 'r-', label='验证损失', linewidth=2)
        axes[row, col].set_xlabel('轮次', fontsize=10)
        axes[row, col].set_ylabel('损失', fontsize=10)
        title = f"{result['model_name']} + {result['optimizer_name']}"
        axes[row, col].set_title(title, fontsize=12)
        axes[row, col].legend(fontsize=9)
        axes[row, col].grid(True, alpha=0.3)

    plt.suptitle('不同模型+优化器组合的训练过程对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./results/training_process_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    return df


# 生成详细报告
def generate_detailed_report(all_results, df):
    report = f"""
    Fashion-MNIST图像分类对比实验报告
    =========================================
    实验时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
    实验配置: 2种模型 × 3种优化器 = 6组实验

    数据集信息:
    - 训练集大小: {len(train_dataset)}
    - 验证集大小: {len(val_dataset)}
    - 测试集大小: {len(test_dataset)}
    - 类别数: 10

    模型配置:
    - 创新CNN: 3层卷积 + 3层全连接，参数量约{all_results[0]['total_params']:,}
    - 高级分类器: 3层卷积块 + 3层全连接，参数量约{all_results[3]['total_params']:,}

    优化器配置:
    - Adam: lr=0.001
    - SGD: lr=0.01, momentum=0.9, weight_decay=1e-4
    - RMSprop: lr=0.001

    实验结果汇总:
    """

    # 添加详细结果
    report += "\n详细性能指标:\n"
    report += "-" * 80 + "\n"
    report += "模型                 优化器     测试准确率   F1分数      最佳验证准确率\n"
    report += "-" * 80 + "\n"

    for result in all_results:
        report += f"{result['model_name']:15} {result['optimizer_name']:10} "
        report += f"{result['test_results']['accuracy']:.4f}      "
        report += f"{result['test_results']['f1']:.4f}       "
        report += f"{result['best_val_acc']:.2f}%\n"

    # 分析最佳组合
    best_result = max(all_results, key=lambda x: x['test_results']['accuracy'])
    report += f"\n最佳组合: {best_result['model_name']} + {best_result['optimizer_name']}优化器\n"
    report += f"测试准确率: {best_result['test_results']['accuracy']:.4f}\n"

    # 分析趋势
    report += "\n观察与结论:\n"
    report += "1. 优化器性能对比:\n"

    # 计算每种优化器的平均性能
    optimizer_stats = {}
    for result in all_results:
        opt = result['optimizer_name']
        if opt not in optimizer_stats:
            optimizer_stats[opt] = {'acc': [], 'count': 0}
        optimizer_stats[opt]['acc'].append(result['test_results']['accuracy'])
        optimizer_stats[opt]['count'] += 1

    for opt, stats in optimizer_stats.items():
        avg_acc = np.mean(stats['acc'])
        report += f"   - {opt}: 平均准确率 = {avg_acc:.4f}\n"

    # 保存报告
    with open('./results/experiment_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    # 保存CSV格式结果
    df.to_csv('./results/experiment_results.csv', index=False, encoding='utf-8-sig')

    print("实验报告已保存为 './results/experiment_report.txt'")
    print("详细结果已保存为 './results/experiment_results.csv'")


# 主函数
def main():
    print("开始图像分类对比实验...")
    print("实验设计: 创新CNN vs 高级分类器 × 三种优化器(Adam, SGD, RMSprop)")

    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 定义所有实验组合
    experiments = [
        {'model_name': '创新CNN', 'model_class': InnovativeCNN, 'optimizer_name': 'Adam',
         'optimizer_class': optim.Adam},
        {'model_name': '创新CNN', 'model_class': InnovativeCNN, 'optimizer_name': 'SGD', 'optimizer_class': optim.SGD},
        {'model_name': '创新CNN', 'model_class': InnovativeCNN, 'optimizer_name': 'RMSprop',
         'optimizer_class': optim.RMSprop},
        {'model_name': '高级分类器', 'model_class': AdvancedClassifier, 'optimizer_name': 'Adam',
         'optimizer_class': optim.Adam},
        {'model_name': '高级分类器', 'model_class': AdvancedClassifier, 'optimizer_name': 'SGD',
         'optimizer_class': optim.SGD},
        {'model_name': '高级分类器', 'model_class': AdvancedClassifier, 'optimizer_name': 'RMSprop',
         'optimizer_class': optim.RMSprop},
    ]

    # 运行所有实验
    all_results = []

    for i, exp in enumerate(experiments):
        # 设置学习率
        lr = 0.01 if exp['optimizer_name'] == 'SGD' else 0.001

        # 运行实验
        result = run_experiment(
            model_name=exp['model_name'],
            model_class=exp['model_class'],
            optimizer_name=exp['optimizer_name'],
            optimizer_class=exp['optimizer_class'],
            lr=lr,
            epochs=15,
            device=device
        )

        all_results.append(result)

        # 保存训练曲线图
        plot_training_curves(
            result,
            f'./results/{exp["model_name"]}_{exp["optimizer_name"]}_training_curves.png'
        )

    # 生成综合对比图
    print("\n" + "=" * 60)
    print("生成综合对比图表...")
    print("=" * 60)

    df = plot_comprehensive_comparison(all_results)

    # 生成详细报告
    generate_detailed_report(all_results, df)

    # 打印总结
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)

    # 找出最佳组合
    best_result = max(all_results, key=lambda x: x['test_results']['accuracy'])
    print(f"最佳模型组合: {best_result['model_name']} + {best_result['optimizer_name']}优化器")
    print(f"测试准确率: {best_result['test_results']['accuracy']:.4f}")
    print(f"F1分数: {best_result['test_results']['f1']:.4f}")

    print("\n所有实验完成!")
    print("结果文件:")
    print("1. 实验报告: ./results/experiment_report.txt")
    print("2. 详细数据: ./results/experiment_results.csv")
    print("3. 综合对比图: ./results/comprehensive_comparison.png")
    print("4. 训练过程图: ./results/training_process_comparison.png")
    print("5. 各个实验的训练曲线图: ./results/[模型]_[优化器]_training_curves.png")


if __name__ == "__main__":
    main()
