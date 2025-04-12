import numpy as np
import pickle
import cv2
import os

def autocorrelation(x, nlags):
    result = np.correlate(x, x, mode='full')
    return result[len(result) // 2: len(result) // 2 + nlags + 1]


def lpc_manual(signal, order):
    r = autocorrelation(signal, order)
    phi = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            phi[i, j] = r[np.abs(i - j)]
    a = np.linalg.solve(phi, -r[1:])
    a = np.insert(a, 0, 1)
    return a


def mem(signal, order):
    # 检查输入信号是否为空
    if len(signal) == 0:
        raise ValueError("输入信号不能为空")
    # 检查模型阶数是否合法
    if order < 0 or order >= len(signal):
        raise ValueError("模型阶数必须为非负且小于信号长度")

    a = lpc_manual(signal, order)
    spectrum = np.abs(np.fft.fft(1 / a, len(signal)))
    return spectrum


def estimate_gait_period(gait_signal, order=10):
    """
    估计步态周期
    :param gait_signal: 步态信号（如轮廓下半部分大小随时间的变化序列），一维numpy数组
    :param order: 最大熵谱估计的阶数，默认为10
    :return: 估计的步态周期
    """
    spectrum = mem(gait_signal, order)
    frequencies = np.fft.fftfreq(len(spectrum))
    max_freq_index = np.argmax(spectrum)
    max_frequency = frequencies[max_freq_index]
    if max_frequency == 0:
        # 这里可以返回一个默认值，比如序列的长度
        return len(gait_signal)
        # 或者也可以选择跳过当前序列的处理，返回None
        # return None
    gait_period = 1 / abs(max_frequency)
    return gait_period


def compute_gei(silhouette_sequence, gait_period):
    """
    计算步态能量图像（GEI）
    :param silhouette_sequence: 二值化轮廓图像序列，是一个形状为 (帧数, 高度, 宽度) 的NumPy数组
    :param gait_period: 步态周期
    :return: GEI图像，是一个形状为 (高度, 宽度) 的NumPy数组
    """
    num_frames, height, width = silhouette_sequence.shape
    num_cycles = int(num_frames // gait_period)
    gei = np.zeros((height, width), dtype=np.float32)
    for cycle in range(num_cycles):
        start_frame = int(cycle * gait_period)
        end_frame = int(start_frame + gait_period)
        cycle_frames = silhouette_sequence[start_frame:end_frame]
        for frame in cycle_frames:
            gei += frame
    gei /= num_cycles * gait_period * 255
    # 确保GEI的值域在0 - 1之间，再乘以255转换为uint8
    gei = np.clip(gei, 0, 1)
    gei = (gei*255).astype(np.uint8)
    return gei


# 遍历pretreatment中处理后生成的所有pickle文件
output_path = r'E:\HKU\HKU_Sem2\COMP 7404\Project\GaitDatasetB-silh\CASIA-B-processed'
gei_output_path = r'E:\HKU\HKU_Sem2\COMP 7404\Project\GaitDatasetB-silh\GEI'


for root, dirs, files in os.walk(output_path):
    for file in files:
        if file.endswith('.pkl'):
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                preprocessed_silhouette_sequence = pickle.load(f)

            # 提取用于估计周期的信号，轮廓下半部分大小
            height, width = preprocessed_silhouette_sequence[0].shape
            lower_half_sizes = np.array([np.sum(frame[height // 2:, :]) for frame in preprocessed_silhouette_sequence])

            # 检查gait_signal的长度，调整order的值
            order = 10
            if len(lower_half_sizes) <= order:
                order = len(lower_half_sizes) - 1 if len(lower_half_sizes) > 0 else 0

            gait_period = estimate_gait_period(lower_half_sizes, order)
            print(f"文件 {file_path} 的估计步态周期: {gait_period}")

            if gait_period is not None:
                gei_image = compute_gei(preprocessed_silhouette_sequence, gait_period)
                print(f"文件 {file_path} 的GEI图像计算完成")
            # 构建对应的输出目录
                relative_path = os.path.relpath(root, output_path)
                output_subdir = os.path.join(gei_output_path, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

            # 保存GEI图像
                output_file_name = os.path.splitext(os.path.basename(file_path))[0] + '_GEI.png'
                output_file_path = os.path.join(output_subdir, output_file_name)
                cv2.imwrite(output_file_path, gei_image)
                print(f"文件 {file_path} 的GEI图像已保存到 {output_file_path}")
            else:
                print(f"文件 {file_path} 的步态周期估计失败，跳过该文件")