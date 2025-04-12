import os
import pickle
import cv2
from pathlib import Path

def pickle_to_png(pickle_dir, output_dir):
    """
    将pickle文件转换为PNG图像文件。

    Args:
        pickle_dir (str): pickle文件所在的目录。
        output_dir (str): 输出PNG图像文件的目录。
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 遍历pickle目录下的所有文件和子目录
    for root, dirs, files in os.walk(pickle_dir):
        for file in files:
            if file.endswith('.pkl'):
                # 构建pickle文件的完整路径
                pickle_path = os.path.join(root, file)
                # 构建对应的输出目录
                relative_path = os.path.relpath(root, pickle_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                Path(output_subdir).mkdir(parents=True, exist_ok=True)

                # 读取pickle文件
                with open(pickle_path, 'rb') as f:
                    images = pickle.load(f)

                # 遍历pickle文件中的所有图像
                for i, image in enumerate(images):
                    # 构建输出PNG图像文件的路径
                    img_name = f"{os.path.splitext(file)[0]}_{i}.png"
                    img_path = os.path.join(output_subdir, img_name)
                    # 保存图像为PNG文件
                    cv2.imwrite(img_path, image)
                    print(f"Saved {img_path}")

if __name__ == '__main__':
    # 指定pickle文件所在的目录
    pickle_dir = 'E:\HKU\HKU_Sem2\COMP 7404\Project\GaitDatasetB-silh\CASIA-B-processed'
    # 指定输出PNG图像文件的目录
    output_dir = 'E:\HKU\HKU_Sem2\COMP 7404\Project\GaitDatasetB-silh\PNG-Output'
    # 调用函数进行转换
    pickle_to_png(pickle_dir, output_dir)