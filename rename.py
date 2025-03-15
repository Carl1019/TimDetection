import os

folder_path = r"D:\SYDE770 Project"  # 你的文件夹路径
os.chdir(folder_path)

for i, filename in enumerate(os.listdir(folder_path), start=1):
    ext = os.path.splitext(filename)[1]  # 获取文件扩展名
    new_name = f"img_{i:04d}{ext}"  # 例如 img_0001.jpg
    os.rename(filename, new_name)

print("文件重命名完成！")
