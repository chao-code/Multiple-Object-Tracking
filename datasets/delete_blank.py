"""
删除.txt文件末尾的空行
"""
import os


# 定义一个函数来删除文本文件中的空行
def remove_blank_lines(file):
    # 打开输入和输出文件
    input_file = open(file, "r")
    output_file = open(file + "_new", "w")

    # 读取输入文件中的所有行，并去掉两端的空格
    lines = [line.strip() for line in input_file]
    # 过滤掉空行，并获取非空行的数量
    lines = list(filter(lambda x: x != "", lines))
    num_lines = len(lines)

    # 遍历非空行
    for i in range(num_lines):
        # 如果是最后一行，就不加换行符'\n'
        if i == num_lines - 1:
            output_file.write(lines[i])
        # 否则，加上换行符'\n'
        else:
            output_file.write(lines[i] + "\n")

    # 关闭输入和输出文件
    input_file.close()
    output_file.close()

    # 删除原来的文件
    os.remove(file)
    # 把新的文件重命名为原来的文件名
    os.rename(file + "_new", file)


# 指定要处理的文件夹路径
folder_path = "./labels/train"

# 遍历指定文件夹下的所有.txt文件
for file in os.listdir(folder_path):
    # 获取.txt文件的完整路径
    file_path = os.path.join(folder_path, file)
    # 调用函数来删除文本文件中的空行
    remove_blank_lines(file_path)