import os

select_methods = ['ensemble_sample_by_weight', 'random', 'uncertainty']
concretes = [[''], [''], ['entropy_sampling']]

# 在第k份数据上，用select_methods[i][j]采样方法来挑选样本
for k in range(1, 2):
    for i in range(len(select_methods)):
        for j in range(len(concretes[i])):
            py_cmd = "python main.py select_method='" + str(select_methods[i]) + "' concrete='" + str(
                concretes[i][j]) + "' data_path='data/origin_" + str(k) + "' out_path='data/out_" + str(k) + "'"
            print(py_cmd)
            try:
                os.system(py_cmd)
            except:
                print('当前程序出错，直接跳过运行下一个')
                continue
