# 此程序返回采用teacher forcing的概率
# 直接return 0来取消Teacher Forcing (效果肯定不好)
import numpy as np

def inverse_sigmoid(x):
    return 1 - 1/(1 + np.exp(5-10*x))

def exponential(x):
    return np.exp(-5*x)

def schedule_sampling(total_steps, train_type = "teacher_force"):
    num_steps = 12000        
    if train_type == "teacher_force":
        return 1
    elif train_type == "linear_decay":
        return 1 - (total_steps / num_steps)   
    elif train_type == "inverse_sigmoid_decay":
        x = total_steps / num_steps
        return inverse_sigmoid(x)
    elif train_type == "exponential_decay":
        x = total_steps / num_steps
        return exponential(x)
    
    
    # 验证decay曲线
    # import math
    # import matplotlib.pyplot as plt
    # x = np.arange(0, 1, 0.01)
    # y = []
    # for t in x:
    #     y_1 = exponential(t)
    #     y.append(y_1)
    # plt.plot(x, y, label="sigmoid")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.ylim(0, 1)
    # plt.legend()
    # plt.show()
        
