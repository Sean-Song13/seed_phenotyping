import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    data = {}
    skip = True
    with open("0428result480.csv") as file:
        for line in file:
            if skip:
                skip = False
                continue
            line = line.strip()
            numbers = line.split(',')
            cate = numbers[3]
            if cate not in data.keys():
                data[cate] = []
            score = float(numbers[4])
            peek = float(numbers[6])
            peek_diff = float(numbers[7])
            curve_diff = float(numbers[8])
            hausdorff = float(numbers[9])
            data[cate].append(score)


    print("# 2row mean: {0:.6f} std: {1:.6f} number:{2}".format(np.mean(data["2row"]), np.std(data["2row"]), len(data["2row"])))
    print("# 6row mean: {0:.6f} std: {1:.6f} number:{2}".format(np.mean(data["6row"]), np.std(data["6row"]), len(data["6row"])))

    sns.set_theme()
    sns.distplot(data["2row"], label="2row",bins=10, kde=True)
    sns.distplot(data["6row"], label="6row",bins=10, kde=True)
    plt.title('score')
    plt.legend()
    plt.show()
