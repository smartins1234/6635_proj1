import os
import csv
from pprint import pprint
# import skimage
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from radar_chart import *
# from PIL import Image

OUT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images")

def get_hist(data, bin_count, min, max):
    bins = np.zeros(bin_count)
    bin_size = (max - min) / bin_count
    for val in data:
        my_bin = int(val // bin_size)
        if my_bin < 0:
            my_bin = 0
        elif my_bin >= bin_count:
            my_bin = bin_count-1
        bins[my_bin] += 1
    return bins


def heatify(data, bin_count, min, max):
    bins = np.zeros((100, 100))
    bin_size = (max - min) / bin_count
    for point in data:
        x = point[0]
        y = point[1]

        x_bin = int(x // bin_size)
        y_bin = int(y // bin_size)

        if x_bin < 0:
            x_bin = 0
        elif x_bin >= bin_count:
            x_bin = bin_count - 1
        elif y_bin < 0:
            y_bin = 0
        elif y_bin >= bin_count:
            y_bin = bin_count - 1

        bins[y_bin][x_bin] += 1

    return bins

def part1():
    set1 = np.random.uniform(0, 1, (100))
    set2 = np.random.normal(50, 16, (200))
    
    # box plots
    plt.boxplot(set1)
    plt.savefig(os.path.join(OUT_FOLDER, "uniform_boxplot.png"))
    plt.clf()

    plt.boxplot(set2)
    plt.savefig(os.path.join(OUT_FOLDER, "gaussian_boxplot.png"))
    plt.clf()

    # bar charts
    bins1 = get_hist(set1, 20, 0, 1)
    plt.bar([x / 100.0 for x in range(0, 100, 5)], bins1, align="edge", edgecolor="grey", width=1.0/20)
    plt.savefig(os.path.join(OUT_FOLDER, "uniform_bars.png"))
    plt.clf()

    bins2 = get_hist(set2, 20, 1, 100)
    plt.bar(range(1, 100, 5), bins2, align="edge", edgecolor="grey", width=100/20)
    plt.savefig(os.path.join(OUT_FOLDER, "gaussian_bars.png"))
    plt.clf()

    # write and read files
    with open("arrays.bin", "w") as file:
        set1.tofile(file)
        set2.tofile(file)

    read1 = None
    read2 = None
    with open("arrays.bin", "r") as file:
        read1 = sorted(np.fromfile(file, dtype=float, count=100))
        read2 = sorted(np.fromfile(file, dtype=float, count=200))
    
    if read1 is None or read2 is None:
        print("uh-oh spaghetti-oh")
        return
    
    # cumulative distributions
    cdf1 = np.cumsum(read1)
    cdf1 = cdf1/max(cdf1)
    plt.plot(read1, cdf1)
    plt.savefig(os.path.join(OUT_FOLDER, "uniform_cdf.png"))
    plt.clf()

    cdf2 = np.cumsum(read2)
    cdf2 = cdf2/max(cdf2)
    plt.plot(read2, cdf2)
    plt.savefig(os.path.join(OUT_FOLDER, "gaussian_cdf.png"))
    plt.clf()

    # scatter plots
    uniform = np.random.uniform(0, 1, (5000, 2))
    plt.scatter(uniform[:,0], uniform[:, 1], s=2)
    plt.savefig(os.path.join(OUT_FOLDER, "uniform_scatter.png"))
    plt.clf()

    gaussian = np.random.normal(0.5, 0.16, (5000, 2))
    plt.scatter(gaussian[:,0], gaussian[:, 1], s=2)
    plt.savefig(os.path.join(OUT_FOLDER, "gaussian_scatter.png"))
    plt.clf()

    bins = heatify(uniform, 100, 0, 1)
    plt.imshow(bins, cmap="pink")
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(OUT_FOLDER, "uniform_heatmap.png"))
    plt.clf()

    bins = heatify(gaussian, 100, 0, 1)
    plt.imshow(bins, cmap="pink")
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(OUT_FOLDER, "gaussian_heatmap.png"))
    plt.clf()


def part2():
    # # NOAA Data
    # data = None
    # with open("./data/NOAA-Temperatures.csv", mode ='r')as file:
    #     data = list(csv.reader(file))
    
    # data = np.array(data[5:], dtype=float)
    # colors = ["red" if val > 0 else "blue" for val in data[:, 1]]
    # xticks = [year for year in data[:, 0] if int(year)%20 == 0]
    # vals = [temp for temp in data[:, 1]]
    # start = round(min(vals), 1)
    # end = round(max(vals), 1)
    # yticks = [round(float(temp), 1) for temp in np.arange(start, end, 0.1)]

    # plt.bar(data[:, 0], data[:, 1], color=colors)
    # plt.xticks(xticks)
    # plt.yticks(yticks)
    # plt.xlabel("Year")
    # plt.ylabel("Degrees C +/- From Average")
    # plt.savefig(os.path.join(OUT_FOLDER, "NOAA_chart.png"))
    # plt.clf()

    # Cereal Data
    filepath = "./data/Breakfast-Cereals.xls"
    sheet = pd.read_excel(filepath, header=0, index_col=0)
    
    stats = ["Calories", "Protein", "Fat", "Sodium", "Fiber", "Carbohydrates", "Sugars", "Potassium"]
    data = {
        "Cheerios": [],
        "Kix": [],
        "Trix": []
    }
    for key in data.keys():
        for name in stats:
            data[key].append(sheet[name][key])
    maxes = [0, 0, 0, 0, 0, 0, 0, 0]
    for key in data.keys():
        for i in range(len(stats)):
            temp = data[key][i]
            if temp > maxes[i]:
                maxes[i] = temp

    for key in data.keys():
        for i in range(len(stats)):
            data[key][i] /= maxes[i]

    stats_max = []
    for i in range(len(stats)):
        stats_max.append("{} (Max: {})".format(stats[i], maxes[i]))

    theta = radar_factory(8, frame='polygon')
    colors = ["r", "g", "b"]
    ax = plt.subplot(projection="radar")
    ax.set_yticklabels([])
    for d, color in zip(data.keys(), colors):
        ax.plot(theta, data[d], color=color)
        ax.fill(theta, data[d], facecolor=color, alpha=0.25, label="_nolegend_")
    ax.set_varlabels(stats_max)

    labels = data.keys()
    legend = ax.legend(labels, loc=(0.9, .95), labelspacing=0.1, fontsize='small')

    plt.savefig(os.path.join(OUT_FOLDER, "cereal_chart.png"))
    plt.clf()

    

def part4():
    pass

if __name__ == "__main__":
    # Part 1
    # part1()

    # Part 2
    part2()
