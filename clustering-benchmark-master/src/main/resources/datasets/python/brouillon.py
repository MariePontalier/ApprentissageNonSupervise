import matplotlib.pyplot as plt

x = ["KMean", "KMedoids", "AggloDist", "AggloK"]
x1 = [172, 118, 0, 573]
y1 = [23243, 8579, 0, 102582]
x6 = [61, 219, 1008, 209]
y6 = [27959, 18239, 135168, 256448]
x7 = [30, 7, 14, 7]
y7 = [10031, 718, 15121, 5240]

# plt.plot(x, x1, label="x1")
plt.plot(x, y1, label="x1")
# plt.plot(x, x6, label="zz1")
plt.plot(x, y6, label="zz1")
# plt.plot(x, x7, label="zz2")
plt.plot(x, y7, label="zz2")
plt.legend()
plt.title("Time of all iterations")
plt.show()
