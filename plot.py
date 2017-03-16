import matplotlib.pyplot as plt

acc = [0.5544, 0.6716, 0.6985, 0.7166, 0.7317, 0.7477, 0.7604, 0.7737, 0.7866, 0.7964, 0.8051,
       0.8131, 0.8199, 0.8260
       ]
val_acc = [0.6490, 0.6829, 0.6948, 0.7061, 0.7049, 0.7069, 0.7006, 0.7108, 0.7097, 0.7024, 0.7030,
           0.7039, 0.7007, 0.6965]

plt.ylabel('Acc.')
plt.xlabel('Epochs')

acc_line, = plt.plot(acc, label="Training Acc.", linewidth=2)
val_acc_line, = plt.plot(val_acc, label="Test Acc.", linewidth=2)
first_legend = plt.legend(handles=[acc_line, val_acc_line], loc=4)

# closest point between 2 line
min_diff_idx = 0
max_acc_idx = 0
for i in range(0, len(acc)):
    diff = abs(acc[i] - val_acc[i])
    if abs(acc[i] - val_acc[i]) < abs(acc[min_diff_idx] - val_acc[min_diff_idx]):
        min_diff_idx = i
    if val_acc[i] > val_acc[max_acc_idx]:
        max_acc_idx = i

plt.annotate(str(val_acc[min_diff_idx]), xy=(min_diff_idx, val_acc[min_diff_idx]),
             xytext=(min_diff_idx + 2, val_acc[min_diff_idx] - 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )
plt.annotate(str(val_acc[max_acc_idx]), xy=(max_acc_idx, val_acc[max_acc_idx]),
             xytext=(max_acc_idx - 2, val_acc[max_acc_idx] - 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

plt.show()
