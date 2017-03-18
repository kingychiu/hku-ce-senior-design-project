import matplotlib.pyplot as plt

acc = [0.802417329366,0.825154222565,0.839628285518,0.849852409144,0.855087228174,0.860889270172,0.861834641806,0.868598987463,0.868641317536,0.87270218254,0.874034168834,0.878388522343,0.879189971727,0.881839834291,0.884263936474,0.882370371205]
val_acc = [0.796402138704,0.816544630865,0.828074380391,0.835653330529,0.840591829752,0.843719545926,0.845306450343,0.849020201756,0.847913977933,0.852760291834,0.853135617775,0.854946400826,0.855374404092,0.857191771814,0.858627228914,0.857040324494]
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

# plt.annotate(str(val_acc[min_diff_idx]), xy=(min_diff_idx, val_acc[min_diff_idx]),
#              xytext=(min_diff_idx + 2, val_acc[min_diff_idx] - 0.01),
#              arrowprops=dict(facecolor='black', shrink=0.05),
#              )
plt.annotate(str(val_acc[max_acc_idx]), xy=(max_acc_idx, val_acc[max_acc_idx]),
             xytext=(max_acc_idx - 1, val_acc[max_acc_idx] - 0.01),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

plt.annotate(str(acc[max_acc_idx]), xy=(max_acc_idx, acc[max_acc_idx]),
             xytext=(max_acc_idx - 1, acc[max_acc_idx] - 0.01),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

plt.show()
