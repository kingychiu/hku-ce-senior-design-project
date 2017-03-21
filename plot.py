import matplotlib.pyplot as plt

acc = [
0.839587939237,0.873963968208,0.874029822668,0.898903993607,0.906326732212,0.910390893271,0.920033868013,0.928867773653,0.929253492641,0.936535114544,0.94089091679,0.946676701637,0.948370102077]
val_acc = [
0.833128457278,0.862015980327,0.85762577925,0.876393888835,0.880191412761,0.880827991932,0.884603564833,0.887896215636,0.887435244524,0.88932303099,0.891232768471,0.891649837552,0.890310826241]
plt.ylabel('Acc.')
plt.xlabel('Epochs')

acc_line, = plt.plot(acc, label="Train Acc.", linewidth=2)
val_acc_line, = plt.plot(val_acc, label="Test Acc.", linewidth=2)
first_legend = plt.legend(handles=[acc_line, val_acc_line], loc=4)

# closest point between 2 line
min_diff_idx = 0
max_acc_idx = 0
max_acc_idx2 = 0
for i in range(0, len(acc)):
    diff = abs(acc[i] - val_acc[i])
    if abs(acc[i] - val_acc[i]) < abs(acc[min_diff_idx] - val_acc[min_diff_idx]):
        min_diff_idx = i
    if val_acc[i] > val_acc[max_acc_idx]:
        max_acc_idx = i
    if acc[i] > acc[max_acc_idx]:
        max_acc_idx = i

# plt.annotate(str(val_acc[min_diff_idx]), xy=(min_diff_idx, val_acc[min_diff_idx]),
#              xytext=(min_diff_idx + 2, val_acc[min_diff_idx] - 0.01),
#              arrowprops=dict(facecolor='black', shrink=0.05),
#              )
plt.annotate(str(val_acc[max_acc_idx]), xy=(max_acc_idx, val_acc[max_acc_idx]),
             xytext=(max_acc_idx - 2, val_acc[max_acc_idx] - 0.01),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )
# plt.annotate(str(acc[max_acc_idx]), xy=(max_acc_idx, acc[max_acc_idx]),
#              xytext=(max_acc_idx - 10, acc[max_acc_idx] - 0.1),
#              arrowprops=dict(facecolor='black', shrink=0.05),
#              )

plt.show()
