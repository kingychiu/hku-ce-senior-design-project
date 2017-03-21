import matplotlib.pyplot as plt

acc = [0.336476471652,0.401277339282,0.512592637227,0.590317527271,0.656052298608,0.701331565953,0.741097788764,0.756190877866,0.775772730019,0.791769596922,0.809483641618,0.834217027188,0.837380249439,0.85162981262,0.85006326446,0.868259323989,0.874465264821,0.893534976215,0.897210339214,0.915105139487,0.915707658003,0.923209013677,0.921642465499,0.932517924919,0.931915406402,0.942851117665,0.939266132426,0.943242754715,0.953304814116,0.957883954924,0.964511658734,0.963788636501,0.971922636621,0.968066518045,0.972103392172,0.96887991805,0.978731095974,0.979664999702,0.982828221961,0.978791347827,0.982165451588,0.986051696082,0.985208170151,0.98737723685,0.986593962764,0.989576429475]
val_acc = [0.335371854355,0.404962744275,0.510895543405,0.578658793741,0.631871221715,0.66294109378,0.693870378189,0.704062983278,0.709827077191,0.711584422896,0.724237312048,0.731899339204,0.73471109245,0.741951356679,0.735835793701,0.741881062851,0.741107830774,0.752073667982,0.756923942128,0.762758329785,0.756221003762,0.759735695214,0.753971601302,0.764093912521,0.756713060602,0.764023618735,0.755236890251,0.760579221195,0.762688036074,0.767468016358,0.767397722446,0.771896527451,0.773583579328,0.76817095464,0.772318290421,0.761844510136,0.772810347218,0.773442991705,0.7735132855,0.765359201429,0.77203711515,0.7735132855,0.767468016325,0.771545058428,0.774989455968,0.770560944766]
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
             xytext=(max_acc_idx - 20, val_acc[max_acc_idx] - 0.2),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )
# plt.annotate(str(acc[max_acc_idx]), xy=(max_acc_idx, acc[max_acc_idx]),
#              xytext=(max_acc_idx - 10, acc[max_acc_idx] - 0.1),
#              arrowprops=dict(facecolor='black', shrink=0.05),
#              )

plt.show()
