import matplotlib.pyplot as plt

l1 = [0.368677635589,0.519669578177,0.564019229499,0.577290270193,0.619202383391,0.628952535749,0.629291082701,0.644796533303,0.656307129823,0.666124991561,0.671000067734,0.669307332951,0.664229128604,0.677771006863,0.681088767036,0.685286749296,0.683729433297,0.688672218861,0.693411876251,0.69124517573,0.679260613471,0.700047396598,0.702823481641,0.699911977816,0.704787053993,0.700724490511,0.699302593294,0.701469293819,0.700589071732,0.699708849642,0.703162028602,0.701604712598,0.704583925815,0.69679734582,0.706276660601
]
l2 = [0.76441232147,0.791946308725,0.813457236286,0.829375322664,0.841851660664,0.845809671323,0.848907244891,0.849681638272,0.86129753916,0.860523145758,0.853639648944,0.863706763046,0.85923249011,0.868611254528,0.86534159354,0.874978489083,0.871278609534,0.865083462409,0.871794871795,0.872655308917,0.87781793153,0.875322663934,0.859146446405,0.872053002936,0.876441232166,0.877043538138,0.873601789719,0.876441232156,0.871536740664,0.875236620203,0.879022543473,0.876871450697,0.87790397524,0.875838926195,0.877817931519]
l3 = []
plt.ylabel('Acc.')
plt.xlabel('Epochs')

l1_line, = plt.plot(l1, label="AG1", linewidth=2)
l2_line, = plt.plot(l2, label="AG2", linewidth=2)
# l3_line, = plt.plot(l3, label="BBC", linewidth=2)
first_legend = plt.legend(handles=[l1_line, l2_line], loc=4)

# closest point between 2 line
min_diff_idx = 0
max_acc_idx = 0
max_acc_idx2 = 0
max_acc_idx3 = 0
for i in range(0, len(l1)):
    if l1[i] > l1[max_acc_idx]:
        max_acc_idx = i
    if l2[i] > l2[max_acc_idx2]:
        max_acc_idx2 = i
    # if l3[i] > l3[max_acc_idx3]:
    #     max_acc_idx3 = i

plt.annotate(str(l1[max_acc_idx]), xy=(max_acc_idx, l1[max_acc_idx]),
             xytext=(max_acc_idx - 5, l1[max_acc_idx] - 0.05),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

plt.annotate(str(l2[max_acc_idx2]), xy=(max_acc_idx2, l2[max_acc_idx2]),
             xytext=(max_acc_idx2 - 5, l2[max_acc_idx2] - 0.05),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

# plt.annotate(str(l3[max_acc_idx3]), xy=(max_acc_idx3, l3[max_acc_idx3]),
#              xytext=(max_acc_idx3 - 5, l3[max_acc_idx3] - 0.05),
#              arrowprops=dict(facecolor='black', shrink=0.05),
#              )
plt.show()
