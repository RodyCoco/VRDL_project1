import numpy as np

submission = []
with open('answer.txt',"r") as f:
    x = f.readlines()
    for idx,item in enumerate(x):
        x[idx] = (x[idx][2:10] +" "+ x[idx][16:]).strip()
        submission.append(x[idx])

np.savetxt('test.txt', submission, fmt='%s')
