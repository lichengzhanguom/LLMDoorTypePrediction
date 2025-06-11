#It is for calculating the accuracy of the prediction of GPT-4.1
# result.txt contains the results of GPT-4.1 and gt.txt is labeled manually by us, which is the ground truth
f = open("result.txt", 'r')
lines = f.readlines()
f.close()

result = dict()

for line in lines:
    name = line.split(':')[0].strip()
    count = line.split(":")[1].strip()
    result[name] = int(count)

f = open("gt.txt", 'r')
lines = f.readlines()
f.close()

gt = dict()

for line in lines:
    name = line.split(":")[0].strip()
    count = line.split(":")[1].strip()
    gt[name] = int(count)

total = len(list(gt.keys()))
acc = 0

for name in list(gt.keys()):
    ground_truth = gt[name]
    prediction = result[name]
    if ground_truth == prediction:
        acc += 1
print(acc/total)