from Model import RandomForestModel

## Test 1
model = RandomForestModel(X_train=[[1, 2, 3], [11, 12, 13]], y_train = [0, 1], X_test=[[3, 4, 1],[14, 11, 17]], n_estimators=1)

model.fit()

out = list(model.predict())
desired_out = [0, 1]

print("Desired out:" + "\t" + str(desired_out))
print("Actual out:" + "\t" + str(out))
for index in range(0, len(out)):
    if out[index]!=desired_out[index]:
        print("Test 1 failed")
        exit(0)
print("Test 1 passed")
