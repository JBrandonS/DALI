import pickle
print(pickle.__version__)
data = {"tester": 1, "test2": 2}
with open('pkl_test.pkl', 'wb') as file:
    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)