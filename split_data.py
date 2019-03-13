import random as rand
import HP

pr_test_file = open(HP.TEST_PATH + "/prediction_test.tsv", "w+")
mc_test_file = open(HP.TEST_PATH + "/multi_choice_test.tsv", "w+")
pr_train_file = open(HP.TRAIN_PATH + "/prediction_train.tsv", "w+")
mc_train_file = open(HP.TRAIN_PATH + "/multi_choice_train.tsv", "w+")
prediction_data = open(HP.DATA_PATH + "/prediction_data_set.tsv", "r")
multi_choice_data = open(HP.DATA_PATH + "/multi_choice_data_set.tsv", "r")

mc_line = multi_choice_data.readline()
pr_line = prediction_data.readline()

pr_test_file.write(pr_line)
pr_train_file.write(pr_line)
mc_test_file.write(mc_line)
mc_train_file.write(mc_line)

mc_line = multi_choice_data.readline()
pr_line = prediction_data.readline()

while mc_line or pr_line:
    if mc_line:
        random = rand.random()
        if random < 0.02:
            mc_test_file.write(mc_line)
        else:
            mc_train_file.write(mc_line)
        mc_line = multi_choice_data.readline()
    if pr_line:
        random = rand.random()
        if random < 0.02:
            pr_test_file.write(pr_line)
        else:
            pr_train_file.write(pr_line)
        pr_line = prediction_data.readline()
