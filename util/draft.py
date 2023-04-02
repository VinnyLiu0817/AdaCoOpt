import numpy as np
p = [500 for i in range(3)]
q = list(range(10,20))
b=[];
a = [[(j+1)*100 for j in range(20)] for i in range(3)]

round_num = 5
cli_num = 3

def get_random_data_arrive(cli_num, round_num, total_k, data_total):
    arrive_round = []
    arrive_num = []
    for i in range(cli_num):
        round = list(np.random.randint(1, total_k, round_num))
        round.sort()
        arrive_round.append(round)

        m = list(np.random.randint(1,data_total,round_num))
        sum_m = sum(m)
        if sum(m) >= data_total:
            for i in range(len(m)):
                m[i] = int(m[i] / sum_m * data_total)

        while sum(m) < data_total:
            idx = int(np.random.randint(0,len(m),1))
            m[idx] += 1

        arrive_num.append(m)

    # print(arrive_num)
    # print(arrive_round)
    return [arrive_round, arrive_num]

[arrive_round, arrive_num] = get_random_data_arrive(100, 5, 3000, 40000)
print(arrive_num)
print(arrive_round)





