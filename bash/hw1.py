import sys

testing_data_path = sys.argv[1]
output_file_path  = sys.argv[2]

with open(testing_data_path, 'r') as f:
    ans = []
    for line in f:
        x, y = [int(i) for i in line.split(',')]
        ans.append(x+y)

with open(output_file_path, 'w') as f:
    for a in ans:
        f.write(f'{a}\n')
