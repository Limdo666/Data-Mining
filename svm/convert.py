from tqdm import tqdm

dest = open('test.txt','w')
with open('svmtest.txt') as file:
    for line in tqdm(file):
        words = line.strip().split()
        label = words[0]
        values = {}
        for kv in words[1:]:
            k,v = kv.split(':')
            k = int(k)+1
            values[k] = v
        s = label + ' ' + ' '.join(str(k)+':'+values[k] for k in sorted(values)) + '\n'
        dest.write(s)
dest.close()
