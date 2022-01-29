

numbers = list(range(1,20+1))
print(numbers)

seq = [0, 1, 2, 3, 4, 5]
window_size = 4

#overlap_size == window size : means no overlap, the next segment will be created after the end of the previous
overlap_size = 2
step = window_size

for i in range(0, len(numbers) - window_size + 1, overlap_size):
    print(numbers[i: i + window_size])