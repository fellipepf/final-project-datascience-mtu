from numpy import array

# load...
data = list()
n = 5000
for i in range(n):
    data.append([i + 1, (i + 1) * 10])
data = array(data)
print(data[:5, :])
print(data.shape)

# drop time
data = data[:, 1]
print(data.shape)


# split into samples (e.g. 5000/200 = 25)
samples = list()
length = 200
# step over the 5,000 in jumps of 200
for i in range(0,n,length):
	# grab from i to i + 200
	sample = data[i:i+length]
	samples.append(sample)
print(len(samples))


#Reshape Subsequences
#The LSTM needs data with the format of [samples, time steps and features].
#Here, we have 25 samples, 200 time steps per sample, and 1 feature.
#First, we need to convert our list of arrays into a 2D NumPy array of 25 x 200.

# convert list of arrays into 2d array
data = array(samples)
print(data.shape)

# reshape into [samples, timesteps, features]
# expect [25, 200, 1]
data = data.reshape((len(samples), length, 1))
print(data.shape)