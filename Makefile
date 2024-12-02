all:
	nvcc multithread.cu -o multithread
	nvcc thrust.cu -o thrust
	nvcc singlethread.cu -o singlethread
