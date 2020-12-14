all:
	nvcc test.cu -o test
clean:
	rm test
run: test
	env LD_PRELOAD=../build/libcuda_hooker.so ./test
