all:
	nvcc test.cu -o test
clean:
	rm test
run: test
	env LD_PRELOAD=../build/hook_v2/libcuda_hook_v2.so ./test
