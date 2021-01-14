add_library(memory
          ischeduler.cpp
          naive_scheduler.cpp)

add_executable(naive_scheduler_test naive_scheduler_test.cpp)
target_link_libraries(naive_scheduler_test PUBLIC memory th_catch2_test_main)
add_test(NAME naive_scheduler_test COMMAND naive_scheduler_test)
