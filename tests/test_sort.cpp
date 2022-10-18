#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>
#include "algo_and_ds/sort_helper.hpp"
#include "algo_and_ds/select_sort.hpp"
#include "algo_and_ds/insert_sort.hpp"
#include "algo_and_ds/merge_sort.hpp"
#include "algo_and_ds/quick_sort.hpp"
#include "algo_and_ds/heap_sort.hpp"

using namespace algo_and_ds::sort;

int main(int argc, char *argv[]) {

  // const int numElements = 209399;
  const int numElements = 100000;
  using myArray = std::array<int, numElements>;

  myArray randomArr1;
  generateRandomArray<myArray>(randomArr1, 10000);
  myArray randomArr2 = randomArr1;
  myArray randomArr4 = randomArr1;
  testSort<myArray>("Merge bu sort", merge_sort_bu, randomArr1);
  testSort<myArray>("Quick sort 3 way", quick_sort_3way, randomArr4);
  testSort<myArray>("Heap sort", heap_sort, randomArr2);

  myArray nearlyOrderArray1;
  generateNearlyOrderedArray<myArray>(nearlyOrderArray1, 10);
  myArray nearlyOrderArray2 = nearlyOrderArray1;
  myArray nearlyOrderArray4 = nearlyOrderArray1;
  testSort<myArray>("Merge bu sort", merge_sort_bu, nearlyOrderArray1);
  testSort<myArray>("Quick sort 3 way", quick_sort_3way, nearlyOrderArray4);
  testSort<myArray>("Heap sort", heap_sort, nearlyOrderArray2);

  myArray repeatArr1;
  generateRandomArray<myArray>(repeatArr1, 10);
  myArray repeatArr2 = repeatArr1;
  myArray repeatArr4 = repeatArr1;
  testSort<myArray>("Merge bu sort", merge_sort_bu, repeatArr1);
  testSort<myArray>("Quick sort 3 way", quick_sort_3way, repeatArr4);
  testSort<myArray>("Heap sort", heap_sort, repeatArr2);

  return 0;
}