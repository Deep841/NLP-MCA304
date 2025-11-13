import time
import tracemalloc

def student_code():
   # 👇 write your code below this line
   
   n = int(input())
   nums = list(map(int, input().split()))
   k = int(input())
   limit = int(input())
   
   max_product = -1
   
   for mask in range(1, 1 << n):
       subseq = []
       for i in range(n):
           if mask & (1 << i):
               subseq.append(nums[i])
       
       alt_sum = 0
       for idx, val in enumerate(subseq):
           if idx % 2 == 0:
               alt_sum += val
           else:
               alt_sum -= val
       
       if alt_sum != k:
           continue
       
       product = 1
       for val in subseq:
           product *= val
           if product > limit:  
               break
       
       if product <= limit:
           max_product = max(max_product, product)
   
   print(max_product)
   
   return 0


def analyze():
    start_time = time.perf_counter()
    tracemalloc.start()

    student_code()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.perf_counter()

    print(f"Execution Time: {(end_time - start_time):.6f} seconds")
    print(f"Current Memory Usage: {current / 10**6:.6f} MB")
    print(f"Peak Memory Usage: {peak / 10**6:.6f} MB")

if __name__ == "__main__":
    analyze()
