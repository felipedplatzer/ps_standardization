import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def my_fn(number, letter):
    print(f"Starting: {number}{letter}")
    time.sleep(2)
    return number**2, letter.upper()

if __name__ == "__main__":
    args = [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(my_fn, num, char) for num, char in args]

        for future in as_completed(futures):
            result = future.result()
            print("Result:", result)
