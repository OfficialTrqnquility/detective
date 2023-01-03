
def main():
    import concurrent.futures

    def worker(x):
        """Thread worker function"""
        return x * x

    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = executor.map(worker, [1, 2, 3])
        print(list(result))

if __name__ == '__main__':
    main()