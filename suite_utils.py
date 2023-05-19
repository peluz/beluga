def get_test_info(suite):
    for test_name, test in suite.tests.items():
        print(f"Capability: {test.capability}")
        print(f"Test name: {test_name}")
        print(f"Number of test cases: {len(test.data)}")
        print(test.data[:3])
        if test.labels is None:
            print("No labels!")
        elif isinstance(test.labels, int):
            print("Label:")
            print(test.labels)
        else:
            print(test.labels[:3])
        print()