[raw_images, raw_labels] = readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', 60000, 0);
[images, labels] = process_data(raw_images, raw_labels);


[raw_images_test, raw_labels_test] = readMNIST('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 10000, 0);
[images_test, labels_test] = process_data(raw_images_test, raw_labels_test);